# Optimizer

**Dosya:** `src/AdamOptimizer.cs`

Gradyanlar hesaplandı. Şimdi ağırlıkları güncelleyeceğiz. Bu iş göründüğünden çok daha inceliklidir.

---

## Naif yöntem: düz gradyan inişi

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

```csharp
p.Data[i] -= learningRate * p.Grad[i];
```

Çalışır ama üç sorunu vardır:

!!! failure "Sorun 1: Gürültü"
    Her adımda rastgele 8 örnek seçiliyor. Gradyan yönü adımdan adıma zıplar. Model sürekli yön değiştirip zaman kaybeder.

!!! failure "Sorun 2: Ölçek farkı"
    Bazı ağırlıkların gradyanı 0,0001 mertebesinde, bazılarınınki 10. Aynı learning rate ikisine de uygulanınca biri hiç hareket etmez, diğeri fırlar.

!!! failure "Sorun 3: Dar vadiler"
    Loss yüzeyi bir yönde dik, diğerinde düz olabilir. Düz gradyan inişi dik yönde zıplayıp dururken düz yönde ilerleyemez.

**Adam** üçünü de çözer.

---

## Adam: iki hareketli ortalama

Adam = **Ada**ptive **M**oment estimation. Her ağırlık için iki ek sayı tutar:

```csharp
private readonly double[][] _m;   // birinci moment: gradyanın ortalaması
private readonly double[][] _v;   // ikinci moment: gradyan karesinin ortalaması
```

### Birinci moment (momentum)

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$\beta_1 = 0.9$ olduğu için: yeni değer %90 eski + %10 yeni gradyan. Bu **üssel hareketli ortalama**dır.

Etkisi: Gürültü yumuşar. Gradyan sürekli aynı yönü gösteriyorsa momentum birikir ve hız artar — tıpkı yokuş aşağı yuvarlanan bir top gibi.

### İkinci moment (adaptif ölçek)

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$\beta_2 = 0.999$ — çok daha yavaş değişir. Gradyanın **büyüklüğünü** izler.

### Güncelleme

$$w \leftarrow w - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

`m`'yi `√v`'ye bölmek, her ağırlığa **kendi adım boyunu** verir:

- Gradyanı sürekli büyük olan ağırlık → büyük `v` → küçük adım
- Gradyanı hep küçük olan ağırlık → küçük `v` → büyük adım

Sonuç: tüm ağırlıklar benzer hızda ilerler.

---

## Bias düzeltmesi

```csharp
double bc1 = 1.0 - Math.Pow(_beta1, _step);
double bc2 = 1.0 - Math.Pow(_beta2, _step);
...
p.Data[i] -= (float)(LearningRate * (m[i] / bc1) / (Math.Sqrt(v[i] / bc2) + _eps));
```

`m` ve `v` sıfırdan başlar. İlk adımlarda gerçek değerin çok altında kalırlar — sıfıra doğru **yanlıdırlar** (biased).

| Adım | `bc1` | Düzeltme etkisi |
|---|---|---|
| 1 | 0.100 | 10× büyütür |
| 10 | 0.651 | 1,5× büyütür |
| 100 | 0.99997 | Neredeyse etkisiz |

Bu düzeltme olmadan eğitimin ilk adımları çok yavaş olurdu.

---

## Tam kod

```csharp
public void Step()
{
    _step++;
    double bc1 = 1.0 - Math.Pow(_beta1, _step);
    double bc2 = 1.0 - Math.Pow(_beta2, _step);

    for (int pi = 0; pi < _parameters.Length; pi++)
    {
        var p = _parameters[pi];
        var m = _m[pi];
        var v = _v[pi];

        for (int i = 0; i < p.Length; i++)
        {
            double g = p.Grad[i];
            m[i] = (_beta1 * m[i]) + ((1.0 - _beta1) * g);
            v[i] = (_beta2 * v[i]) + ((1.0 - _beta2) * g * g);
            p.Data[i] -= (float)(LearningRate * (m[i] / bc1) / (Math.Sqrt(v[i] / bc2) + _eps));
        }
    }
}
```

!!! note "Karma hassasiyet"
    Parametreler `float`, optimizer durumu `double`. Neden? `v` değerleri çok küçük olabilir ($10^{-10}$ mertebesinde) ve `float` hassasiyeti yetersiz kalır. Bellek maliyeti sadece 5,6 MB — göz ardı edilebilir.

    Bu, büyük ölçekli eğitimde de kullanılan standart bir tekniktir.

---

## Gradient clipping: patlamaya karşı sigorta

```csharp
public void ClipGradients(double maxNorm)
{
    double total = 0.0;
    foreach (var p in _parameters)
        foreach (float g in p.Grad)
            total += (double)g * g;

    double norm = Math.Sqrt(total);
    if (norm <= maxNorm || norm == 0.0)
        return;

    float scale = (float)(maxNorm / norm);
    foreach (var p in _parameters)
        for (int i = 0; i < p.Grad.Length; i++)
            p.Grad[i] *= scale;
}
```

Tüm gradyanların **global normu** hesaplanır:

$$\|g\| = \sqrt{\sum_i g_i^2}$$

Eşiği aşarsa hepsi orantılı olarak küçültülür. Kritik nokta: **yön korunur**, sadece büyüklük kısılır.

!!! danger "Neden gerekli?"
    Zaman zaman zor bir batch denk gelir ve gradyanlar patlar. Tek bir kötü adım tüm ağırlıkları bozabilir ve eğitim asla toparlanamaz — loss `NaN` olur.

    Clipping bu felaketi engeller. Dil modellerinde neredeyse her zaman kullanılır.

---

## Learning rate programı

Sabit learning rate iyi fikir değildir. Bu projede iki aşamalı bir program var:

```csharp
private static double CosineSchedule(int step)
{
    const int Warmup = 100;
    if (step < Warmup)
        return LearningRate * step / Warmup;

    double progress = (double)(step - Warmup) / Math.Max(1, TrainingSteps - Warmup);
    return 0.1 * LearningRate + (0.9 * LearningRate * 0.5 * (1.0 + Math.Cos(Math.PI * progress)));
}
```

### Aşama 1: Isınma (warmup)

İlk 100 adımda LR sıfırdan `3e-3`'e doğrusal yükselir.

**Neden?** Başlangıçta ağırlıklar rastgele, gradyanlar büyük ve güvenilmez. Adam'ın `m` ve `v` tahminleri henüz oturmamış. Büyük adımlar atmak modeli bozar.

### Aşama 2: Kosinüs sönümü

Kalan adımlarda LR yumuşak bir eğriyle `3e-4`'e (başlangıcın %10'u) düşer.

**Neden?** Eğitimin sonunda minimuma yaklaşıyoruz. Büyük adımlar minimumun etrafında zıplamaya yol açar. Küçük adımlar ince ayar yapar.

```text
LR
3e-3 |      ___
     |    /    ---___
     |  /            ---____
3e-4 |/                     ----
     +----+------------------------
     0   100                   1500
```

| Adım | LR | Aşama |
|---|---|---|
| 1 | 3.00e-5 | Isınma başı |
| 100 | 3.00e-3 | Isınma sonu, tepe |
| 750 | 1.80e-3 | Orta |
| 1500 | 3.00e-4 | Son |

---

## Hiperparametre seçimi

| Parametre | Değer | Neden |
|---|---|---|
| `learningRate` | 3e-3 | Küçük modeller için tipik; büyük modellerde 1e-4 civarı |
| `beta1` | 0.9 | Standart, neredeyse hiç değiştirilmez |
| `beta2` | 0.999 | Standart; dil modellerinde bazen 0.95 |
| `eps` | 1e-8 | Sıfıra bölmeyi engeller |
| `maxNorm` | 1.0 | Dil modellerinde yaygın değer |

!!! tip "En önemli sayı"
    Learning rate. Diğer hepsini varsayılanda bırakabilirsiniz ama LR'yi veri ve model boyutuna göre ayarlamanız gerekir.

    **Hızlı test:** 3e-3 ile loss patlıyorsa 1e-3 deneyin. Çok yavaş düşüyorsa 5e-3 deneyin.

---

## Alternatif optimizer'lar

| Optimizer | Özellik | Kullanım |
|---|---|---|
| **SGD** | En basit, momentum yok | Görüntü modellerinde hâlâ rakip |
| **SGD + Momentum** | Gürültüyü yumuşatır | ResNet vb. |
| **RMSprop** | Sadece ikinci moment | Adam'ın atası |
| **Adam** (bu proje) | İki moment | Dil modellerinde varsayılan |
| **AdamW** | Adam + ayrıştırılmış weight decay | Modern LLM standardı |
| **Lion, Sophia** | Yeni nesil, daha az bellek | Araştırma aşamasında |

AdamW eklemek kolay bir alıştırmadır: güncelleme satırına `- LearningRate * weightDecay * p.Data[i]` terimi eklemek yeterli.

---

Sıradaki adım: [Eğitim Döngüsü](dongu.md)
