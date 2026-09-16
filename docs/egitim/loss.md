# Loss Fonksiyonu

**Dosya:** `src/Ops.cs` — `CrossEntropy`

Loss (kayıp), eğitimin **tek** geri bildirim kaynağıdır. Model iyi mi kötü mü, sadece bu sayıdan anlarız.

---

## Cross-entropy nedir?

Türkçesi "çapraz düzensizlik". Anlamı: *"Modelin tahmin dağılımı, gerçek dağılımdan ne kadar uzak?"*

Bizim durumumuzda gerçek dağılım basittir — doğru karakter %100, diğerleri %0:

```text
Gerçek:  [0, 0, 0, 1, 0, 0, ...]   ('d' karakteri doğru)
Tahmin:  [0.1, 0.05, 0.2, 0.6, 0.03, 0.02, ...]
```

Formül bu durumda tek terime iner:

$$L = -\log(p_{\text{doğru}})$$

Yukarıdaki örnekte: $-\log(0.6) = 0.51$

---

## Sezgi: sürpriz miktarı

Loss aslında **bilgi teorisindeki "şaşkınlık"** ölçüsüdür. Birimi *nat*'tır (doğal logaritma tabanında bit).

| Modelin verdiği olasılık | Loss | Yorum |
|---|---|---|
| 0.999 | 0.001 | "Zaten biliyordum" |
| 0.90 | 0.105 | Çok emin |
| 0.60 | 0.511 | İyi tahmin |
| 0.21 (1/47 rastgele) | **3.85** | Hiçbir fikri yok |
| 0.01 | 4.61 | Yanılmış |
| 0.001 | 6.91 | Fena halde yanılmış |

!!! success "Bizim eğitimimiz"
    **3.87 → 0.26**

    Başlangıçta tam rastgele ($\ln 47 = 3.85$), sonunda doğru karaktere ortalama **%77 olasılık** veriyor.

### Neden logaritma?

İki sebep:

1. **Ceza asimetrisi**: Doğruya 0.01 olasılık vermek, 0.5 vermekten 9 kat daha fazla cezalandırılır. Model "çok emin olup yanılmaktan" kaçınmayı öğrenir.
2. **Çarpımı toplamaya çevirir**: Bir dizinin olasılığı, tek tek olasılıkların çarpımıdır. Logaritma bunu toplamaya çevirir — hem sayısal olarak kararlı hem hesabı kolay.

---

## Kod: ileri geçiş

```csharp
public static Tensor CrossEntropy(Tensor logits, int[] targets)
{
    var loss = new Tensor(1, 1) { Parents = [logits] };
    var probs = new float[logits.Length];
    int rows = logits.Rows;      // 512
    int vocab = logits.Cols;     // 47
    double total = 0.0;

    for (int r = 0; r < rows; r++)
    {
        int b = r * vocab;

        // 1. Sayısal kararlılık için en büyük logiti bul
        float max = float.NegativeInfinity;
        for (int c = 0; c < vocab; c++)
            max = MathF.Max(max, logits.Data[b + c]);

        // 2. Softmax
        double sum = 0.0;
        for (int c = 0; c < vocab; c++)
        {
            float e = MathF.Exp(logits.Data[b + c] - max);
            probs[b + c] = e;
            sum += e;
        }

        float inv = (float)(1.0 / sum);
        for (int c = 0; c < vocab; c++)
            probs[b + c] *= inv;

        // 3. Doğru karakterin olasılığının negatif logaritması
        total += -Math.Log(Math.Max(probs[b + targets[r]], 1e-12f));
    }

    loss.Data[0] = (float)(total / rows);
    ...
}
```

### Üç savunma mekanizması

| Satır | Neyi engeller |
|---|---|
| `- max` | $e^{100}$ taşması (`float` sonsuz olur) |
| `Math.Max(..., 1e-12f)` | $\log(0) = -\infty$ |
| `double total` | 512 terimin toplamında hassasiyet kaybı |

!!! tip "Softmax + log birleşik olabilirdi"
    Üretim kütüphaneleri `log_softmax` kullanır: $\log(p_i) = x_i - \max - \log\sum e^{x_j - \max}$. Bu, `exp` sonra `log` almaktan daha kararlıdır. Burada anlaşılırlık için ayrı tutuldu.

---

## Kod: geri geçiş (şaşırtıcı derecede basit)

Softmax + cross-entropy kombinasyonunun türevi olağanüstü sadedir:

$$\frac{\partial L}{\partial \text{logit}_i} = p_i - y_i$$

Yani **tahmin eksi gerçek**. Hepsi bu.

```csharp
loss.BackwardFn = () =>
{
    float scale = loss.Grad[0] / rows;
    for (int r = 0; r < rows; r++)
    {
        int b = r * vocab;
        for (int c = 0; c < vocab; c++)
        {
            float target = c == targets[r] ? 1.0f : 0.0f;
            logits.Grad[b + c] += (probs[b + c] - target) * scale;
        }
    }
};
```

### Neden bu kadar basit?

Ayrı ayrı hesaplasaydık softmax'ın türevi (Jacobian matrisi) ve logaritmanın türevi çirkin ifadeler üretirdi. Ama birleştirildiğinde çoğu terim sadeleşir ve geriye $p - y$ kalır.

Bu, derin öğrenmedeki en zarif sonuçlardan biridir ve softmax + cross-entropy ikilisinin neden her yerde birlikte kullanıldığını açıklar.

### Gradyanın yorumu

```text
Doğru karakter (y=1):  grad = p - 1  (negatif) → logiti ARTIR
Yanlış karakter (y=0): grad = p - 0  (pozitif) → logiti AZALT
```

Model ne kadar yanılıyorsa, düzeltme o kadar büyük olur. Zaten doğru biliyorsa ($p \approx 1$) gradyan sıfıra yaklaşır ve ağırlıklar neredeyse hiç değişmez.

---

## Ortalama alma

```csharp
loss.Data[0] = (float)(total / rows);   // rows = 512
float scale = loss.Grad[0] / rows;
```

512 satırın (8 dizi × 64 karakter) ortalaması alınır. Toplam yerine ortalama kullanmanın faydası: `BatchSize` veya `BlockSize` değiştiğinde **learning rate'i değiştirmeye gerek kalmaz**.

Toplam kullansaydık, batch'i 8'den 16'ya çıkardığınızda gradyanlar iki katına çıkar ve learning rate'i yarıya indirmeniz gerekirdi.

---

## Perplexity: alternatif ölçü

Dil modelleri genelde **perplexity** ile raporlanır:

$$\text{PPL} = e^{L}$$

| Loss | Perplexity | Yorum |
|---|---|---|
| 3.85 | 47.0 | 47 seçenek arasında kararsız (tam rastgele) |
| 2.00 | 7.4 | ~7 seçeneğe indirmiş |
| 1.00 | 2.7 | ~3 seçenek |
| **0.26** | **1.30** | Neredeyse kesin biliyor |

Perplexity'nin sezgisel yorumu: *"Model her adımda kaç seçenek arasında bocalıyor?"*

---

## Eğitim loss'u ≠ gerçek başarı

!!! danger "Overfitting (ezberleme) tehlikesi"
    Bu projede **doğrulama seti yok**. Yani model eğitim verisini ezberliyor mu, yoksa gerçekten öğreniyor mu ölçmüyoruz.

    10 KB veri ve 350 bin parametre ile bir miktar ezber kesinlikle var. Loss'un 0,26'ya kadar inmesi kısmen bunun göstergesi.

Doğru yöntem: veriyi %90 eğitim, %10 doğrulama diye ayırmak ve ikisinin loss'unu ayrı izlemek.

```text
Sağlıklı:      eğitim ↓, doğrulama ↓
Ezberleme:     eğitim ↓, doğrulama ↑   ← burada durmalı
```

Bunu eklemek iyi bir alıştırmadır — bkz. [Deneyler](../ileri/deneyler.md).

---

Sıradaki adım: [Autograd](autograd.md)
