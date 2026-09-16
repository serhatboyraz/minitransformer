# Deneyler

Bu sayfa, öğrendiklerinizi pekiştirmek için yapabileceğiniz deneyleri kolaydan zora sıralar.

---

## Seviye 1: Ayar oynama (kod değişikliği yok)

### D1.1 — Farklı promptlar

```powershell
dotnet run -c Release -- generate "togg "
dotnet run -c Release -- generate "elektrikli araç "
dotnet run -c Release -- generate "bmw x"
dotnet run -c Release -- generate "9"
```

**Gözlemleyin:** Model prompt'un bağlamına göre farklı davranıyor mu? `"elektrikli "` ile başlarsanız açıklama cümlesi, marka adıyla başlarsanız kayıt satırı gelir mi?

### D1.2 — Aynı prompt, tekrar tekrar

```powershell
1..5 | ForEach-Object { dotnet run -c Release --no-build -- generate "fiat " }
```

**Gözlemleyin:** Her seferinde farklı çıktı geliyor. Neden? Çünkü `Generate` içinde tohumsuz `new Random()` var ve örnekleme rastgele.

---

## Seviye 2: Hiperparametre deneyleri

Her deneyden sonra:

```powershell
dotnet build -c Release
dotnet run -c Release --no-build -- train deney.mtf
```

### D2.1 — Learning rate

| Değer | Beklenen sonuç |
|---|---|
| `1e-4` | Çok yavaş öğrenme, loss 1.5 civarında kalır |
| `1e-3` | Kararlı ama yavaş |
| **`3e-3`** | **Varsayılan** |
| `1e-2` | Dalgalı, muhtemelen kararsız |
| `1e-1` | Loss patlar, `NaN` olabilir |

**Soru:** Hangi noktada eğitim bozuluyor? Gradient clipping olmasaydı daha erken mi bozulurdu?

### D2.2 — Model derinliği

```csharp
private const int Layers = 1;   // sonra 2, 3, 6, 12
```

| Katman | Beklenen loss | Süre |
|---|---|---|
| 1 | ~0,75 | ~50 sn |
| 2 | ~0,40 | ~95 sn |
| 3 | ~0,26 | ~137 sn |
| 6 | ~0,18 | ~270 sn |

**Soru:** Azalan verim nerede başlıyor? Süre doğrusal artarken kazanç azalıyor mu?

### D2.3 — Bağlam penceresi

```csharp
private const int BlockSize = 16;   // sonra 32, 64, 128
```

`BlockSize = 16` yapın. Model bir araç kaydının tamamını göremez (kayıtlar ~58 karakter).

**Gözlemleyin:** Üretilen metin hâlâ tutarlı mı? Satır sonları doğru yerde mi?

### D2.4 — Attention kafası sayısı

```csharp
private const int Heads = 1;   // sonra 2, 4, 8
```

`EmbedSize = 96` sabitken: 1 kafa → 96 boyut, 8 kafa → 12 boyut/kafa.

**Soru:** Çok fazla kafa, kafa başına boyutu azaltıyor. Bir denge noktası var mı?

### D2.5 — Batch boyutu

```csharp
private const int BatchSize = 1;   // sonra 4, 8, 32
```

**Gözlemleyin:** `loss.csv` içindeki dalgalanma. Küçük batch = gürültülü gradyan = zıplayan loss.

---

## Seviye 3: Küçük kod değişiklikleri

### D3.1 — Temperature deneyi

`Program.cs` içinde `GenerateFrom` metodunda:

```csharp
Console.WriteLine(model.Generate(tokenizer, prompt, maxNewTokens: 400,
                                 temperature: 0.3, topK: 10, new Random()));
```

`0.3`, `0.8`, `1.2`, `2.0` deneyin.

**Gözlemleyin:** Düşük sıcaklıkta tekrar, yüksek sıcaklıkta bozulma.

### D3.2 — Nedensel maskeyi kaldırın

`src/Layers.cs` içinde:

```csharp
var weights = Ops.SoftmaxRows(scores);   // CausalMask kaldırıldı
```

**Beklenen:** Loss neredeyse sıfıra iner (model cevabı görüyor), ama üretilen metin tamamen saçma olur.

!!! danger "Bu deney çok öğreticidir"
    Veri sızıntısının neden felaket olduğunu somut olarak gösterir. Loss'a bakıp "harika!" dersiniz, çıktıya bakınca gerçeği görürsünüz.

### D3.3 — Residual bağlantıları kaldırın

```csharp
public Tensor Forward(Tensor x, int batchCount, int seqLen)
{
    x = _attention.Forward(_norm1.Forward(x), batchCount, seqLen);   // x + ... kaldırıldı
    return _feedForward.Forward(_norm2.Forward(x));
}
```

**Beklenen:** 3 katmanda fark az olabilir. `Layers = 12` yapıp tekrar deneyin — eğitim çok kötüleşmeli.

### D3.4 — Pozisyon embedding'i kaldırın

```csharp
var x = Ops.Gather(_tokenEmbedding, tokens);   // pozisyon toplanmıyor
```

**Beklenen:** Model sıra bilgisini kaybeder. Loss belirgin şekilde yüksek kalır, format bozulur.

### D3.5 — GELU yerine ReLU

`src/Ops.cs` içine ekleyin:

```csharp
public static Tensor Relu(Tensor x)
{
    var outTensor = new Tensor(x.Rows, x.Cols) { Parents = [x] };
    for (int i = 0; i < x.Length; i++)
        outTensor.Data[i] = MathF.Max(0f, x.Data[i]);

    outTensor.BackwardFn = () =>
    {
        for (int i = 0; i < x.Length; i++)
            if (x.Data[i] > 0f)
                x.Grad[i] += outTensor.Grad[i];
    };

    return outTensor;
}
```

**Gözlemleyin:** Loss eğrisi ne kadar farklı? ReLU biraz daha hızlı ama biraz daha kötü olmalı.

---

## Seviye 4: Yeni özellikler

### D4.1 — Doğrulama seti ekleyin

Veriyi %90/%10 bölün:

```csharp
int split = (int)(data.Length * 0.9);
int[] trainData = data[..split];
int[] valData = data[split..];
```

Her 100 adımda doğrulama loss'unu ölçün (geri geçiş yapmadan):

```csharp
if (step % 100 == 0)
{
    var (vi, vt) = SampleBatch(valData, valRng);
    var vLoss = model.Loss(vi, vt);
    Console.WriteLine($"  val loss: {vLoss.Data[0]:F4}");
}
```

**Gözlemleyin:** Eğitim loss'u düşerken doğrulama loss'u ne zaman yükselmeye başlıyor? İşte orada **overfitting** başlıyor.

!!! tip "En değerli deney"
    Bu, projeye ekleyebileceğiniz en öğretici özelliktir. Makine öğrenmesinin en temel kavramını somutlaştırır.

### D4.2 — Top-p (nucleus) örnekleme

`SampleLastRow` içinde top-k yerine:

```csharp
// Olasılıkları hesapla, büyükten küçüğe sırala
// Kümülatif toplam p'yi (örneğin 0.9) geçene kadar al, gerisini ele
```

**Karşılaştırın:** Top-k ile top-p çıktıları arasındaki fark.

### D4.3 — Weight tying

`_head.Weight` yerine `_tokenEmbedding`'in transpozunu kullanın.

**Kazanç:** 4.512 parametre tasarrufu ve genelde daha iyi kalite.

**Zorluk:** `Transpose` işleminin gradyanı doğru akmalı. Autograd bunu otomatik halleder.

### D4.4 — Dropout

Eğitim sırasında rastgele nöronları sıfırlayan bir katman:

```csharp
public static Tensor Dropout(Tensor x, float rate, Random rng, bool training)
{
    if (!training || rate <= 0f) return x;
    // rate oranında elemanı sıfırla, kalanları 1/(1-rate) ile ölçekle
}
```

**Gözlemleyin:** Eğitim loss'u yükselir ama doğrulama loss'u (D4.1 ile) düşer mi?

### D4.5 — Eğitime devam (resume)

Optimizer durumunu (`m`, `v`, `_step`) da checkpoint'e ekleyin:

```powershell
dotnet run -c Release -- train model.mtf
dotnet run -c Release -- resume model.mtf 500   # 500 adım daha
```

### D4.6 — KV cache

Üretim sırasında önceki tokenlerin Key/Value vektörlerini saklayın.

**Beklenen kazanç:** 400 token üretimi 4 saniyeden ~0,3 saniyeye iner.

---

## Seviye 5: Büyük değişiklikler

### D5.1 — BPE tokenizer

Karakter yerine alt-kelime tokenleri kullanın:

1. Eğitim metnindeki en sık karakter çiftini bul
2. Onu tek token olarak birleştir
3. İstenen sözlük boyutuna ulaşana kadar tekrarla

**Beklenen:** Aynı bağlam penceresinde çok daha fazla metin sığar.

### D5.2 — RoPE (döndürmeli pozisyon kodlaması)

Öğrenilen pozisyon tablosu yerine, Q ve K vektörlerini pozisyona göre döndürün.

**Kazanç:** Eğitimde görülenden daha uzun dizilere genelleme.

### D5.3 — Soru-cevap formatı

Veriyi şu formatta hazırlayın:

```text
S: renault clio kaç beygir?
C: 100hp
S: en ucuz suv hangisi?
C: dacia duster 920bin tl
```

**Gözlemleyin:** Model soru-cevap yapısını taklit ediyor mu? (Cevapların doğru olmasını beklemeyin — model bilgiyi değil formatı öğrenir.)

### D5.4 — Model boyutunu 10 kat büyütün

```csharp
private const int EmbedSize = 256;
private const int Layers = 8;
private const int Heads = 8;
private const int BlockSize = 128;
```

~4 milyon parametre. Eğitim saatler sürecek.

**Şimdi GPU sorusu anlamlı hale gelir** — bkz. [GPU](gpu.md).

---

## Deney günlüğü tutun

Her deneyde şunları kaydedin:

| Alan | Örnek |
|---|---|
| Değişiklik | `Layers: 3 → 6` |
| Son 50 ortalama loss | 0,187 |
| Süre | 270 sn |
| Örnek çıktı kalitesi | "Format doğru, marka isimleri daha temiz" |
| Checkpoint adı | `6katman.mtf` |

```powershell
dotnet run -c Release --no-build -- train 6katman.mtf
Copy-Item loss.csv deney-6katman.csv
dotnet run -c Release --no-build -- loss deney-6katman.csv
```

!!! success "Bilimsel yöntem"
    **Aynı anda tek bir şey değiştirin.** İki ayarı birden değiştirirseniz hangisinin etki ettiğini bilemezsiniz.

    Tohum sabit olduğu için (`new Random(1337)`) aynı ayarlar her zaman aynı sonucu verir — bu, karşılaştırmayı güvenilir kılar.

---

Sıradaki adım: [Kod Haritası](../referans/kod-haritasi.md)
