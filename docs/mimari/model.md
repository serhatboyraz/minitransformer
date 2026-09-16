# Model

**Dosya:** `src/GptModel.cs`

Tüm parçaları birleştiren sınıf. GPT (Generative Pre-trained Transformer) mimarisinin minyatür bir uygulaması.

---

## Yapılandırma

```csharp
public sealed record GptConfig(int VocabSize, int BlockSize, int EmbedSize, int Heads, int Layers);
```

`record` seçilmesinin sebebi: değer eşitliği ve değişmezlik. Checkpoint yüklenirken config karşılaştırması kolaylaşır.

Varsayılan değerler:

```csharp
new GptConfig(
    VocabSize: 47,     // tokenizer belirler
    BlockSize: 64,     // bağlam penceresi
    EmbedSize: 96,     // d_model
    Heads: 4,          // attention kafası
    Layers: 3)         // blok sayısı
```

---

## Bileşenler

```csharp
private readonly Tensor _tokenEmbedding;       // [47 × 96]
private readonly Tensor _positionEmbedding;    // [64 × 96]
private readonly TransformerBlock[] _blocks;   // 3 adet
private readonly LayerNorm _finalNorm;         // [96]
private readonly Linear _head;                 // [96 × 47]
```

### Son LayerNorm neden var?

Son bloktan çıkan vektör, çıkış katmanına girmeden önce bir kez daha normalize edilir. Bu, logitlerin makul aralıkta kalmasını sağlar. GPT-2'den beri standart uygulamadır.

### Çıkış katmanı (`_head`)

96 boyutlu vektörü 47 skora çevirir. Bu skorlara **logit** denir — henüz olasılık değiller, softmax'tan geçmeleri gerekir.

!!! note "Ağırlık paylaşımı yapılmadı"
    Büyük modellerde `_head` ile `_tokenEmbedding` aynı matrisin transpozu olarak paylaşılır (*weight tying*). Parametre tasarrufu sağlar ve genelde kaliteyi artırır. Bu projede anlaşılırlık için ayrı tutuldu.

---

## İleri geçiş

```csharp
public Tensor Forward(int[][] sequences)
{
    int batch = sequences.Length;
    int seqLen = sequences[0].Length;

    if (seqLen > _config.BlockSize)
        throw new ArgumentException($"Sequence of {seqLen} exceeds block size {_config.BlockSize}.");

    var tokens = new int[batch * seqLen];
    var positions = new int[batch * seqLen];

    for (int b = 0; b < batch; b++)
    {
        if (sequences[b].Length != seqLen)
            throw new ArgumentException("Batch içindeki tüm diziler aynı uzunlukta olmalı.");

        Array.Copy(sequences[b], 0, tokens, b * seqLen, seqLen);
        for (int t = 0; t < seqLen; t++)
            positions[(b * seqLen) + t] = t;
    }

    var x = Ops.Add(
        Ops.Gather(_tokenEmbedding, tokens),
        Ops.Gather(_positionEmbedding, positions));

    foreach (var block in _blocks)
        x = block.Forward(x, batch, seqLen);

    return _head.Forward(_finalNorm.Forward(x));
}
```

Girdi `int[][]` — dizi dizisi. Çıktı `[batch*seqLen × VocabSize]` boyutunda logit matrisi.

!!! warning "Eşit uzunluk şartı"
    Bu implementasyon tüm dizilerin aynı uzunlukta olmasını bekler. Gerçek sistemlerde farklı uzunluklar **padding** (dolgu) ve **attention mask** ile çözülür. Burada eğitim verisi sabit uzunlukta dilimlendiği için gerek kalmadı.

---

## Kayıp hesabı

```csharp
public Tensor Loss(int[][] inputs, int[][] targets)
{
    var flatTargets = new int[targets.Length * targets[0].Length];
    for (int b = 0; b < targets.Length; b++)
        Array.Copy(targets[b], 0, flatTargets, b * targets[0].Length, targets[0].Length);

    return Ops.CrossEntropy(Forward(inputs), flatTargets);
}
```

Hedefler de düzleştirilir çünkü logitler zaten `[512 × 47]` şeklinde düz.

### Girdi ve hedef ilişkisi

```text
inputs  = "renault clio 202"
targets = "enault clio 2021"
```

Hedef, girdinin **bir karakter kaydırılmış** halidir. Böylece her pozisyon "bir sonraki karakter" görevini öğrenir:

| Pozisyon | Girdi | Hedef |
|---|---|---|
| 0 | `r` | `e` |
| 1 | `e` | `n` |
| 2 | `n` | `a` |
| ... | ... | ... |

64 karakterlik bir dizi, aslında **64 ayrı eğitim örneği** demektir. Bu, transformer'ın çok verimli olmasının önemli bir sebebidir — RNN'lerde bu paralellik yoktu.

---

## Parametre sayımı

```csharp
public IEnumerable<Tensor> Parameters()
{
    yield return _tokenEmbedding;
    yield return _positionEmbedding;

    foreach (var p in _blocks.SelectMany(b => b.Parameters())
                             .Concat(_finalNorm.Parameters())
                             .Concat(_head.Parameters()))
        yield return p;
}
```

!!! danger "Sıra kritiktir"
    Bu metot hem optimizer'ı beslemek hem de checkpoint yazmak için kullanılır. Sıralamanın **her çağrıda aynı** olması şarttır, yoksa kaydedilen ağırlıklar yanlış tensörlere yüklenir.

    `yield return` ile deterministik sıra garanti edilir.

### Dağılım

| Bileşen | Parametre | Oran |
|---|---|---|
| Token embedding | 4.512 | %1,3 |
| Position embedding | 6.144 | %1,8 |
| 3 × Attention | 111.744 | %31,8 |
| 3 × Feed-forward | 222.624 | %63,4 |
| 6 × LayerNorm (blok içi) | 1.152 | %0,3 |
| Son LayerNorm | 192 | %0,1 |
| Çıkış katmanı | 4.559 | %1,3 |
| **Toplam** | **350.927** | |

---

## Metin üretimi

```csharp
public string Generate(Tokenizer tokenizer, string prompt, int maxNewTokens,
                       double temperature, int topK, Random rng)
{
    var context = new List<int>(tokenizer.Encode(prompt));
    if (context.Count == 0) context.Add(0);

    var generated = new List<int>();

    for (int step = 0; step < maxNewTokens; step++)
    {
        var window = context.Skip(Math.Max(0, context.Count - _config.BlockSize)).ToArray();
        var logits = Forward([window]);
        int next = SampleLastRow(logits, temperature, topK, rng);
        context.Add(next);
        generated.Add(next);
    }

    return prompt + tokenizer.Decode(generated);
}
```

Her adımda:

1. Son 64 karakteri al (kayan pencere)
2. İleri geçiş yap
3. **Son satırın** logitlerini al — sadece o "bir sonraki karakter"i temsil eder
4. Örnekle
5. Bağlama ekle, tekrarla

!!! info "Verimsizlik uyarısı"
    Her adımda tüm pencere yeniden hesaplanıyor. Gerçek sistemler **KV cache** kullanır: önceki tokenlerin Key ve Value vektörleri saklanır, sadece yeni token hesaplanır. Bu 10-50 kat hızlandırır ama kodu epey karmaşıklaştırır.

Örnekleme stratejileri (temperature, top-k) [Metin Üretimi](../kullanim/uretim.md) sayfasında ayrıntılı anlatılıyor.

---

## Ağırlık ilkleme

```csharp
_tokenEmbedding.Data[i] = (float)(Linear.NextGaussian(rng) * 0.02);
```

Tüm ağırlıklar normal dağılımdan, standart sapma 0,02 ile başlatılır.

### Neden rastgele?

Hepsi sıfır olsaydı, aynı katmandaki tüm nöronlar **birebir aynı** gradyanı alır ve sonsuza dek aynı kalırdı. Buna **simetri kırılması problemi** denir. Rastgelelik her nöronun farklı bir şey öğrenmesini sağlar.

### Neden 0,02 gibi küçük bir değer?

Büyük başlangıç değerleri aktivasyonları patlatır, softmax doyar, gradyan akmaz. GPT-2'nin kullandığı değer budur.

### Box-Muller dönüşümü

```csharp
public static double NextGaussian(Random rng)
{
    double u1 = 1.0 - rng.NextDouble();
    double u2 = rng.NextDouble();
    return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
}
```

`Random` sınıfı düzgün (uniform) dağılım verir, bize normal (Gauss) dağılım lazım. Box-Muller iki uniform sayıdan bir normal sayı üretir.

`1.0 - rng.NextDouble()` ifadesi `u1`'in asla tam 0 olmamasını sağlar — `Log(0)` eksi sonsuz verirdi.

---

Sıradaki adım: [Loss Fonksiyonu](../egitim/loss.md)
