# Checkpoint (Model Kaydetme)

**Dosya:** `src/Checkpoint.cs`

Eğitim pahalıdır, kullanım ucuzdur. Öğrenilen ağırlıkları diske yazıp sonradan yüklemek şarttır.

---

## Neyi kaydediyoruz?

Model = **mimari** + **ağırlıklar** + **sözlük**. Üçü de dosyaya yazılır:

| Bilgi | Neden gerekli |
|---|---|
| Config (`VocabSize`, `BlockSize`, `EmbedSize`, `Heads`, `Layers`) | Aynı yapıyı yeniden kurmak için |
| Sözlük (47 karakter) | Id → karakter eşlemesi aynı olmalı |
| 54 tensör, 350.927 `float` | Öğrenilen her şey |

!!! danger "Sözlüğü kaydetmezsek ne olur?"
    Model yüklenir, çalışır, ama çıktı anlamsız olur. Çünkü eğitimde `22 = 'f'` iken yeni sözlükte `22 = 'q'` olabilir. Model doğru id'leri üretir ama yanlış karakterlere çevrilirler.

---

## İki format

Uzantıya göre otomatik seçilir:

```csharp
private static bool IsText(string path) =>
    Path.GetExtension(path).Equals(".txt", StringComparison.OrdinalIgnoreCase);

public static void Save(string path, GptModel model, Tokenizer tokenizer)
{
    if (IsText(path)) SaveText(path, model, tokenizer);
    else SaveBinary(path, model, tokenizer);
}
```

| Format | Boyut | Hız | Okunabilir |
|---|---|---|---|
| `.mtf` (ikili) | 1,37 MB | Hızlı | Hayır |
| `.txt` (metin) | 4,60 MB | Yavaş | Evet |

---

## İkili format

```csharp
writer.Write(Magic);           // "MTRF"
writer.Write(Version);         // 1
writer.Write(config.VocabSize);
writer.Write(config.BlockSize);
writer.Write(config.EmbedSize);
writer.Write(config.Heads);
writer.Write(config.Layers);
writer.Write(tokenizer.Vocabulary);

var parameters = model.Parameters().ToArray();
writer.Write(parameters.Length);

foreach (var p in parameters)
{
    writer.Write(p.Rows);
    writer.Write(p.Cols);
    writer.Write(MemoryMarshal.AsBytes<float>(p.Data));   // (1)!
}
```

1. `float[]` dizisini doğrudan bayt dizisi olarak yazar. Eleman eleman yazmaktan çok daha hızlıdır.

### Dosya yapısı

```text
┌──────────────────────────────┐
│ "MTRF"          (magic)      │  Dosya türü imzası
│ 1               (version)    │  Format sürümü
├──────────────────────────────┤
│ 47 64 96 4 3    (config)     │  Mimari
│ "\n\r .0123..." (vocab)      │  Sözlük
│ 54              (param count)│  Tensör sayısı
├──────────────────────────────┤
│ 47 96 [4512 float]           │  Token embedding
│ 64 96 [6144 float]           │  Position embedding
│ 1  96 [96 float]             │  LayerNorm gain
│ ...                          │
└──────────────────────────────┘
```

### Magic number neden var?

```csharp
if (reader.ReadString() != Magic)
    throw new InvalidDataException($"{path} bir MiniTransformer checkpoint dosyası değil.");
```

Yanlış dosya verildiğinde anlaşılır hata almak için. Olmasaydı rastgele baytlar config olarak okunur ve `new Tensor(-38291, 91823)` gibi çılgınca bir hata alırdınız.

### Sürüm numarası neden var?

Format ileride değişirse eski dosyaları tanıyıp uygun şekilde işleyebilmek (ya da en azından net bir hata vermek) için.

---

## Metin format

```text
minitransformer-text 1
config 47 64 96 4 3
vocab 10 13 32 46 48 49 50 51 52 53 54 55 56 57 58 97 98 ...
params 54
tensor 47 96
-0.0128374 0.0331201 -0.00914772 ... (96 sayı)
0.0219488 -0.0442901 0.0117362 ... (96 sayı)
... (47 satır)
tensor 64 96
...
```

```csharp
writer.WriteLine($"{TextHeader} {Version}");
writer.WriteLine($"config {config.VocabSize} {config.BlockSize} ...");
writer.WriteLine("vocab " + string.Join(' ', tokenizer.Vocabulary.Select(c => (int)c)));
writer.WriteLine($"params {parameters.Length}");

foreach (var p in parameters)
{
    writer.WriteLine($"tensor {p.Rows} {p.Cols}");
    for (int r = 0; r < p.Rows; r++)
    {
        // satırdaki tüm float'ları boşlukla ayırarak yaz
        line.Append(p.Data[(r * p.Cols) + c].ToString("G9", CultureInfo.InvariantCulture));
    }
}
```

### Neden sözlük kod noktası olarak yazılıyor?

```csharp
writer.WriteLine("vocab " + string.Join(' ', tokenizer.Vocabulary.Select(c => (int)c)));
```

Sözlükte `\n` (10) ve `\r` (13) karakterleri var! Düz yazılsaydı dosyanın satır yapısı bozulur, okuma imkânsız hale gelirdi.

Kod noktası olarak yazmak hem güvenli hem gözle kontrol edilebilir.

### Neden `G9`?

`float` tipini kayıpsız yazmak için gereken minimum basamak sayısı 9'dur. `F4` gibi bir format kullansaydık ağırlıklar yuvarlanır, model biraz bozulurdu.

### Neden `InvariantCulture`?

Türkçe yerelde `0,0128` yazılırdı. Boşlukla ayrılmış sayılar arasında virgül karışıklık yaratmasa da, dosya başka bir makinede okunamazdı. `InvariantCulture` her yerde nokta kullanır.

---

## Yükleme ve doğrulama

```csharp
private static (GptModel Model, Tensor[] Parameters) BuildTarget(GptConfig config, int expectedCount)
{
    // Ağırlıklar dosyadan gelecek, bu yüzden başlangıç değerleri önemsiz.
    var model = new GptModel(config, new Random(0));
    var parameters = model.Parameters().ToArray();

    if (expectedCount != parameters.Length)
        throw new InvalidDataException($"Parametre sayısı uyuşmuyor: dosyada {expectedCount}, modelde {parameters.Length}.");

    return (model, parameters);
}

private static void ExpectShape(Tensor p, int rows, int cols)
{
    if (rows != p.Rows || cols != p.Cols)
        throw new InvalidDataException($"Tensör boyutu uyuşmuyor: dosyada [{rows}x{cols}], modelde [{p.Rows}x{p.Cols}].");
}
```

Üç katmanlı doğrulama:

1. **Magic/header** — doğru dosya türü mü?
2. **Parametre sayısı** — aynı mimari mi?
3. **Her tensörün boyutu** — sıralama doğru mu?

!!! success "Neden bu kadar kontrol?"
    Sessiz hatalar en kötüsüdür. Yanlış ağırlıklar yüklenirse model çalışır ama saçmalar — ve nedenini bulmak saatler sürer. Net hata mesajı hemen yol gösterir.

### Sıralama bağımlılığı

Kaydetme ve yükleme, `model.Parameters()` metodunun **aynı sırayı** üretmesine dayanır:

```csharp
public IEnumerable<Tensor> Parameters()
{
    yield return _tokenEmbedding;
    yield return _positionEmbedding;
    foreach (var p in _blocks.SelectMany(b => b.Parameters())...)
        yield return p;
}
```

`yield return` deterministik sıra garanti eder. Bu sırayı değiştirirseniz **eski checkpoint'ler bozulur** (ama boyut kontrolü sayesinde sessizce değil, hata vererek).

---

## Kullanım

```powershell
# İkili format (varsayılan, hızlı)
dotnet run -c Release -- train
dotnet run -c Release -- generate "fiat egea "

# Metin format (incelemek için)
dotnet run -c Release -- train model.txt
dotnet run -c Release -- generate model.txt "bmw 320i "

# Farklı deneyleri saklamak
dotnet run -c Release -- train 3katman.mtf
# ... Layers = 6 yapıp yeniden derle ...
dotnet run -c Release -- train 6katman.mtf
dotnet run -c Release -- generate 3katman.mtf "toyota "
dotnet run -c Release -- generate 6katman.mtf "toyota "
```

---

## Eksik: optimizer durumu

Şu an sadece **ağırlıklar** kaydediliyor. Adam'ın `m` ve `v` dizileri kaydedilmiyor.

Sonucu: Eğitime kaldığı yerden devam edemezsiniz. Yüklenen modelle eğitime devam ederseniz Adam sıfırdan başlar ve ilk adımlarda model bozulabilir.

!!! example "Alıştırma: eğitime devam"
    Şunları eklemek gerekir:

    1. `AdamOptimizer` içine `m`, `v` ve `_step` erişimcileri
    2. Checkpoint'e ek bir bölüm
    3. `train --resume model.mtf` gibi bir komut

    Büyük modellerde optimizer durumu ağırlıklardan **iki kat** büyüktür (her parametre için 2 `double`), bu yüzden çoğu proje ayrı dosyaya yazar.

---

## Gerçek dünyada formatlar

| Format | Kullanan | Özellik |
|---|---|---|
| `.pt` / `.pth` | PyTorch | Python pickle — güvenlik riski taşır |
| `.safetensors` | HuggingFace | Güvenli, hızlı, kısmi yükleme destekler |
| `.gguf` | llama.cpp | Kuantizasyon (4-bit vb.) içerir |
| `.onnx` | Çapraz platform | Mimariyi de içinde taşır |
| `.mtf` | Bu proje | Basit, öğretici |

!!! warning "Pickle güvenlik notu"
    PyTorch'un `.pt` formatı Python pickle kullanır ve **yüklerken kod çalıştırabilir**. İnternetten indirdiğiniz bir `.pt` dosyası bilgisayarınızı ele geçirebilir. Bu yüzden HuggingFace `.safetensors` formatına geçti.

    Bizim formatımız sadece veri okur, kod çalıştırmaz — bu açıdan güvenlidir.

---

Sıradaki adım: [Loss Eğrisi](loss-egrisi.md)
