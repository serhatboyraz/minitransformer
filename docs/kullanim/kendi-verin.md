# Kendi Verinle Eğitmek

Modeli istediğiniz metinle eğitebilirsiniz. Bu sayfa nelere dikkat etmeniz gerektiğini anlatır.

---

## En basit yol

`data/input.txt` dosyasını kendi metninizle değiştirin ve yeniden derleyin:

```powershell
# Metninizi kopyalayın
Copy-Item benim-verim.txt data\input.txt

# input.txt çıktı klasörüne kopyalanır
dotnet build -c Release
dotnet run -c Release --no-build -- train
```

!!! warning "Yeniden derleme şart"
    `data/input.txt`, `.csproj` içindeki şu ayarla çıktı klasörüne kopyalanır:

    ```xml
    <None Include="data\input.txt" CopyToOutputDirectory="PreserveNewest" />
    ```

    Sadece dosyayı değiştirip `--no-build` ile çalıştırırsanız **eski veri** kullanılır.

---

## Veri hazırlama kuralları

### 1. Yeterince büyük olsun

| Veri boyutu | Sonuç |
|---|---|
| < 2 KB | Model ezberler, yeni bir şey üretmez |
| **5-50 KB** | **Bu model için ideal** |
| > 500 KB | Model kapasitesi yetersiz kalır, büyütmeniz gerekir |

Bizim `input.txt` 10 KB.

### 2. Tutarlı olsun

Model desen arar. Veri ne kadar düzenliyse o kadar iyi öğrenir.

!!! success "İyi veri örnekleri"
    - Ürün/araç/ilan listeleri (sabit format)
    - Şiir veya şarkı sözleri (ritim, kafiye)
    - Log kayıtları
    - Reçeteler
    - Kod (tek dilde)
    - Tek yazarın metinleri

!!! failure "Kötü veri örnekleri"
    - Rastgele web kazıması (tutarsız)
    - Çok dilli karışım (sözlük patlar)
    - Tablo/HTML karışımı metin
    - Çok kısa ve birbirinden alakasız parçalar

### 3. Küçük harfe çevirmeyi düşünün

```powershell
(Get-Content veri.txt -Raw).ToLower() | Set-Content data\input.txt
```

Büyük harfleri kaldırmak sözlüğü ~%40 küçültür. Küçük modellerde bu ciddi bir kazançtır.

### 4. Karakter çeşitliliğini sınırlayın

Sözlük büyüdükçe:

- Embedding tablosu büyür
- Çıkış katmanı büyür
- Her karakteri öğrenmek için daha çok örnek gerekir

Türkçe metin için ideal sözlük 40-60 karakter arasıdır. Emoji, nadir semboller ve karışık noktalama bunu şişirir.

```powershell
# Sözlük boyutunu önceden görün
(Get-Content data\input.txt -Raw).ToCharArray() |
    Select-Object -Unique |
    Measure-Object |
    Select-Object Count
```

### 5. UTF-8 kullanın

Program dosyayı `File.ReadAllText` ile okur, bu varsayılan olarak UTF-8 bekler. Türkçe karakterler için şart.

```powershell
# Kodlamayı UTF-8'e çevirin
Get-Content veri.txt | Set-Content -Encoding utf8 data\input.txt
```

---

## Veri türüne göre ayar önerileri

=== "Kısa kayıtlar (ilan, ürün)"

    ```csharp
    private const int BlockSize = 64;    // bir kayıt sığmalı
    private const int EmbedSize = 96;
    private const int Layers = 3;
    private const int TrainingSteps = 1500;
    ```

    Bir kaydın tamamı bağlam penceresine sığmalı. Kayıt 100 karakterse `BlockSize = 128` yapın.

=== "Şiir / şarkı sözü"

    ```csharp
    private const int BlockSize = 128;   // kıta yapısı için
    private const int EmbedSize = 128;
    private const int Layers = 4;
    private const int TrainingSteps = 3000;
    ```

    Kafiye ve ritim uzun menzilli bağımlılıktır, daha geniş pencere gerekir.

=== "Kod"

    ```csharp
    private const int BlockSize = 128;   // fonksiyon gövdesi
    private const int EmbedSize = 128;
    private const int Layers = 4;
    private const int TrainingSteps = 4000;
    ```

    Kod çok yapısaldır; parantez eşleşmesi için derinlik ister.

=== "Düz Türkçe metin"

    ```csharp
    private const int BlockSize = 96;
    private const int EmbedSize = 128;
    private const int Layers = 4;
    private const int TrainingSteps = 5000;
    ```

    Doğal dil en zorudur. Küçük modelde dilbilgisi tam oturmaz ama kelime yapısı öğrenilir.

!!! danger "Boyut değiştirince eski checkpoint bozulur"
    `EmbedSize`, `Layers` veya `Heads` değiştirdiyseniz eski `model.mtf` yüklenemez. Yeni bir ad verin:

    ```powershell
    dotnet run -c Release --no-build -- train siir.mtf
    ```

---

## Örnek: şiir modeli

```powershell
# 1. Veriyi hazırlayın (en az 10 KB şiir metni)
Get-Content siirler.txt -Raw | ForEach-Object { $_.ToLower() } |
    Set-Content -Encoding utf8 data\input.txt

# 2. Program.cs içinde BlockSize = 128, Layers = 4 yapın

# 3. Eğitin
dotnet build -c Release
dotnet run -c Release --no-build -- train siir.mtf

# 4. Üretin
dotnet run -c Release --no-build -- generate siir.mtf "gece "
```

---

## Sorun giderme

| Belirti | Muhtemel sebep | Çözüm |
|---|---|---|
| Loss 3+ takılı kaldı | Veri çok karmaşık veya çok küçük | Daha düzenli/büyük veri |
| Loss çok hızlı 0'a indi | Veri çok küçük, ezberledi | Veriyi büyütün |
| Çıktıda bozuk karakter | Kodlama sorunu | UTF-8'e çevirin |
| `Sequence exceeds block size` | Prompt 64 karakterden uzun | Kısaltın veya `BlockSize` artırın |
| Çıktı hep aynı | Temperature çok düşük | `temperature: 1.0` deneyin |
| Çıktı tamamen saçma | Yeterince eğitilmedi | `TrainingSteps` artırın |
| `OutOfMemoryException` | Batch × block çok büyük | `BatchSize` düşürün |

---

## Veri kaynağı fikirleri

Telif hakkına dikkat ederek:

- **Kendi yazışmalarınız** — WhatsApp dışa aktarımı (kişisel veri, dikkatli olun)
- **Kendi kodunuz** — bir projedeki tüm `.cs` dosyalarını birleştirin
- **Açık veri setleri** — TÜİK, açık veri portalları
- **Kamuya açık metinler** — telifi düşmüş eserler
- **Sentetik veri** — bir betikle üretilmiş listeler (bizim `input.txt` gibi)

```powershell
# Örnek: tüm C# dosyalarını tek dosyada birleştirme
Get-ChildItem -Recurse -Filter *.cs |
    Get-Content -Raw |
    Set-Content -Encoding utf8 data\input.txt
```

!!! tip "Kod modeli eğlencelidir"
    Kendi projenizin kodunu verirseniz model sizin yazım stilinizi taklit eden (ama çalışmayan) kod üretir. Girintileme, parantez eşleşmesi ve isimlendirme desenlerini öğrenir.

---

Sıradaki adım: [Performans](../ileri/performans.md)
