# Kurulum

Bu projeyi çalıştırmak için tek bir şeye ihtiyacınız var: **.NET SDK**.

## 1. .NET SDK kurulumu

=== "Windows"

    En kolay yol — PowerShell'i açıp:

    ```powershell
    winget install Microsoft.DotNet.SDK.10
    ```

    Alternatif olarak [dotnet.microsoft.com/download](https://dotnet.microsoft.com/download) adresinden indirebilirsiniz.

=== "macOS"

    ```bash
    brew install --cask dotnet-sdk
    ```

=== "Linux"

    ```bash
    sudo apt-get update && sudo apt-get install -y dotnet-sdk-10.0
    ```

### Kurulumu doğrulayın

```powershell
dotnet --version
```

Şuna benzer bir çıktı görmelisiniz:

```text
10.0.101
```

!!! warning "Sürüm uyarısı"
    Proje `net10.0` hedefliyor. Daha eski bir SDK'nız varsa (örneğin 8.0), [MiniTransformer.csproj](../referans/kod-haritasi.md) dosyasındaki `<TargetFramework>net10.0</TargetFramework>` satırını `net8.0` yapmanız yeterli. Kodda kullanılan hiçbir özellik .NET 8'in ötesinde değil.

---

## 2. Projeyi derleyin

Proje klasörüne gidip:

```powershell
cd project
dotnet build -c Release
```

`-c Release` çok önemlidir. Debug modunda derlerseniz eğitim **5-10 kat yavaş** çalışır, çünkü JIT optimizasyonları ve SIMD komutları devre dışı kalır.

Başarılı çıktı:

```text
MiniTransformer -> bin\Release\net10.0\MiniTransformer.dll
Oluşturma başarılı oldu.
    0 Uyarı
    0 Hata
```

---

## 3. Klasör yapısı

Derlemeden sonra klasörünüz şöyle görünür:

```text
project/
├── MiniTransformer.csproj      # Proje tanımı
├── Program.cs                  # Giriş noktası ve CLI
├── data/
│   └── input.txt               # Eğitim verisi (araç bilgileri)
├── src/
│   ├── Tensor.cs               # Matris + otomatik türev düğümü
│   ├── Ops.cs                  # Türevlenebilir işlemler
│   ├── Layers.cs               # Linear, LayerNorm, Attention, FFN
│   ├── GptModel.cs             # Modelin kendisi
│   ├── AdamOptimizer.cs        # Ağırlık güncelleyici
│   ├── Tokenizer.cs            # Metin ↔ sayı dönüşümü
│   ├── Checkpoint.cs           # Ağırlıkları kaydet/yükle
│   └── LossChart.cs            # ASCII grafik çizici
├── docs/                       # Bu doküman
├── mkdocs.yml                  # Doküman yapılandırması
└── bin/                        # Derleme çıktısı (otomatik)
```

---

## 4. Donanım gereksinimleri

| Bileşen | Minimum | Önerilen | Notlar |
|---|---|---|---|
| CPU | 2 çekirdek | 8+ çekirdek | Matris çarpımı tüm çekirdekleri kullanır |
| RAM | 1 GB | 4 GB | Hesap grafiği adım başına ~100 MB tutar |
| Disk | 50 MB | 100 MB | `model.txt` tek başına 4,6 MB |
| GPU | **Gerekmez** | — | Bkz. [GPU sayfası](../ileri/gpu.md) |

!!! info "Çekirdek sayısı ne kadar fark eder?"
    14 çekirdekli bir makinede tüm eğitim 137 saniye sürüyor. 4 çekirdekli bir makinede yaklaşık 5-6 dakika bekleyin. Tek çekirdekte ~15 dakika.

---

## 5. Dokümanı yerel olarak açmak (isteğe bağlı)

Okuduğunuz bu siteyi kendi bilgisayarınızda çalıştırmak isterseniz Python gerekir:

```powershell
pip install mkdocs-material
mkdocs serve
```

Ardından tarayıcıdan `http://127.0.0.1:8000` adresine gidin. Statik HTML üretmek için:

```powershell
mkdocs build
```

Çıktı `site/` klasörüne yazılır ve herhangi bir web sunucusunda yayınlanabilir.

---

Sıradaki adım: [İlk Çalıştırma](ilk-calistirma.md)
