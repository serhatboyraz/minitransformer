# MiniTransformer

MiniTransformer, C# ile yazılmış, GPT benzeri bir dönüştürücü (transformer) modelidir. Bu proje, otomatik üretimli dil modellerinin temel kavramlarını öğrenmek için tasarlanmıştır: tokenizasyon, embedding, çok başlı dikkat (multi-head attention), besleme ağı (feed-forward), eğitim döngüsü, kayıp (loss) optimizasyonu, checkpoint kaydı ve metin üretimi.

Bu proje küçük ve okunabilir şekilde hazırlandığı için, minimal bir decoder-only transformer yapısını gerçek bir uygulama üzerinden öğrenmek isteyenler için uygun bir örnek sunar.

## Özellikler

- Eğitim corpusundan türetilen karakter seviyesinde tokenizer
- Token ve konum embeddingleri
- Ön katman normalizasyonu ile decoder-only transformer blokları
- Çok başlı öz-dikkat (self-attention)
- Feed-forward ağ
- Gradient clipping içeren Adam optimizer
- CSV olarak loss takibi
- Model checkpoint kaydetme/yükleme
- Eğitilmiş modelden metin üretimi

## Proje yapısı

- `Program.cs` – CLI giriş noktası ve eğitim / üretim komutları
- `src/GptModel.cs` – model ve üretim mantığı
- `src/Layers.cs` – transformer blokları, dikkat mekanizması, projeksiyon ve normalizasyon
- `src/Tokenizer.cs` – karakter seviyesinde sözlük ve kodlama/çözme işlemleri
- `src/Checkpoint.cs` – checkpoint kaydetme/yükleme
- `src/AdamOptimizer.cs` – optimizer uygulaması
- `src/LossChart.cs` – ASCII loss grafiği üreten yardımcı sınıf
- `data/input.txt` – varsayılan eğitim verisi

## Gereksinimler

- .NET 10 SDK (veya ortamınızın gerektirdiği uygun sürüm)

## Hızlı başlama

Bağımlılıkları geri yükleyin:

```bash
dotnet restore
```

Model eğitin:

```bash
dotnet run -c Release -- train
```

Bu işlem çalıştığınız dizinde `model.mtf` dosyasını oluşturur ve `loss.csv` dosyasına eğitim kaybı geçmişini kaydeder.

Eğitilmiş modelden metin üretin:

```bash
dotnet run -c Release -- generate
```

Veya belirli bir checkpoint kullanın:

```bash
dotnet run -c Release -- generate model.mtf "toyota corolla 2021 "
```

Eğitim kaybını gösterin:

```bash
dotnet run -c Release -- loss
```

## Komut referansı

```bash
dotnet run -c Release -- train [checkpoint_yolu]
dotnet run -c Release -- generate [checkpoint_yolu] [istem]
dotnet run -c Release -- loss [loss_csv_yolu]
```

Komut verilmezse uygulama otomatik olarak şu davranışı gösterir:

- `model.mtf` varsa yükler ve metin üretir
- yoksa yeni bir model eğitir

## Eğitim ayarları

Varsayılan eğitim yapılandırması `Program.cs` içinde tanımlıdır:

- block size: 64
- embedding size: 96
- heads: 4
- layers: 3
- batch size: 8
- training steps: 1500
- learning rate: 3e-3

## Notlar

Bu proje üretim amaçlı büyük ölçekli eğitim yerine öğrenme ve deney yapma için tasarlanmıştır. Model küçük ve kompakt bir metin corpusu üzerinde eğitim aldığı için, transformer iç yapısını anlamak ve mimari değişiklikleri denemek için uygundur.

## Lisans

Bu proje eğitim ve deney amaçlı olarak sağlanmıştır.
