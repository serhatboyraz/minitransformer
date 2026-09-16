# İlk Çalıştırma

Şimdi modeli eğitelim ve ne olduğunu satır satır anlayalım.

```powershell
dotnet run -c Release -- train
```

---

## Çıktının anatomisi

### Bölüm 1: Künye

```text
corpus     : 10128 tokens, vocab 47
model      : 3 layers, 4 heads, d_model 96, 350.927 parameters
training   : 1500 steps, batch 8, block 64
```

| Satır | Anlamı |
|---|---|
| `10128 tokens` | Eğitim metni 10.128 karakterden oluşuyor |
| `vocab 47` | Metinde 47 farklı karakter var (a-z, 0-9, ç, ğ, ı, ö, ş, ü, boşluk, nokta, satır sonu...) |
| `3 layers` | 3 adet transformer bloğu üst üste |
| `4 heads` | Her blokta 4 paralel attention kafası |
| `d_model 96` | Her karakter 96 boyutlu bir vektörle temsil ediliyor |
| `350.927 parameters` | Modelin öğrenebileceği toplam sayı adedi |
| `1500 steps` | 1500 kez ağırlık güncellemesi yapılacak |
| `batch 8` | Her adımda 8 farklı metin parçası kullanılacak |
| `block 64` | Model bir seferde en fazla 64 karakter geriye bakabilir |

### Bölüm 2: Eğitim ilerlemesi

```text
step     1/1500  loss 3.8743  lr 3.00E-005  0.4s
step    50/1500  loss 2.3483  lr 1.50E-003  5.3s
step   100/1500  loss 1.8631  lr 3.00E-003  9.5s
...
step  1500/1500  loss 0.2609  lr 3.00E-004  136.6s
```

**`loss`** en önemli sayı. "Model ne kadar şaşırdı?" demek.

- **3.87** ile başlıyor. Bu tesadüf değil: $\ln(47) = 3.85$. Yani model 47 karakter arasından **tamamen rastgele** seçiyor.
- **0.26** ile bitiyor. Artık bir sonraki karakteri neredeyse kesin biliyor.

**`lr`** (learning rate, öğrenme oranı) adım büyüklüğü. Önce küçükten başlayıp yükseliyor (ısınma), sonra yavaşça düşüyor. Neden böyle olduğu [Optimizer](../egitim/optimizer.md) sayfasında.

!!! question "Loss neden bazen yükseliyor?"
    `step 150: 1.3983` sonra `step 200: 1.4573` görebilirsiniz. Bu normaldir. Her adımda **rastgele** 8 metin parçası seçiliyor; bazıları daha zor olabilir. Genel eğilim aşağı doğru olduğu sürece sorun yok.

### Bölüm 3: Loss eğrisi

```text
  3.511 |*
  3.127 |
  2.744 | *
  2.361 |  *
  1.977 |    *
  1.594 |       *
  1.211 |           * ***** * *
  0.828 |                             ******  *
  0.444 |                                              *************
  0.253 |                                                           *****************
        +----------------------------------------------------------------------------
         1                                                                       1500
```

Bu grafiğin şekli **çok tipiktir** ve her sinir ağı eğitiminde görülür:

1. **Dik düşüş** (0-200 adım): Model en kolay şeyleri öğreniyor — hangi karakterler sık geçiyor, boşluk nerede olur.
2. **Orta bölge** (200-900): Kelimeler oluşuyor, "otomatik", "benzin" gibi kalıplar yerleşiyor.
3. **Yavaş kuyruk** (900-1500): İnce ayar. Satır formatı, sayı-birim eşleşmeleri.

### Bölüm 4: Özet ve kayıt

```text
ilk: 3.8743   en iyi: 0.1873   son 50 ortalama: 0.2553
csv : C:\...\loss.csv
model: C:\...\model.mtf (1371 KB)
```

- `loss.csv` — her adımın loss ve lr değeri, Excel'de grafik çizebilirsiniz
- `model.mtf` — modelin öğrendiği 350.927 sayı, 1,37 MB'lık ikili dosya

### Bölüm 5: Örnek üretim

```text
toyota corolla 2021 hibrit otomatik hatchback 120hp 10240bin tl
kia sedan 2021 dizel otomatik suv 180hp 1750bin tl
mg 4 2023 elektrik otomatik suv 299hp 150bin tl
mitsubishi asx 2020 benzin otomatik suv 100hp 1890bin tl
hyundai itonic 2022 benzin otomatik hatchback 100hp 780bin tl
```

Bu çıktıyı dikkatle inceleyin, çok şey anlatıyor:

!!! success "Doğru öğrenilenler"
    - **Format kusursuz**: marka → model → yıl → yakıt → vites → kasa → güç → fiyat
    - **Gerçek marka isimleri**: toyota, kia, mitsubishi hepsi doğru yazılmış
    - **Anlamlı eşleşmeler**: "elektrik" ile "otomatik" birlikte geliyor (veride hep öyle)
    - **Yıl aralığı**: 2018-2024 arası, veriyle uyumlu

!!! warning "Hatalar ve nedenleri"
    - **`kia sedan`** — "sedan" bir model değil kasa tipi. Model kelimelerin *rolünü* değil, sadece *pozisyonunu* öğrendi.
    - **`10240bin tl`** — Sayı mantığı yok. Model rakamların nasıl göründüğünü öğrendi, ne anlama geldiğini değil.
    - **`hyundai itonic`** — Veride yok! "hyundai" + "ioniq" karışımı. Bu bir **hata gibi görünse de aslında başarıdır**: model ezberlemiyor, üretiyor.

---

## İkinci çalıştırma: eğitim yok, anında üretim

```powershell
dotnet run -c Release -- generate "fiat egea 2021 "
```

```text
model     : C:\...\model.mtf (1371 KB, vocab 47)

fiat egea 2021 dizel otomatik sedan 183hp 2980bin tl
audi q5 2020 dizel otomatik sedan 180hp 1780bin tl
citroen c3 2020 benzin otomatik hatchback 110hp 820bin tl
```

**4 saniye** sürdü, çünkü eğitim yapılmadı — sadece `model.mtf` dosyasındaki ağırlıklar okundu. Öğrenme bir kez yapılır, kullanım sonsuz kere.

---

## Aynı sonucu alıyor muyum?

Eğitim `new Random(1337)` ile sabit tohumdan başlar, yani **her eğitim aynı sonucu verir**. Üretim ise `new Random()` kullanır — her çalıştırmada farklı metin alırsınız.

Üretimi de sabitlemek isterseniz [Program.cs](../referans/kod-haritasi.md) içindeki `new Random()` çağrısını `new Random(42)` yapın.

---

Sıradaki adım: [Komut Satırı](cli.md)
