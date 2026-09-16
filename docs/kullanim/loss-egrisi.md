# Loss Eğrisi

**Dosya:** `src/LossChart.cs`, `Program.cs` — `ShowLoss`

Loss eğrisi, eğitimin röntgenidir. Neyin yolunda gittiğini veya gitmediğini bir bakışta gösterir.

---

## Görüntüleme

```powershell
dotnet run -c Release -- loss
dotnet run -c Release -- loss eski-deney.csv
```

```text
kaynak: C:\...\loss.csv (1500 adım)

  3.511 |*
  3.319 |
  3.127 |
  2.936 |
  2.744 | *
  2.552 |
  2.361 |  *
  2.169 |   *
  1.977 |    *
  1.786 |     **
  1.594 |       *
  1.403 |        *** *
  1.211 |           * ***** * *
  1.019 |                  * * *******
  0.828 |                             ******  *
  0.636 |                                   ** ********
  0.444 |                                              *************
  0.253 |                                                           *****************
        +----------------------------------------------------------------------------
         1                                                                       1500
         adım ->
ilk: 3.8743   en iyi: 0.1873   son 50 ortalama: 0.2553
```

---

## Grafik nasıl çiziliyor?

```csharp
public static string Render(IReadOnlyList<double> values, int width = 76, int height = 18)
{
    int columns = Math.Min(width, values.Count);
    var points = new double[columns];

    // Her sütun, adım aralığının ortalamasıdır; bu aynı zamanda gürültüyü yumuşatır.
    for (int c = 0; c < columns; c++)
    {
        int start = (int)((long)c * values.Count / columns);
        int end = (int)((long)(c + 1) * values.Count / columns);
        if (end <= start) end = start + 1;

        double sum = 0.0;
        for (int i = start; i < end; i++) sum += values[i];
        points[c] = sum / (end - start);
    }
    ...
}
```

1500 değeri 76 sütuna sığdırmak için her sütun ~20 adımın **ortalamasını** gösterir. Bu hem sığdırma hem de yumuşatma (smoothing) sağlar.

Ardından her nokta dikey konuma eşlenir:

```csharp
int row = (int)Math.Round((max - points[c]) / (max - min) * (height - 1));
grid[row][c] = '*';
```

`max` en üstte, `min` en altta. Eksen etiketleri de bu aralığa göre hesaplanır.

!!! note "Neden ASCII?"
    Bağımlılık gerektirmez, terminalde anında görünür, SSH üzerinden çalışır. Gerçek grafik isterseniz `loss.csv` dosyasını Excel veya Python'da açabilirsiniz.

---

## CSV dosyası

```text
step,loss,lr
1,3.874300,3E-05
2,3.812100,6E-05
3,3.755400,9E-05
...
1500,0.260900,0.0003
```

### Excel'de grafik

1. `loss.csv` dosyasını açın
2. `step` ve `loss` sütunlarını seçin
3. Ekle → Grafik → Dağılım (düz çizgi)
4. Dikey ekseni **logaritmik** yapın — loss eğrileri log ölçekte çok daha okunaklıdır

### Python ile

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("loss.csv")
plt.plot(df["step"], df["loss"], alpha=0.3, label="ham")
plt.plot(df["step"], df["loss"].rolling(50).mean(), label="50 adım ortalama")
plt.yscale("log")
plt.xlabel("adım"); plt.ylabel("loss"); plt.legend()
plt.show()
```

---

## Eğri okuma rehberi

### Sağlıklı eğri

```text
    |*
    | *
    |   **
    |      ****
    |           *********
    |                    ****************
```

Hızlı düşüş, sonra yavaşlayan ama devam eden iniş. Beklenen davranış.

**Üç aşama:**

| Aşama | Adım | Ne oluyor |
|---|---|---|
| Dik düşüş | 0-200 | Karakter frekansları, boşluk yeri |
| Orta bölge | 200-900 | Kelimeler, yaygın kalıplar |
| Yavaş kuyruk | 900-1500 | Format ince ayarı |

### Sorunlu eğriler

=== "Düz çizgi"

    ```text
    |*****************************
    |
    |
    ```

    **Sebep:** Learning rate çok küçük veya gradyanlar akmıyor.

    **Çözüm:** LR'yi 10 kat artırın. Düzelmezse `ZeroGrad` çağrısını ve residual bağlantıları kontrol edin.

=== "Patlama"

    ```text
    |            *****
    |        ****
    |*   ****
    | ***
    ```

    **Sebep:** Learning rate çok büyük.

    **Çözüm:** LR'yi yarıya indirin, `ClipGradients` eşiğini düşürün, warmup'ı uzatın.

=== "Erken plato"

    ```text
    |*
    |  *
    |    ***********************
    ```

    **Sebep:** Model kapasitesi yetersiz veya LR sönümü çok hızlı.

    **Çözüm:** `EmbedSize` veya `Layers` artırın; sönüm programını yumuşatın.

=== "Aşırı dalgalanma"

    ```text
    |* * *   *  *
    | * * * * **
    |  *   *  *
    ```

    **Sebep:** Batch çok küçük, gradyan tahmini gürültülü.

    **Çözüm:** `BatchSize` artırın.

---

## Deneyleri karşılaştırma

```powershell
# Deney 1
dotnet run -c Release -- train
Copy-Item loss.csv deney-3katman.csv

# Layers = 6 yapın, yeniden derleyin
dotnet build -c Release
dotnet run -c Release --no-build -- train
Copy-Item loss.csv deney-6katman.csv

# Karşılaştırın
dotnet run -c Release -- loss deney-3katman.csv
dotnet run -c Release -- loss deney-6katman.csv
```

Özet satırı hızlı karşılaştırma sağlar:

```text
ilk: 3.8743   en iyi: 0.1873   son 50 ortalama: 0.2553
```

| Metrik | Ne söyler |
|---|---|
| **ilk** | Başlangıç noktası ($\ln(\text{vocab})$ olmalı) |
| **en iyi** | Ulaşılan en düşük değer (tek şanslı batch olabilir) |
| **son 50 ortalama** | **En güvenilir ölçü** — gürültüden arınmış son durum |

!!! tip "Hangi sayıya bakmalı?"
    **Son 50 ortalama.** "En iyi" değeri yanıltıcıdır çünkü tek bir kolay batch'ten gelmiş olabilir.

---

## Loss değerlerinin anlamı

| Loss | Perplexity | Model ne durumda |
|---|---|---|
| 3.85 | 47.0 | Tam rastgele |
| 2.50 | 12.2 | Harf frekanslarını öğrendi |
| 1.50 | 4.5 | Kelime parçaları oluşuyor |
| 1.00 | 2.7 | Kelimeler doğru |
| 0.50 | 1.65 | Format oturdu |
| **0.26** | **1.30** | **Bizim sonucumuz** |
| 0.05 | 1.05 | Muhtemelen ezberlemiş |

!!! danger "Çok düşük loss iyi değildir"
    Loss 0,05'e inerse model muhtemelen eğitim verisini **ezberlemiştir**. Üretilen metinler veriden birebir kopya olur, yeni kombinasyon üretmez.

    Doğrulama seti olmadan bunu kesin söyleyemeyiz — ama üretilen metinlerin veride birebir geçip geçmediğini kontrol ederek sezebilirsiniz.

---

## Learning rate eğrisi

CSV'nin üçüncü sütunu `lr`. Onu da çizerseniz warmup + kosinüs sönümünü görürsünüz:

```text
3e-3 |      ___
     |    /    ---___
     |  /            ---____
3e-4 |/                     ----
     +----+------------------------
     0   100                   1500
```

Loss'un düşüş hızının, LR'nin yüksek olduğu bölgede en fazla olduğunu fark edeceksiniz. Bu beklenen davranıştır.

---

Sıradaki adım: [Kendi Verinle Eğitmek](kendi-verin.md)
