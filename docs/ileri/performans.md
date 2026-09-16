# Performans

Bu proje başlangıçta 12 dakikada eğitiliyordu. Şimdi **2 dakika 17 saniye** sürüyor. Üç optimizasyonla 8 kat hızlandı.

---

## Hızlanma tablosu

| Aşama | 50 adım | Toplam | Kazanç |
|---|---|---|---|
| Başlangıç (`double`, tek çekirdek) | 19,0 sn | ~12 dk | — |
| \+ Paralel matris çarpımı | 12,5 sn | ~6 dk | 2,1× |
| \+ Batch birleştirme, `float`, SIMD | **5,3 sn** | **2 dk 17 sn** | **8×** |

!!! note "Adil karşılaştırma"
    İlk ölçüm `BlockSize = 48` ile yapıldı, son ölçüm `BlockSize = 64` ile. Yani son sürüm **%40 daha fazla iş** yaparken bu hıza ulaşıyor. Aynı iş yükünde kazanç yaklaşık 8 kattır.

---

## Optimizasyon 1: Paralel matris çarpımı

Matris çarpımı toplam sürenin %90'ını yiyordu. 14 çekirdekli bir makinede tek çekirdek kullanmak israftı.

### Zorluk: veri yarışı

Naif paralelleştirme yanlış sonuç verir. Geri geçişte hem `a.Grad` hem `b.Grad` güncellenir:

```csharp
// TEHLİKELİ: farklı iş parçacıkları aynı b.Grad[j] hücresine yazar
for (int i = 0; i < m; i++)          // paralelleştirilirse
    for (int j = 0; j < n; j++)
        b.Grad[bRow + j] += av * og[oRow + j];   // YARIŞ!
```

### Çözüm: iki ayrı döngü

```csharp
// dA = dC * B^T : i satırları birbirinden bağımsız
void GradARow(int i)
{
    int oRow = i * n, aRow = i * k;
    for (int p = 0; p < k; p++)
        a.Grad[aRow + p] += Dot(og, oRow, bd, p * n, n);
}

// dB = A^T * dC : p satırları birbirinden bağımsız
void GradBRow(int p)
{
    int bRow = p * n;
    for (int i = 0; i < m; i++)
    {
        float av = ad[i * k + p];
        if (av != 0f)
            Axpy(b.Grad, bRow, og, i * n, av, n);
    }
}

Parallel.For(0, m, GradARow);
Parallel.For(0, k, GradBRow);
```

Her iş parçacığı **kendi satırına** yazar. Kilit yok, atomik işlem yok, yavaşlama yok.

!!! success "Bonus: determinizm korunuyor"
    Her çıktı hücresi tek bir iş parçacığı tarafından, sabit sırada hesaplanır. Bu yüzden paralel sürüm seri sürümle **birebir aynı** sonucu verir. Kayan nokta toplama sırası değişseydi sonuçlar hafif farklılaşırdı.

### Küçük matrisler için eşik

```csharp
bool parallel = (long)m * k * n >= 64_000;
```

Çok küçük matrislerde `Parallel.For`'un kendi maliyeti işten büyük olur. Eşik altında seri çalışır.

---

## Optimizasyon 2: Batch'i tek matriste birleştirme

### Önce: 8 ayrı geçiş

```csharp
for (int b = 0; b < BatchSize; b++)
{
    var loss = model.Loss(inputs[b], targets[b]);
    Ops.Scale(loss, 1.0 / BatchSize).Backward();
}
```

8 ayrı ileri geçiş, 8 ayrı geri geçiş. Her matris çarpımı `[64 × 96] × [96 × 96]` — paralelleştirmek için fazla küçük.

### Sonra: tek büyük matris

8 dizi alt alta yığılır: `[512 × 96]`

```csharp
var (inputs, targets) = SampleBatch(data, rng);
var loss = model.Loss(inputs, targets);
loss.Backward();
```

Artık çarpımlar `[512 × 96] × [96 × 96]` — 8 kat daha büyük, paralelleştirme çok daha verimli.

### Nasıl mümkün oldu?

Çoğu katman **satır bazlıdır** — her satırı bağımsız işler:

| Katman | Satırlar arası etkileşim | Yığınlanabilir mi |
|---|---|---|
| Linear | Yok | ✅ |
| LayerNorm | Yok (satır içi normalize) | ✅ |
| GELU | Yok (eleman bazlı) | ✅ |
| Embedding | Yok | ✅ |
| **Attention** | **Var!** | ❌ |

Attention satırlar arasında bilgi taşır, bu yüzden dizi dizi ayrılmak zorunda:

```csharp
for (int b = 0; b < batchCount; b++)
{
    int rowStart = b * seqLen;
    var qh = Ops.Slice(q, rowStart, seqLen, offset, _headSize);
    ...
}
```

Bunun için `Ops.Slice` satır+sütun bloğu alacak şekilde genelleştirildi ve `Ops.ConcatRows` eklendi.

### Ek kazanç: tek geri geçiş

Adım başına 8 `Backward()` yerine **1** tane. Topolojik sıralama, grafik gezintisi ve kapanış çağrıları 8 kat azaldı.

---

## Optimizasyon 3: `double` → `float`

```csharp
public readonly float[] Data;
public readonly float[] Grad;
```

### Kazançlar

| Etki | Açıklama |
|---|---|
| Bellek yarıya indi | 8 bayt → 4 bayt |
| Önbellek verimi arttı | Aynı L1/L2 önbelleğe 2× veri sığar |
| SIMD genişliği 2× oldu | AVX2: 4 `double` yerine **8 `float`** |

### Hassasiyet endişesi

`float` yaklaşık 7 anlamlı basamak tutar. Bazı yerlerde bu yetersizdir:

```csharp
// Toplama işlemleri double içinde birikir
double sum = 0.0;
for (int c = 0; c < x.Cols; c++)
{
    float e = MathF.Exp(x.Data[b + c] - max);
    probs[b + c] = e;
    sum += e;              // (1)!
}
```

1. 512 terimin toplamında `float` hassasiyeti kaybolur. `double` birikim kullanılır, sonuç `float`'a çevrilir.

Bu **karma hassasiyet** (mixed precision) tekniğidir ve nerede uygulandı:

| Yer | Birikim tipi |
|---|---|
| Softmax toplamı | `double` |
| LayerNorm ortalama/varyans | `double` |
| Cross-entropy toplamı | `double` |
| Gradient norm | `double` |
| Adam `m` ve `v` | `double` |
| Parametreler | `float` |

!!! success "Sonuç: hassasiyet kaybı yok"
    Loss eğrisi `double` sürümle **birebir aynı** yerden başlayıp aynı yere indi (3,87 → 0,19). Sayısal bozulma gözlenmedi.

### Modern uygulamada

Büyük modeller daha da ileri gider:

| Format | Bit | Kullanım |
|---|---|---|
| FP32 | 32 | Klasik |
| **FP16 / BF16** | **16** | **Modern eğitim standardı** |
| FP8 | 8 | H100 ve sonrası |
| INT4/INT8 | 4-8 | Sadece çıkarım (kuantizasyon) |

---

## Optimizasyon 4: SIMD

SIMD = Single Instruction, Multiple Data. Tek komutla 8 sayıyı aynı anda işlemek.

### Axpy çekirdeği

```csharp
private static void Axpy(float[] dst, int dstOffset, float[] src, int srcOffset, float scalar, int count)
{
    int width = Vector<float>.Count;        // (1)!
    var vScalar = new Vector<float>(scalar);
    int j = 0;

    for (; j <= count - width; j += width)
    {
        var acc = new Vector<float>(dst, dstOffset + j)
                + (new Vector<float>(src, srcOffset + j) * vScalar);
        acc.CopyTo(dst, dstOffset + j);
    }

    for (; j < count; j++)                  // (2)!
        dst[dstOffset + j] += src[srcOffset + j] * scalar;
}
```

1. AVX2'de 8, AVX-512'de 16, ARM NEON'da 4. Donanıma göre otomatik.
2. Kalan elemanlar (96 % 8 = 0 olduğu için burada çalışmaz ama genel güvenlik)

### Dot çekirdeği

```csharp
private static float Dot(float[] x, int xOffset, float[] y, int yOffset, int count)
{
    int width = Vector<float>.Count;
    var acc = Vector<float>.Zero;
    int j = 0;

    for (; j <= count - width; j += width)
        acc += new Vector<float>(x, xOffset + j) * new Vector<float>(y, yOffset + j);

    float sum = Vector.Sum(acc);
    for (; j < count; j++)
        sum += x[xOffset + j] * y[yOffset + j];

    return sum;
}
```

!!! tip "`System.Numerics.Vector<T>` neden güzel?"
    Donanımı otomatik algılar. AVX-512 varsa 16 float, AVX2 varsa 8, ARM'de NEON kullanır. Kod değişmez.

    `-c Release` şart — Debug modunda JIT bu vektörleştirmeyi yapmaz.

---

## Bellek düzeni

```csharp
public double this[int r, int c] => Data[r * Cols + c];
```

İki boyutlu dizi (`float[,]`) yerine düz dizi kullanılır. Sebebi:

- `float[,]` erişiminde CLR her seferinde sınır kontrolü yapar
- Düz dizi bellekte kesintisiz durur → önbellek dostu
- `MemoryMarshal.AsBytes` ile doğrudan diske yazılabilir
- `Vector<float>` doğrudan diziden okuyabilir

**Row-major** düzen: aynı satırın elemanları bellekte yan yana. Matris çarpımının iç döngüsü satır boyunca ilerlediği için bu ideal.

---

## Kalan darboğazlar

Daha da hızlandırmak isterseniz sıradaki adımlar:

| Optimizasyon | Tahmini kazanç | Zorluk |
|---|---|---|
| **Tensör havuzu** (allocation azaltma) | 1,3× | Orta |
| **Blok matris çarpımı** (cache tiling) | 1,5× | Orta |
| **Attention'ı batch'li yapmak** | 1,2× | Yüksek |
| **Fused işlemler** (LayerNorm+Linear) | 1,2× | Yüksek |
| **KV cache** (sadece üretim) | 10-50× | Orta |

### Allocation sorunu

Her adımda ~500 `Tensor` nesnesi oluşturuluyor ve çöp toplayıcıya gidiyor. Bir havuz (object pool) kullanmak GC baskısını ciddi azaltırdı.

```csharp
// Şu an: her işlem yeni tensör
var outTensor = new Tensor(a.Rows, b.Cols);
```

!!! info "Neden yapılmadı?"
    Kodun okunabilirliği öncelikliydi. Havuz yönetimi, tensörlerin ne zaman serbest bırakılacağını takip etmeyi gerektirir ve autograd grafiğiyle birleşince karmaşıklaşır.

---

## Ölçüm yöntemi

Kendi optimizasyonunuzu test etmek için:

```powershell
dotnet build -c Release
dotnet run -c Release --no-build -- train
```

`step 50` satırındaki süreye bakın — bu, JIT ısınması bittikten sonraki gerçek hızı gösterir. İlk adım her zaman yavaştır.

!!! warning "Debug modunda ölçmeyin"
    Debug derlemesi 5-10 kat yavaştır. SIMD devre dışıdır, sınır kontrolleri kaldırılmaz, satır içi genişletme (inlining) yapılmaz.

---

Sıradaki adım: [GPU](gpu.md)
