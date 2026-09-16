# Gerekli Matematik

Korkmayın — bu projede kullanılan matematik lise düzeyini pek aşmıyor. Dört konu yeter.

---

## 1. Vektör ve matris

### Vektör

Sıralı sayı listesi. Bu projede her karakter 96 sayılık bir vektörle temsil edilir:

$$\mathbf{v} = [0.13, -0.82, 0.05, \dots, 0.44]$$

### Matris

Sayı tablosu. Satır ve sütunu vardır. Bir metin parçası matristir:

$$X = \begin{bmatrix} 0.13 & -0.82 & \cdots & 0.44 \\ 0.91 & 0.07 & \cdots & -0.31 \\ \vdots & & & \vdots \end{bmatrix}_{64 \times 96}$$

64 satır = 64 karakter, 96 sütun = her karakterin vektör boyutu.

### Kodda nasıl duruyor?

C#'ta iki boyutlu dizi yerine **tek boyutlu dizi** kullanılır (çok daha hızlıdır):

```csharp
// src/Tensor.cs
public readonly float[] Data;

public double this[int r, int c]
{
    get => Data[r * Cols + c];   // (1)!
}
```

1. `r`. satırın `c`. sütunu, düz dizide `r * Cols + c` indeksindedir. Buna **row-major** düzen denir.

---

## 2. Matris çarpımı

Projenin en çok zaman harcayan işlemi budur.

$$C = A \times B$$

$A$ boyutu $[m \times k]$, $B$ boyutu $[k \times n]$ ise $C$ boyutu $[m \times n]$ olur.

**Kural**: $C$'nin $i$. satır $j$. sütunundaki eleman, $A$'nın $i$. satırı ile $B$'nin $j$. sütununun nokta çarpımıdır.

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

### Somut örnek

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1\cdot5+2\cdot7 & 1\cdot6+2\cdot8 \\ 3\cdot5+4\cdot7 & 3\cdot6+4\cdot8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

### Neden bu kadar önemli?

Çünkü **bir katmandaki tüm nöronların hesabı tek matris çarpımıdır**. 64 karakter × 96 nöron = 6.144 ağırlıklı toplam, tek işlemde.

!!! info "İç boyut uyuşmalı"
    $[64 \times 96] \times [96 \times 96]$ ✓ (96 = 96)
    $[64 \times 96] \times [24 \times 96]$ ✗ (96 ≠ 24)

    Kod bunu kontrol eder:
    ```csharp
    if (a.Cols != b.Rows)
        throw new ArgumentException($"Shape mismatch: ...");
    ```

---

## 3. Türev ve zincir kuralı

### Türev nedir?

"Girdiyi birazcık değiştirirsem çıktı ne kadar değişir?"

$f(x) = x^2$ için $f'(x) = 2x$. Yani $x=3$ noktasında, $x$'i 0,001 artırırsanız $f$ yaklaşık $6 \times 0.001 = 0.006$ artar.

Sinir ağında sorduğumuz soru aynen budur: *"Bu ağırlığı birazcık değiştirirsem loss ne kadar değişir?"*

### Zincir kuralı

İç içe fonksiyonların türevi:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Örnek**: $y = (3x + 1)^2$

- $u = 3x+1$, $\frac{du}{dx} = 3$
- $y = u^2$, $\frac{dy}{du} = 2u$
- $\frac{dy}{dx} = 2u \cdot 3 = 6(3x+1)$

### Neden bu projenin temeli?

Bir sinir ağı, iç içe geçmiş yüzlerce fonksiyondur:

$$L = \text{loss}(\text{katman}_3(\text{katman}_2(\text{katman}_1(x))))$$

$\text{katman}_1$'in ağırlıklarının loss'a etkisini bulmak için zincir kuralını **sondan başa** uygularız. Buna **geri yayılım** denir ve [Autograd](../egitim/autograd.md) sayfasında ayrıntılı anlatılır.

### Kısmi türev

Birden fazla değişken varsa, birini değiştirip diğerlerini sabit tutarız. Gösterimi $\partial$ (del):

$$\frac{\partial L}{\partial w_{42}}$$

"42 numaralı ağırlığı değiştirirsem loss ne olur?" Bu sayıya **gradyan** denir.

---

## 4. Olasılık ve softmax

### Softmax: sayıları olasılığa çevirme

Modelin çıktısı ham sayılardır (**logit** denir), örneğin:

```text
[2.1, -0.5, 3.7, 0.2]
```

Bunları olasılığa çevirmek için üç şart var: hepsi pozitif olmalı, toplamları 1 olmalı, sıralama korunmalı.

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Adım adım:

| Adım | Değerler |
|---|---|
| Ham logit | 2.1, -0.5, 3.7, 0.2 |
| $e^x$ | 8.17, 0.61, 40.4, 1.22 |
| Toplam | 50.4 |
| Bölünce | **0.162, 0.012, 0.802, 0.024** |

Toplam 1.0 ✓

### Sayısal kararlılık hilesi

$e^{100}$ taşma yapar (`float` sonsuz olur). Çözüm: en büyük değeri çıkarmak. Sonuç matematiksel olarak aynıdır ama taşma olmaz.

```csharp
// src/Ops.cs — SoftmaxRows
float max = float.NegativeInfinity;
for (int c = 0; c < x.Cols; c++)
    max = MathF.Max(max, x.Data[b + c]);

for (int c = 0; c < x.Cols; c++)
{
    float e = MathF.Exp(x.Data[b + c] - max);   // (1)!
    ...
}
```

1. En büyük değer çıkarıldığı için üs en fazla 0 olur, $e^0 = 1$. Taşma imkânsız.

### Logaritma ve loss

Cross-entropy şudur:

$$L = -\log(p_{\text{doğru}})$$

Neden logaritma?

- $p = 1$ (kesin doğru) → $-\log(1) = 0$ — hata yok
- $p = 0.5$ → $0.69$
- $p = 0.01$ → $4.6$ — büyük ceza
- $p \to 0$ → $\infty$ — sonsuz ceza

Logaritma, "çok emin olup yanılmayı" ağır cezalandırır. Bu istediğimiz davranıştır.

---

## Özet tablo

| Kavram | Nerede kullanılıyor | Kod |
|---|---|---|
| Matris çarpımı | Her katmanda | `Ops.MatMul` |
| Transpoze | Attention skorlarında | `Ops.Transpose` |
| Zincir kuralı | Geri yayılımda | `Tensor.Backward` |
| Kısmi türev | Her ağırlık için | `Tensor.Grad` |
| Softmax | Attention ve çıktıda | `Ops.SoftmaxRows` |
| Logaritma | Loss hesabında | `Ops.CrossEntropy` |
| Ortalama/varyans | Normalizasyonda | `Ops.LayerNorm` |

Bu kadar. Daha fazlası gerekmiyor.

---

Sıradaki adım: [Mimari Genel Bakış](../mimari/genel-bakis.md)
