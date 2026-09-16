# Sözlük

Yapay zeka terimlerinin Türkçe açıklamaları ve bu projedeki karşılıkları.

---

## A

**Adam**
: *Adaptive Moment estimation*. Gradyanın hem ortalamasını hem karesinin ortalamasını izleyerek her ağırlığa kendi adım boyunu veren optimizer. → [Optimizer](../egitim/optimizer.md)

**Aktivasyon fonksiyonu**
: Doğrusallığı kıran fonksiyon. Onsuz derin ağ tek katmana eşdeğer olur. Bu projede GELU. → [Sinir Ağı](../temeller/sinir-agi.md)

**Attention (dikkat)**
: Her tokenin kendinden öncekilere bakıp ihtiyacı olan bilgiyi çektiği mekanizma. Transformer'ın kalbi. → [Attention](../mimari/attention.md)

**Autograd**
: Otomatik türev. İleri geçişte hesap grafiği kurup geri geçişte zincir kuralını otomatik uygulayan sistem. → [Autograd](../egitim/autograd.md)

**Ağırlık (weight)**
: Modelin öğrendiği sayı. Bu projede 350.927 adet var.

---

## B

**Backpropagation (geri yayılım)**
: Loss'tan başlayıp geriye doğru her ağırlığın gradyanını hesaplama işlemi.

**Batch**
: Bir adımda birlikte işlenen örnek grubu. Bu projede 8 dizi.

**Bias**
: Nöronun çıktısına eklenen sabit kaydırma terimi.

**Bias düzeltmesi (bias correction)**
: Adam'da `m` ve `v`'nin sıfırdan başlamasından kaynaklanan yanlılığı gideren çarpan.

**BlockSize**
: Bağlam penceresi uzunluğu. Bu projede 64 karakter.

**BPE (Byte Pair Encoding)**
: Sık geçen karakter çiftlerini birleştirerek alt-kelime sözlüğü oluşturan tokenizasyon yöntemi. GPT ailesi kullanır.

---

## C-Ç

**Checkpoint**
: Model ağırlıklarının diske kaydedilmiş hali. → [Checkpoint](../kullanim/checkpoint.md)

**Context window (bağlam penceresi)**
: Modelin bir seferde görebildiği maksimum token sayısı.

**Cross-entropy**
: Tahmin dağılımı ile gerçek dağılım arasındaki farkı ölçen loss fonksiyonu. → [Loss](../egitim/loss.md)

**Causal mask (nedensel maske)**
: Modelin geleceği görmesini engelleyen üçgen maske.

**Çıkarım (inference)**
: Eğitilmiş modeli kullanma aşaması. Eğitimin tersine hızlı ve ucuz.

---

## D

**d_model**
: Model boyutu, her tokenin vektör uzunluğu. Bu projede 96.

**Derin öğrenme**
: Çok katmanlı sinir ağlarıyla yapılan makine öğrenmesi.

**Dropout**
: Eğitimde rastgele nöronları kapatarak ezberlemeyi azaltan teknik. Bu projede yok.

---

## E

**Embedding**
: Ayrık bir id'yi (karakter, kelime) sürekli bir vektöre çeviren öğrenilebilir tablo. → [Embedding](../mimari/embedding.md)

**Epoch**
: Eğitim verisinin tamamının bir kez görülmesi.

**Eğim** → **Gradyan**

---

## F

**Feed-forward (FFN)**
: Her tokeni bağımsız işleyen iki katmanlı ağ. Genişlet → aktivasyon → daralt. Parametrelerin çoğu buradadır.

**Float / FP32 / FP16**
: Kayan nokta sayı formatları. Bu projede `float` (32 bit) kullanılır, birikimler `double` (64 bit).

**Forward pass (ileri geçiş)**
: Girdiden tahmine doğru hesaplama.

---

## G

**GELU**
: *Gaussian Error Linear Unit*. Yumuşak aktivasyon fonksiyonu. GPT modellerinin standardı.

**Gradient clipping**
: Gradyan normunu bir eşikle sınırlayarak patlamayı önleme.

**Gradyan (gradient)**
: `∂L/∂w`. "Bu ağırlığı değiştirirsem loss ne kadar değişir?" sorusunun cevabı.

**Gradyan inişi (gradient descent)**
: Loss'u azaltmak için gradyanın tersi yönde adım atma.

**GPT**
: *Generative Pre-trained Transformer*. Sadece çözücü (decoder-only) transformer mimarisi.

**Greedy sampling**
: Her adımda en olası tokeni seçme. Kısır döngüye yol açar.

---

## H

**Hesap grafiği (computation graph)**
: İşlemler ve aralarındaki bağımlılıkların oluşturduğu yönlü graf. Autograd bunun üzerinde çalışır.

**Head (kafa)**
: Attention'ın paralel çalışan alt birimlerinden biri. Bu projede 4 tane.

**Hiperparametre**
: Eğitimden önce elle belirlenen ayar (learning rate, katman sayısı vb.). Ağırlıklardan farkı: öğrenilmez.

---

## I-İ

**Inference** → **Çıkarım**

**İleri geçiş** → **Forward pass**

---

## K

**Katman (layer)**
: Yan yana duran nöron grubu; matematiksel olarak bir matris çarpımı.

**Kernel fusion**
: Birden çok işlemi tek GPU kerneline birleştirerek overhead azaltma.

**KV cache**
: Üretim sırasında önceki tokenlerin Key/Value vektörlerini saklayarak tekrar hesaplamayı önleme.

**Key (K)**
: Attention'da her tokenin "ben şuyum" etiketi.

---

## L

**LayerNorm**
: Her token vektörünü ortalama 0, varyans 1 olacak şekilde normalize eden katman.

**Learning rate (öğrenme oranı)**
: Gradyan yönünde atılan adımın büyüklüğü. En kritik hiperparametre.

**Logit**
: Softmax'tan önceki ham skor. Olasılık değil.

**Loss (kayıp)**
: Modelin hatasını ölçen tek sayı. Eğitimin tek geri bildirimi.

---

## M

**Matris çarpımı**
: Sinir ağlarının temel işlemi. Bir katmandaki tüm nöronların hesabı.

**Momentum**
: Gradyanların hareketli ortalaması. Gürültüyü yumuşatır, hızı artırır.

**Multi-head attention**
: Attention'ı birden çok paralel kafaya bölme. Her kafa farklı ilişki öğrenir.

---

## N

**Nöron**
: Girdilerin ağırlıklı toplamını alıp aktivasyondan geçiren birim.

**Nucleus sampling** → **Top-p**

---

## O-Ö

**Optimizer**
: Gradyanları kullanarak ağırlıkları güncelleyen algoritma.

**Overfitting (aşırı uyum / ezberleme)**
: Modelin eğitim verisini ezberleyip yeni veriye genelleyememesi.

**Örnekleme (sampling)**
: Olasılık dağılımından rastgele token seçme.

---

## P

**Parametre** → **Ağırlık**

**Perplexity (şaşkınlık)**
: `e^loss`. "Model her adımda kaç seçenek arasında bocalıyor?"

**Position embedding**
: Token'ın sıradaki yerini kodlayan vektör. Attention sırayı göremediği için gerekli.

**Post-norm / Pre-norm**
: LayerNorm'un residual toplamadan sonra mı önce mi uygulandığı. Bu proje pre-norm.

**Prompt**
: Modele verilen başlangıç metni.

---

## Q

**Query (Q)**
: Attention'da bir tokenin "ne arıyorum?" sorusu.

---

## R

**ReLU**
: `max(0, x)`. En basit aktivasyon fonksiyonu.

**Residual bağlantı (skip connection)**
: `y = x + f(x)`. Gradyanın derin ağlarda kaybolmasını önler.

**RoPE**
: *Rotary Position Embedding*. Vektörleri pozisyona göre döndüren modern pozisyon kodlaması.

---

## S-Ş

**Sampling** → **Örnekleme**

**Self-attention**
: Q, K ve V'nin aynı diziden üretildiği attention.

**SIMD**
: *Single Instruction Multiple Data*. Tek komutla birden çok sayıyı işleme. → [Performans](../ileri/performans.md)

**Softmax**
: Ham skorları toplamı 1 olan olasılıklara çeviren fonksiyon.

**Sözlük (vocabulary)**
: Modelin tanıdığı tüm tokenlerin listesi. Bu projede 47 karakter.

---

## T

**Temperature (sıcaklık)**
: Örnekleme dağılımının keskinliğini ayarlayan parametre. Düşük = tekrarcı, yüksek = yaratıcı.

**Tensör**
: Çok boyutlu sayı dizisi. Bu projede 2 boyutlu (matris).

**Token**
: Metnin bölündüğü en küçük birim. Bu projede bir karakter.

**Top-k**
: En olası k tokeni bırakıp gerisini eleme.

**Top-p (nucleus)**
: Kümülatif olasılık p'ye ulaşana kadar token alma. Adaptif top-k.

**Topolojik sıralama**
: Grafik düğümlerini bağımlılık sırasına göre dizme. Autograd için şart.

**Transformer**
: 2017'de tanıtılan, attention tabanlı mimari. Modern yapay zekanın temeli.

---

## V

**Value (V)**
: Attention'da eşleşme olduğunda taşınan asıl bilgi.

**Vanishing gradient (kaybolan gradyan)**
: Derin ağlarda gradyanın geriye giderken sıfıra yaklaşması. Residual bağlantılar çözer.

---

## W

**Warmup (ısınma)**
: Eğitimin başında learning rate'i kademeli artırma.

**Weight decay**
: Ağırlıkları küçük tutmaya zorlayan düzenlileştirme. Bu projede yok.

**Weight tying**
: Embedding tablosu ile çıkış katmanının aynı ağırlıkları paylaşması.

---

## Z

**Zincir kuralı (chain rule)**
: `dy/dx = (dy/du)(du/dx)`. Geri yayılımın matematiksel temeli.
