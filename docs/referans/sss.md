# Sık Sorulan Sorular

## Genel

??? question "Bu model ChatGPT gibi sohbet edebilir mi?"
    Hayır. İki sebepten:

    1. **Ölçek**: 350 bin parametre vs. trilyonlarca. Karşılaştırma bile zor.
    2. **Eğitim türü**: ChatGPT önce genel metinle eğitilir (*pre-training*), sonra talimat takibi için özel olarak eğitilir (*instruction tuning*), sonra insan geri bildirimiyle ayarlanır (*RLHF*). Bizim modelimiz sadece ilk aşamanın minyatür halini yapıyor.

    Ama **mimari birebir aynı**. GPT-2'nin kodu ile bu kodun yapısı neredeyse özdeş.

??? question "Neden çıktılar bazen saçma?"
    350 bin parametre ve 10 KB veri ile bu normaldir. Model şunları öğrendi: karakter dizilimleri, kelime kalıpları, satır formatı. Öğrenmediği şeyler: anlam, mantık, sayı ilişkileri.

    `kia sedan` üretmesi bunun tipik örneği — "sedan" kelimesinin bir kasa tipi olduğunu bilmiyor, sadece o pozisyonda kelime geldiğini biliyor.

??? question "Model eğitim verisini kopyalıyor mu?"
    Kısmen. 10 KB veriyi 76 epoch boyunca gördüğü için bir miktar ezber kaçınılmaz.

    Ama tamamen kopyalamıyor: `hyundai itonic` gibi veride hiç geçmeyen kombinasyonlar üretiyor. Bu, gerçek genellemenin kanıtı.

    Kesin ölçmek için doğrulama seti gerekir → [Deneyler D4.1](../ileri/deneyler.md)

??? question "Bu projeyi ticari olarak kullanabilir miyim?"
    Kod eğitim amaçlıdır. Üretim ortamı için TorchSharp, ML.NET veya ONNX Runtime gibi olgun kütüphaneler kullanın. Bu kodda hata yönetimi, test, ölçeklenebilirlik ve güvenlik sertleştirmesi yok.

---

## Teknik

??? question "Neden `float[]` yerine `float[,]` kullanılmadı?"
    Üç sebep:

    1. **Hız**: C#'ta `float[,]` erişimi her seferinde sınır kontrolü yapar, düz dizi daha hızlıdır.
    2. **SIMD**: `Vector<float>` doğrudan düz diziden okuyabilir.
    3. **Serileştirme**: `MemoryMarshal.AsBytes` ile tek satırda diske yazılabilir.

    Erişim `Data[r * Cols + c]` şeklinde yapılır — **row-major** düzen.

??? question "Neden gradyanlar `=` değil `+=` ile yazılıyor?"
    Bir tensör birden fazla yerde kullanılabilir. Örneğin residual bağlantıda `x` hem attention'a girer hem de toplamaya katılır. Her kullanımdan gelen gradyanlar **toplanmalıdır**.

    `=` kullanmak sessiz bir hata olurdu: kod çalışır, loss düşer, ama model yanlış öğrenir.

??? question "Neden her eğitim aynı sonucu veriyor?"
    ```csharp
    var rng = new Random(1337);
    ```

    Sabit tohum. Ağırlık ilklemesi ve batch seçimi deterministik. Paralel matris çarpımı da deterministik çünkü her çıktı hücresi tek bir iş parçacığı tarafından sabit sırada hesaplanıyor.

    Bu, deney yaparken çok değerlidir.

??? question "Paralel kod neden yarış durumu (race condition) yaratmıyor?"
    Geri geçişte iki ayrı `Parallel.For` kullanılıyor:

    ```csharp
    Parallel.For(0, m, GradARow);   // her i kendi a.Grad satırına yazar
    Parallel.For(0, k, GradBRow);   // her p kendi b.Grad satırına yazar
    ```

    Tek döngüde yapılsaydı `b.Grad` paylaşılırdı ve yarış olurdu. İki döngüye ayırmak kilit ihtiyacını ortadan kaldırıyor.

??? question "`double` ve `float` neden karışık kullanılıyor?"
    **Karma hassasiyet** (mixed precision):

    - Ağırlıklar ve gradyanlar: `float` (bellek ve SIMD kazancı)
    - Toplamalar (softmax, layernorm, loss): `double` (hassasiyet)
    - Adam'ın `m` ve `v`: `double` (çok küçük değerler tutabilir)

    Sonuç: `double` sürümle birebir aynı loss eğrisi, 2 kat hız.

??? question "Neden özyineleme (recursion) yerine yığın kullanılıyor?"
    ```csharp
    var stack = new Stack<(Tensor Node, int Index)>();
    ```

    Hesap grafiği yüzlerce düğüm derinliğinde. Özyinelemeli DFS yığın taşması (`StackOverflowException`) riski taşır. Açık yığın kullanmak bunu tamamen ortadan kaldırır.

??? question "Neden maske değeri `-1e9`, `float.NegativeInfinity` değil?"
    `exp(-∞)` bazı durumlarda `NaN` üretebilir (özellikle `-∞ - (-∞)` işlemi softmax'ın max çıkarma adımında oluşabilir). `-1e9` ise `exp(-1e9) = 0` verir, güvenlidir.

---

## Eğitim

??? question "Loss `NaN` oldu, ne yapmalıyım?"
    Sırasıyla deneyin:

    1. Learning rate'i yarıya indirin
    2. `ClipGradients(1.0)` çağrısının yerinde olduğunu doğrulayın
    3. Warmup süresini uzatın (`Warmup = 300`)
    4. Verinizde garip karakterler olup olmadığını kontrol edin

??? question "Loss düşmüyor, sabit kalıyor"
    | Kontrol | Çözüm |
    |---|---|
    | Learning rate çok mu küçük? | 10 kat artırın |
    | `optimizer.ZeroGrad()` çağrılıyor mu? | Döngünün başında olmalı |
    | `optimizer.Step()` çağrılıyor mu? | Clipping'den sonra olmalı |
    | Veri çok mu karmaşık? | Daha düzenli veri deneyin |

??? question "Kaç adım eğitmeliyim?"
    Loss eğrisi düzleşene kadar. Bu projede 1500 adım yeterli; 3000 adım biraz daha iyileştirir ama ezberleme riski artar.

    Doğru yöntem: doğrulama loss'u yükselmeye başladığında durmak (*early stopping*).

??? question "Eğitim çok yavaş"
    | Kontrol | Etki |
    |---|---|
    | `-c Release` kullanıyor musunuz? | **5-10 kat** |
    | Kaç çekirdeğiniz var? | Doğrusal etki |
    | `BatchSize` çok mu büyük? | Doğrusal etki |
    | `BlockSize` çok mu büyük? | Karesel etki (attention) |

    En sık hata: Debug modunda çalıştırmak.

??? question "Eğitime kaldığı yerden devam edebilir miyim?"
    Şu an hayır. Sadece ağırlıklar kaydediliyor, Adam'ın `m`/`v` durumu kaydedilmiyor. Yüklenen modelle eğitime devam ederseniz optimizer sıfırdan başlar ve ilk adımlarda model bozulabilir.

    Eklemek için → [Deneyler D4.5](../ileri/deneyler.md)

---

## Üretim

??? question "Çıktı hep aynı geliyor"
    Temperature çok düşük olabilir. `GptModel.Generate` çağrısındaki `temperature: 0.8` değerini artırın.

    Ya da `topK` çok kısıtlayıcı olabilir (`topK: 1` greedy'ye eşdeğerdir).

??? question "Çıktı tamamen saçma"
    | Sebep | Kontrol |
    |---|---|
    | Yeterince eğitilmedi | Loss 1.0'ın altına indi mi? |
    | Temperature çok yüksek | 0.8 civarını deneyin |
    | Yanlış checkpoint yüklendi | Dosya adını kontrol edin |
    | Sözlük uyuşmazlığı | Veriyi değiştirip yeniden eğittiniz mi? |

??? question "Prompt'um neden dikkate alınmıyor gibi?"
    İki sebep olabilir:

    1. **Sözlükte olmayan karakterler** sessizce atılıyor (`@`, büyük harfler vb.)
    2. **Bağlam penceresi kayması**: 400 karakter ürettiğinizde, ilk verdiğiniz 20 karakterlik prompt çoktan pencereden çıkmıştır.

??? question "Üretim neden 4 saniye sürüyor?"
    Süre dağılımı:

    - ~2 sn: .NET başlatma + JIT derleme
    - ~1 sn: Checkpoint yükleme
    - ~1 sn: 400 token üretimi (her biri tam ileri geçiş)

    KV cache eklenirse üretim kısmı 10-50 kat hızlanır.

---

## Genişletme

??? question "Modeli nasıl büyütürüm?"
    `Program.cs` içindeki sabitleri artırın:

    ```csharp
    private const int EmbedSize = 256;   // 96'dan
    private const int Layers = 6;        // 3'ten
    private const int Heads = 8;         // 4'ten
    ```

    **Şart**: `EmbedSize % Heads == 0` olmalı.

    Süre kabaca `EmbedSize² × Layers` ile orantılı artar.

??? question "Kendi verimi nasıl kullanırım?"
    → [Kendi Verinle Eğitmek](../kullanim/kendi-verin.md)

    Özet: `data/input.txt` dosyasını değiştirin ve **mutlaka yeniden derleyin** (`dotnet build -c Release`).

??? question "Türkçe dışında dil kullanabilir miyim?"
    Evet, tokenizer sözlüğü veriden otomatik oluşturur. Sadece dikkat edin:

    - Çince/Japonca: binlerce karakter → sözlük patlar, model boyutunu artırmanız gerekir
    - Arapça/İbranice: sağdan sola yazım sorun değil, model karakterleri sırayla öğrenir
    - Emoji: UTF-16 surrogate pair sorunları çıkabilir (`char` tipi 16 bit)

??? question "Bu koda test eklemeli miyim?"
    Kesinlikle öğretici olur. En değerli test: **gradyan kontrolü** (gradient checking).

    Bir ağırlığı `ε` kadar oynatıp loss farkını ölçün, autograd'ın verdiği gradyanla karşılaştırın:

    $$\frac{\partial L}{\partial w} \approx \frac{L(w+\epsilon) - L(w-\epsilon)}{2\epsilon}$$

    İkisi yakınsa autograd doğru çalışıyor demektir. Bu, autograd yazanların standart doğrulama yöntemidir.

??? question "Sonraki adım ne olmalı?"
    1. [Deneyler](../ileri/deneyler.md) sayfasındaki alıştırmaları yapın
    2. Doğrulama seti ekleyin (en öğretici)
    3. Modeli 10 kat büyütüp sınırları görün
    4. Andrej Karpathy'nin "Let's build GPT" videosunu izleyin — aynı konsepti PyTorch ile anlatır
    5. Orijinal *Attention Is All You Need* makalesini okuyun

---

## Doküman

??? question "Bu dokümanı nasıl yayınlarım?"
    ```powershell
    pip install mkdocs-material
    mkdocs build
    ```

    `site/` klasöründeki statik HTML'i herhangi bir web sunucusuna koyabilirsiniz. GitHub Pages için:

    ```powershell
    mkdocs gh-deploy
    ```

??? question "Yerel önizleme nasıl yapılır?"
    ```powershell
    mkdocs serve
    ```

    `http://127.0.0.1:8000` adresinde açılır ve dosya değişikliklerini canlı yansıtır.
