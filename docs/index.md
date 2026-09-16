# MiniTransformer

!!! abstract "Bu doküman kimin için?"
    Hiç yapay zeka bilmeyen biri için. Sinir ağı nedir bilmiyorsanız, "attention" kelimesini ilk kez duyuyorsanız, matris çarpımını unuttuysanız — hiç sorun değil. Her şeyi baştan anlatacağız.

**MiniTransformer**, ChatGPT'nin arkasındaki teknolojinin (Transformer mimarisi) **sıfırdan C# ile yazılmış**, çalışan, eğitilebilen minyatür bir versiyonudur.

Hiçbir yapay zeka kütüphanesi kullanılmaz. PyTorch yok, TensorFlow yok, NuGet paketi bile yok. Matris çarpımından türev hesabına kadar her şey elle yazılmıştır. Toplamda yaklaşık **1200 satır kod**.

---

## Ne yapıyor?

Bilgisayarınızda **2-3 dakikada** eğitiliyor ve sonra şuna benzer metinler üretiyor:

```text
toyota corolla 2021 hibrit otomatik sedan 122hp 1280bin tl
kia sportage 2021 dizel otomatik suv 136hp 1690bin tl
hyundai itonic 2022 benzin otomatik hatchback 100hp 780bin tl
```

Bu satırları model **ezberlemedi**. Kendisine verilen araç listesindeki *deseni* öğrendi ve o desene uyan yeni satırlar üretti. Üçüncü satırdaki `hyundai itonic` veri setinde hiç geçmiyor — model "hyundai" ve "ioniq" kelimelerini harmanlayıp uydurdu.

İşte yapay zekanın özü budur: **ezber değil, desen öğrenme.**

---

## Neden bu projeyi okumalısınız?

Bugün herkes yapay zekadan bahsediyor ama çoğu kişi içinde ne olduğunu bilmiyor. Çünkü modern kütüphaneler her şeyi gizler:

```python
# PyTorch ile: burada gerçekte ne oluyor?
loss.backward()
optimizer.step()
```

Bu iki satırın arkasında yüzlerce türev hesabı, binlerce matris işlemi var. Bu projede o iki satırı **kendimiz yazdık**. Yani:

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } __Sihir yok__

    ---

    Yapay zekanın "bir sonraki kelimeyi tahmin et" kuralından ibaret olduğunu göreceksiniz.

-   :material-function-variant:{ .lg .middle } __Her satır açık__

    ---

    Türev formüllerinden bellek düzenine kadar her şey görünür ve değiştirilebilir.

-   :material-speedometer:{ .lg .middle } __Gerçekten çalışıyor__

    ---

    Oyuncak bir demo değil; gerçekten öğreniyor, loss düşüyor, anlamlı metin üretiyor.

-   :material-language-csharp:{ .lg .middle } __Sadece .NET__

    ---

    Python, CUDA, sanal ortam derdi yok. `dotnet run` yeterli.

</div>

---

## Modelin künyesi

| Özellik | Değer | Karşılaştırma (GPT-2 Small) |
|---|---|---|
| Parametre sayısı | 350.927 | 124.000.000 |
| Katman sayısı | 3 | 12 |
| Attention kafası | 4 | 12 |
| Vektör boyutu (`d_model`) | 96 | 768 |
| Bağlam penceresi | 64 karakter | 1024 token |
| Eğitim verisi | 10 KB | 40 GB |
| Eğitim süresi | ~2,5 dakika (CPU) | Günlerce (GPU) |
| **Mimari** | **Transformer** | **Transformer** |

Son satır en önemlisi: **mimari birebir aynı.** Fark yalnızca ölçekte.

---

## Nereden başlamalı?

=== "Hiç bilmiyorum"

    Sırayla gidin:

    1. [Bu Proje Nedir?](baslangic/proje-nedir.md)
    2. [Yapay Zeka Nedir?](temeller/yapay-zeka-nedir.md)
    3. [Sinir Ağı Nedir?](temeller/sinir-agi.md)
    4. [Dil Modeli Nedir?](temeller/dil-modeli.md)

=== "Programcıyım, AI bilmiyorum"

    1. [Kurulum](baslangic/kurulum.md) — önce çalıştırın, görün
    2. [Genel Bakış](mimari/genel-bakis.md) — veri akışını izleyin
    3. [Kod Haritası](referans/kod-haritasi.md) — dosyaların görevleri

=== "AI biliyorum, koda bakacağım"

    1. [Autograd](egitim/autograd.md) — hesap grafiği nasıl kuruldu
    2. [Attention](mimari/attention.md) — batch'li dilimleme mantığı
    3. [Performans](ileri/performans.md) — SIMD, float, paralelleştirme

---

## Hızlı başlangıç

```powershell
cd project
dotnet run -c Release -- train
```

İki buçuk dakika sonra elinizde eğitilmiş bir dil modeli olacak. Devamı için [Kurulum](baslangic/kurulum.md) sayfasına geçin.
