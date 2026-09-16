# GPU

"Bu projeyi GPU'da çalıştırabilir miyim?" Kısa cevap: **teknik olarak evet, pratikte gereksiz.** Sebeplerini inceleyelim.

---

## GPU neden hızlıdır?

| | CPU | GPU |
|---|---|---|
| Çekirdek sayısı | 8-64 | 2.000-20.000 |
| Çekirdek başına hız | Çok yüksek | Düşük |
| İyi olduğu iş | Karmaşık, dallanmalı mantık | Basit işlemin milyonlarca tekrarı |
| Bellek bant genişliği | ~50 GB/s | ~1000 GB/s |

Matris çarpımı GPU için mükemmel bir iştir: aynı işlem (çarp-topla) binlerce veri üzerinde bağımsız tekrar eder.

---

## Peki neden bu projede işe yaramaz?

### Sebep 1: Matrisler çok küçük

Bizim en büyük çarpımımız: `[512 × 96] × [96 × 384]`

Bir GPU bunu **mikrosaniyeler** içinde yapar. Ama:

```text
Kernel başlatma:     ~5-10 mikrosaniye
Veri transferi:      ~10-50 mikrosaniye
Gerçek hesaplama:    ~2 mikrosaniye
```

**Overhead, işten büyük.** Adım başına ~200 matris çarpımı olduğunu düşünün — sadece kernel başlatma maliyeti 1-2 milisaniye eder. GPU'da çalıştırmak muhtemelen **daha yavaş** olurdu.

### Sebep 2: Çok sayıda küçük işlem

Adım başına ~500 ayrı tensör işlemi var. Her biri ayrı bir GPU kernel çağrısı demektir. Gerçek kütüphaneler bunu **kernel fusion** ile çözer (LayerNorm + Linear + GELU tek kernel), ama bu ciddi mühendislik gerektirir.

### Sebep 3: Eğitim zaten 2 dakika

GPU'ya taşımak günlerce iş demek. Kazanç? En iyi ihtimalle 2 dakika yerine 1 dakika. Değmez.

!!! quote "Mühendislik kararı"
    Optimizasyon, darboğazın olduğu yere yapılır. Bu projenin darboğazı hesaplama gücü değil, **problem boyutu**. Model büyüdüğünde denklem değişir.

---

## Ne zaman GPU mantıklı olur?

GPU'nun anlamlı hale gelmesi için şu eşikler aşılmalı:

| Parametre | Şu an | GPU eşiği |
|---|---|---|
| `EmbedSize` | 96 | 512+ |
| `BatchSize` | 8 | 64+ |
| `BlockSize` | 64 | 512+ |
| Parametre sayısı | 350 bin | 10 milyon+ |
| Eğitim süresi | 2 dk | Saatler |

Bu değerlerde matrisler `[32768 × 512] × [512 × 2048]` gibi olur — GPU'nun parladığı boyut.

---

## .NET'te GPU seçenekleri

### TorchSharp

LibTorch (PyTorch'un C++ çekirdeği) için resmi .NET bağlayıcısı.

```csharp
using TorchSharp;
var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
var model = new GPT(config).to(device);
```

| Artı | Eksi |
|---|---|
| Hazır autograd, optimizer, katmanlar | NVIDIA GPU gerekir (CUDA) |
| Üretim kalitesinde performans | ~2 GB paket indirmesi |
| PyTorch ekosistemiyle uyumlu | Öğrenme amacını yok eder |

!!! tip "CPU modunda bile hızlı"
    NVIDIA kartınız olmasa bile TorchSharp'ın CPU modu bu koddan **çok** hızlıdır. Intel MKL kullanır, gerçek batching yapar, hafızayı verimli yönetir.

### ILGPU

C# kodunu doğrudan GPU kerneline derler. CUDA, OpenCL ve CPU backend'leri var.

```csharp
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(MatMulKernel);
```

| Artı | Eksi |
|---|---|
| Intel/AMD GPU'larda da çalışır (OpenCL) | Kernelleri kendiniz yazarsınız |
| Autograd'ınızı korursunuz | Bellek yönetimi elle |
| Saf C# | Optimize kernel yazmak zor |

### ComputeSharp

DirectX 12 compute shader'ları C# ile yazmanızı sağlar.

| Artı | Eksi |
|---|---|
| Her GPU'da çalışır (Intel dahil) | Sadece Windows |
| Modern, temiz API | Kernel yazma zorunluluğu |

### ONNX Runtime + DirectML

| Artı | Eksi |
|---|---|
| Intel/AMD GPU desteği | **Sadece çıkarım, eğitim yok** |
| Çok hızlı inference | Modeli ONNX'e çevirmek gerekir |

---

## Donanımınıza göre karar

=== "NVIDIA GPU'nuz var"

    **TorchSharp** en mantıklısı. CUDA desteği hazır gelir, tüm ekosistem kullanılabilir.

    ```powershell
    dotnet add package TorchSharp-cuda-windows
    ```

    Ama unutmayın: bu projeyi TorchSharp'a çevirmek, öğrenme amacını ortadan kaldırır. Autograd'ı kendiniz yazmanın değeri, onu kütüphaneye devretmenizde kaybolur.

=== "Intel/AMD entegre GPU'nuz var"

    **ComputeSharp** veya **ILGPU** çalışır — ama bu model boyutunda kazanç görmezsiniz.

    Daha iyi yatırım: [Performans](performans.md) sayfasındaki CPU optimizasyonlarını tamamlamak.

=== "GPU'nuz yok"

    Hiç sorun değil. Bu proje zaten CPU için tasarlandı. Daha büyük modeller için Google Colab (ücretsiz T4 GPU) veya bulut sağlayıcıları kullanabilirsiniz.

---

## GPU'ya hazırlık: zaten yapıldı

İlginç şekilde, [Performans](performans.md) sayfasındaki **batch birleştirme** optimizasyonu aynı zamanda GPU'ya geçişin ön koşuluydu.

Eskiden 8 dizi tek tek işleniyordu — GPU için felaket. Şimdi `[512 × 96]` tek matris var — GPU'nun sevdiği yapı.

```text
Önce:  8 × ([64 × 96] × [96 × 96])    ← GPU'ya uygun değil
Sonra: 1 × ([512 × 96] × [96 × 96])   ← GPU'ya uygun
```

Yani mimari hazır. Sadece `Ops.MatMul` içindeki döngüyü bir GPU kerneliyle değiştirmek yeterli olurdu.

---

## Gerçekçi bir yol haritası

Bu projeyi GPU'ya taşımak isterseniz sıralama şöyle olmalı:

1. **Model boyutunu artırın** — `EmbedSize = 512`, `Layers = 8`, `BatchSize = 64`. Eğitim CPU'da saatler sürsün.
2. **Profil çıkarın** — sürenin gerçekten matris çarpımında geçtiğini doğrulayın.
3. **Tek işlemi taşıyın** — sadece `MatMul`'u ILGPU/ComputeSharp ile yazın, gerisini CPU'da bırakın.
4. **Ölçün** — transfer maliyeti kazancı yiyor mu?
5. **Gerekirse tüm grafiği GPU'da tutun** — veri gidip gelmesin.

Adım 4'te çoğu kişi durur, çünkü veri transferi beklenenden pahalıdır.

!!! warning "Amdahl yasası"
    Sürenin %90'ı matris çarpımındaysa ve onu **sonsuz** hızlandırsanız bile toplam kazanç 10 kattır. Kalan %10 duvar olur.

---

## Özet

| Soru | Cevap |
|---|---|
| Bu proje GPU'da çalışır mı? | Teknik olarak evet, kod yazmak gerekir |
| Hızlanır mı? | Bu boyutta muhtemelen hayır |
| Ne zaman mantıklı? | Model 10M+ parametreye çıkarsa |
| En kolay yol? | TorchSharp (ama öğrenme amacını bitirir) |
| Şimdi ne yapmalı? | CPU optimizasyonlarını tamamlayın |

---

Sıradaki adım: [Deneyler](deneyler.md)
