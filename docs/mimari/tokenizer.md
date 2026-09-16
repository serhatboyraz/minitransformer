# Tokenizer

**Dosya:** `src/Tokenizer.cs` — 30 satır

Sinir ağları sayılarla çalışır. Tokenizer'ın tek görevi metni sayıya, sayıyı metne çevirmektir.

---

## Sözlük nasıl oluşuyor?

```csharp
public Tokenizer(string corpus)
{
    _idToChar = corpus.Distinct().OrderBy(c => c).ToArray();
    _charToId = _idToChar.Select((c, i) => (c, i)).ToDictionary(t => t.c, t => t.i);
}
```

Üç adım:

1. **`Distinct()`** — metindeki benzersiz karakterleri bul
2. **`OrderBy(c => c)`** — sırala (böylece her çalıştırmada aynı sözlük oluşur)
3. **Sözlük kur** — karakterden id'ye hızlı arama için

!!! warning "Sıralama neden şart?"
    `Distinct()` tek başına metindeki ilk görülme sırasını verir. Metin değişirse sözlük de değişir ve eski checkpoint'ler bozulur. Sıralama bunu deterministik yapar.

---

## Bizim verimizin sözlüğü

`data/input.txt` içinde **47 benzersiz karakter** var:

| Grup | Karakterler | Adet |
|---|---|---|
| Kontrol | `\n` `\r` | 2 |
| Boşluk/noktalama | (boşluk) `.` `:` | 3 |
| Rakamlar | `0`–`9` | 10 |
| İngiliz harfleri | `a`–`z` | 26 |
| Türkçe harfler | `ç` `ğ` `ı` `ö` `ş` `ü` | 6 |
| **Toplam** | | **47** |

!!! note "Büyük harf yok"
    Veri dosyası tamamen küçük harfle yazıldı. Büyük harfler eklenseydi sözlük 73'e çıkacak, model hem "A" hem "a" için ayrı temsil öğrenmek zorunda kalacaktı. Küçük bir modelde bu gereksiz yük olurdu.

---

## Kodlama ve çözme

```csharp
public int[] Encode(string text) =>
    text.Where(_charToId.ContainsKey).Select(c => _charToId[c]).ToArray();

public string Decode(IEnumerable<int> ids) =>
    new([.. ids.Select(i => _idToChar[i])]);
```

### Örnek

```text
Encode("fiat 500")
  f → 22
  i → 25
  a → 17
  t → 36
  ' ' → 2
  5 → 10
  0 → 5
  0 → 5
Sonuç: [22, 25, 17, 36, 2, 10, 5, 5]
```

`Decode` bunun tersini yapar ve `"fiat 500"` döner.

### Bilinmeyen karakterler

`Where(_charToId.ContainsKey)` filtresi sözlükte olmayan karakterleri **sessizce atar**:

```text
Encode("fiat@500") → "fiat500" ile aynı sonuç
```

Gerçek sistemlerde bunun yerine `<UNK>` (unknown) diye özel bir token kullanılır. Burada basitlik için atma tercih edildi.

---

## Sözlüğün saklanması

Model bir dosyaya kaydedilirken sözlük de kaydedilmelidir. Aksi halde yükleme sırasında id'ler farklı karakterlere denk gelir ve çıktı anlamsız olur.

```csharp
// src/Tokenizer.cs
public string Vocabulary => new(_idToChar);
```

Bu özellik [Checkpoint](../kullanim/checkpoint.md) tarafından kullanılır. Geri yüklerken:

```csharp
var tokenizer = new Tokenizer(reader.ReadString());
```

!!! success "Zarif detay"
    Kurucu metot `Distinct().OrderBy()` uyguladığı için, zaten sıralı ve benzersiz bir sözlük stringi verildiğinde **aynı sözlük** çıkar. Ayrı bir kurucu metoda gerek kalmadı.

Metin formatında ise karakterler **kod noktası** olarak yazılır:

```text
vocab 10 13 32 46 48 49 50 51 52 ...
```

Çünkü sözlükte satır sonu karakteri (`\n` = 10) var; düz yazılsaydı dosyanın satır yapısı bozulurdu.

---

## Neden karakter seviyesi?

| Kriter | Karakter | BPE (GPT tarzı) |
|---|---|---|
| Sözlük boyutu | 47 | ~50.000 |
| Çıkış katmanı parametresi | 96×47 = 4.512 | 768×50000 = 38 milyon |
| Dizi uzunluğu | Uzun (64 karakter ≈ 10 kelime) | Kısa (64 token ≈ 48 kelime) |
| Öğrenmesi gereken | Yazım + dilbilgisi | Sadece dilbilgisi |
| Kod karmaşıklığı | 30 satır | ~500 satır |

Bu proje **öğretici** olduğu için karakter seviyesi seçildi. Ama unutmayın: model önce *yazmayı* öğrenmek zorunda kaldığı için ilk 200 adım neredeyse tamamen buna gidiyor.

!!! example "Bunu gözlemleyin"
    Eğitimin ilk 100 adımında üretilen metin şuna benzerdi:
    ```text
    aei  otn oiaet r  nea
    ```
    Sadece harf frekansını öğrenmiş. Kelime yok. Adım 500'de:
    ```text
    benzin otoma sedan 100h
    ```
    Kelimeler oluşmuş. Adım 1500'de format tamamen oturmuş.

---

## Ölçeklerken ne değişir?

Türkçe metinler için BPE'ye geçmek isterseniz dikkat edilecekler:

- Türkçe sondan eklemeli: "ev, evde, evimde, evlerimizden" — kelime seviyesi patlar, BPE ekleri ayrı token yapar
- Türkçe karakterler UTF-8'de 2 bayt tutar; byte seviyesi BPE bunları doğal olarak parçalar
- Sözlük büyürse çıkış katmanı ve embedding tablosu büyür; küçük modellerde bu parametrelerin çoğunu yutar

---

Sıradaki adım: [Embedding](embedding.md)
