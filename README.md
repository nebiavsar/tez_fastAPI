# OCR Sınav Değerlendirme FastAPI Servisi

Bu proje, OCR tabanlı sınav değerlendirme mimarisinde yer alan **FastAPI işleme servisi**dir. Sistemdeki ana görevi, Spring Boot tarafından gönderilen sınav görselini almak, görseli doğrulamak, OCR ve değerlendirme adımlarını çalıştırmak ve sonucu Spring tarafının beklediği JSON formatında geri döndürmektir.

Mevcut sürümde OCR ve NLP/değerlendirme katmanı **placeholder** olarak çalışır. Yani bugün gerçek model entegrasyonu olmadan uçtan uca akış ve API sözleşmesi test edilebilir. Daha sonra gerçek OCR/NLP servisleri eklendiğinde, endpoint sözleşmesini bozmadan yalnızca servis iç mantığı değiştirilebilir.

## 1. Projenin Amacı

Bu servis doğrudan son kullanıcıya hizmet veren ana backend değildir. Ana backend rolü Spring Boot tarafındadır. FastAPI burada daha çok:

- sınav görselini işleyen uzman servis,
- OCR ve değerlendirme orkestrasyon katmanı,
- Spring için sade ve sabit bir JSON sağlayıcısı

olarak konumlanır.

Bu sayede sistemde sorumluluklar ayrılır:

- **Frontend / mobil uygulama**: kullanıcıdan görsel alır ve işlemi başlatır
- **Spring Boot**: kimlik doğrulama, iş kuralları, veritabanı, kullanıcı/sınav yönetimi, sonuç saklama
- **FastAPI**: dosya doğrulama, OCR çalıştırma, cevapları çıkarma, puanlama sonucunu üretme

## 2. Yüksek Seviyeli Mimari

Önerilen akış aşağıdaki gibidir:

```text
Frontend / Mobile
        |
        v
Spring Boot Backend
        |
        v
FastAPI OCR Service
   |            |
   v            v
 OCR  -->  NLP / Evaluation
        |
        v
Spring Boot
        |
        v
Frontend / Mobile
```

Kısa özet:

1. Kullanıcı sınav kağıdının fotoğrafını veya görselini frontend üzerinden yükler.
2. Frontend isteği Spring Boot'a gönderir.
3. Spring Boot gerekli iş kurallarını uygular ve görseli FastAPI'ye iletir.
4. FastAPI dosyayı doğrular.
5. FastAPI OCR katmanını çağırır.
6. OCR çıktısı değerlendirme/NLP katmanına aktarılır.
7. Sonuç Spring'in beklediği JSON'a dönüştürülür.
8. Spring isterse sonucu veritabanına kaydeder ve frontend'e kendi response formatıyla döner.

## 3. FastAPI Servisinin Sorumlulukları

Bu servis şunları yapar:

- `multipart/form-data` ile gelen sınav görselini alır
- dosyanın boş olup olmadığını, tipini ve boyutunu kontrol eder
- OCR servisini çağırır
- OCR sonucunu değerlendirme servisine iletir
- sonucu sabit bir response modeliyle döndürür
- hataları anlamlı JSON formatında döner

Bu servis şunları yapmaz:

- kullanıcı girişi / yetkilendirme
- veritabanı işlemleri
- sınav oluşturma, öğrenci yönetimi, öğretmen paneli
- frontend sayfaları
- Spring tarafındaki domain/business logic

## 4. Mevcut İş Mantığı

Şu anki sürüm geliştirme ve entegrasyon odaklıdır.

### OCR katmanı

`app/services/ocr_service.py`

- gerçek OCR çalıştırmaz
- örnek iki soru ve iki cevap üretir
- yüklenen dosya adı ve boyutundan placeholder çıktı oluşturur

### NLP / değerlendirme katmanı

`app/services/nlp_service.py`

- OCR çıktısını alır
- her soru için örnek `expectedAnswer`, `score` ve `feedback` üretir
- toplam puanı hesaplar

### Orkestrasyon katmanı

`app/services/exam_service.py`

- dosyayı okur
- doğrulamaları yapar
- OCR ve NLP servislerini doğru sırayla çağırır
- hata oluşursa uygun uygulama hatasına çevirir

### API katmanı

`app/api/routes/exam.py`

- dış dünyaya açılan endpointleri tanımlar
- dependency injection ile servisleri bağlar
- response modelini belirler

Kısacası asıl merkez akış şudur:

`UploadFile -> validation -> OCRService.extract() -> NLPService.evaluate() -> JSON response`

## 5. Proje Yapısı

```text
app/
├── api/
│   ├── router.py                 # merkezi router
│   └── routes/
│       └── exam.py               # health ve process-exam endpointleri
├── core/
│   ├── config.py                 # uygulama ayarları
│   └── exceptions.py             # özel exception ve handler'lar
├── schemas/
│   ├── __init__.py               # şema export'ları
│   ├── common.py                 # ortak response modelleri
│   └── exam.py                   # sınav işleme modelleri
├── services/
│   ├── exam_service.py           # orkestrasyon
│   ├── nlp_service.py            # placeholder değerlendirme
│   └── ocr_service.py            # placeholder OCR
└── main.py                       # FastAPI app factory
main.py                           # alternatif entrypoint
requirements.txt
README.md
```

## 6. Teknik Özellikler ve Varsayılanlar

Mevcut ayarlar `app/core/config.py` içinde tanımlıdır:

- uygulama adı: `OCR Exam Processing Service`
- sürüm: `0.1.0`
- maksimum dosya boyutu: `10 MB`
- izin verilen içerik tipleri:
  - `image/jpeg`
  - `image/jpg`
  - `image/png`
  - `image/webp`

## 7. Endpointler

Bu projede route prefix tanımlı değildir; endpointler doğrudan root altında yayınlanır.

### `GET /health`

Servisin ayakta olduğunu kontrol etmek için kullanılır.

Örnek response:

```json
{
  "status": "ok"
}
```

Kullanım amacı:

- Kubernetes / Docker health check
- reverse proxy kontrolü
- Spring tarafında servis ayakta mı diye hızlı doğrulama

### `POST /process-exam`

Sınav görselini işleyen ana endpoint budur.

#### İstek formatı

- Method: `POST`
- Content-Type: `multipart/form-data`
- Form alanı adı: `image`

#### Beklenen dosya tipleri

- `image/jpeg`
- `image/jpg`
- `image/png`
- `image/webp`

#### Başarılı response

```json
{
  "questions": [
    {
      "questionId": "1",
      "questionText": "Question 1",
      "extractedAnswer": "Student answer extracted from the uploaded image.",
      "expectedAnswer": "Expected answer for Question 1",
      "score": 10,
      "feedback": "Placeholder evaluation completed successfully."
    },
    {
      "questionId": "2",
      "questionText": "Question 2",
      "extractedAnswer": "Another placeholder answer extracted for demo purposes.",
      "expectedAnswer": "Expected answer for Question 2",
      "score": 10,
      "feedback": "Placeholder evaluation completed successfully."
    }
  ],
  "score": 20
}
```

#### Response alanlarının anlamı

| Alan | Tip | Açıklama |
| --- | --- | --- |
| `questions` | `array` | Her soru için üretilen değerlendirme çıktısı |
| `questions[].questionId` | `string` | Sorunun kimliği veya numarası |
| `questions[].questionText` | `string` | OCR/NLP akışındaki soru metni |
| `questions[].extractedAnswer` | `string` | Görselden çıkarılan öğrenci cevabı |
| `questions[].expectedAnswer` | `string` | Beklenen cevap |
| `questions[].score` | `int` | Sorunun puanı |
| `questions[].feedback` | `string` | Soru bazlı kısa geri bildirim |
| `score` | `int` | Toplam puan |

## 8. Hata Yönetimi

Servis hata durumlarında okunabilir JSON döner:

```json
{
  "detail": "Hata açıklaması",
  "error": "makine_okunur_hata_kodu"
}
```

### Sık karşılaşılacak hata senaryoları

#### 1. Dosya gönderilmedi

HTTP `400`

```json
{
  "detail": "Image file is required.",
  "error": "missing_file"
}
```

#### 2. Dosya boş

HTTP `400`

```json
{
  "detail": "Uploaded image is empty.",
  "error": "empty_file"
}
```

#### 3. Dosya tipi desteklenmiyor

HTTP `415`

```json
{
  "detail": "Unsupported file type. Allowed types: image/jpeg, image/jpg, image/png, image/webp.",
  "error": "unsupported_file_type"
}
```

#### 4. Dosya boyutu çok büyük

HTTP `413`

```json
{
  "detail": "Uploaded image exceeds the maximum allowed size.",
  "error": "file_too_large"
}
```

#### 5. OCR katmanı hata verdi

HTTP `502`

```json
{
  "detail": "OCR processing failed.",
  "error": "ocr_processing_failed"
}
```

#### 6. NLP / değerlendirme katmanı hata verdi

HTTP `502`

```json
{
  "detail": "NLP processing failed.",
  "error": "nlp_processing_failed"
}
```

#### 7. Beklenmeyen sunucu hatası

HTTP `500`

```json
{
  "detail": "Internal processing error.",
  "error": "internal_error"
}
```

## 9. Spring Boot Tarafı Bu Servisi Nasıl Kullanmalı?

Bu proje mimari olarak **Spring'in arkasında çalışan uzman servis** olarak tasarlanmıştır. Yani ideal kullanımda frontend doğrudan FastAPI'ye değil, Spring Boot'a istek atar.

### Önerilen entegrasyon yaklaşımı

1. Frontend, kullanıcıdan sınav görselini alır.
2. Frontend bu görseli Spring Boot'taki örneğin `/api/exams/process` benzeri bir endpoint'e yollar.
3. Spring Boot kullanıcı doğrulaması, sınav/öğrenci doğrulaması ve iş kurallarını uygular.
4. Spring Boot görseli FastAPI'deki `POST /process-exam` endpoint'ine iletir.
5. FastAPI sonucu JSON olarak döner.
6. Spring Boot isterse:
   - sonucu veritabanına kaydeder,
   - ek domain alanları ekler,
   - sonucu frontend için farklı bir response modeline map eder.

### Spring neden arada olmalı?

Çünkü genellikle şu sorumluluklar Spring'de bulunur:

- authentication / authorization
- kullanıcı, öğretmen, öğrenci, sınav yönetimi
- sonuç geçmişi ve raporlama
- veritabanına kayıt
- dosya saklama politikaları
- audit log ve transaction yönetimi

FastAPI bu yapıda sadece "işleme motoru" olur.

### Spring tarafında request nasıl atılmalı?

FastAPI şu kontratı bekler:

- URL: `POST /process-exam`
- Body: `multipart/form-data`
- Alan adı: `image`

### Spring için örnek DTO karşılığı

```java
public class ExamProcessingResponse {
    private List<QuestionItem> questions;
    private int score;
}

public class QuestionItem {
    private String questionId;
    private String questionText;
    private String extractedAnswer;
    private String expectedAnswer;
    private int score;
    private String feedback;
}
```

### Spring WebClient ile örnek çağrı

```java
MultipartBodyBuilder builder = new MultipartBodyBuilder();
builder.part("image", imageResource)
       .filename("exam.jpg")
       .contentType(MediaType.IMAGE_JPEG);

ExamProcessingResponse response = webClient.post()
    .uri("http://fastapi-service:8000/process-exam")
    .contentType(MediaType.MULTIPART_FORM_DATA)
    .body(BodyInserters.fromMultipartData(builder.build()))
    .retrieve()
    .bodyToMono(ExamProcessingResponse.class)
    .block();
```

### Spring tarafında önerilen servis akışı

```text
Controller
  -> Application Service
     -> FastApiClient
        -> FastAPI /process-exam
     -> ResultMapper / Persistence
  -> Frontend response
```

### Spring tarafında yapılması iyi olur

- FastAPI timeout yönetimi eklemek
- `502`, `500`, `415`, `413` gibi hataları düzgün map etmek
- FastAPI sonucu veritabanına kaydetmek
- tekrar deneme, circuit breaker veya fallback kurgulamak
- log ve correlation id geçirmek

## 10. Frontend Tarafı Bu Yapıda Nasıl Çalışmalı?

Frontend'in en sağlıklı yaklaşımı, FastAPI ile doğrudan konuşmak yerine Spring Boot ile konuşmasıdır.

### Önerilen frontend akışı

1. Kullanıcı dosya seçer.
2. Frontend `FormData` oluşturur.
3. İstek Spring endpoint'ine gönderilir.
4. Spring FastAPI'yi çağırır.
5. Spring nihai sonucu frontend'e döner.
6. Frontend soru bazlı puanları ve toplam puanı gösterir.

### Neden frontend doğrudan FastAPI'ye gitmemeli?

- auth ve token doğrulama çoğunlukla Spring'dedir
- CORS yönetimini sade tutar
- business logic tek yerde toplanır
- FastAPI iç servis olarak kalır
- servis değişse bile frontend kontratı daha stabil olur

### Frontend'de örnek istek mantığı

```javascript
const formData = new FormData();
formData.append("image", file);

const response = await fetch("/api/exams/process", {
  method: "POST",
  body: formData,
});

const data = await response.json();
```

Burada dikkat edilmesi gereken nokta şudur: frontend örnekte **Spring endpoint'ine** istek atar. Spring ise arka planda FastAPI'yi çağırır.

### Frontend'in gösterebileceği örnek ekran alanları

- toplam puan
- soru listesi
- her sorunun çıkarılan cevabı
- beklenen cevap
- öğretmen geri bildirimi / model feedback
- işleme hatası varsa kullanıcı dostu hata mesajı

## 11. Uçtan Uca Akış Senaryosu

Örnek senaryo:

1. Öğrenci ya da öğretmen bir sınav görseli yükler.
2. Frontend dosyayı Spring'e yollar.
3. Spring dosyanın ilgili kullanıcıya/sınava ait olup olmadığını kontrol eder.
4. Spring dosyayı FastAPI'ye aktarır.
5. FastAPI dosya tipini ve boyutunu doğrular.
6. OCR placeholder çıktısı üretilir.
7. NLP placeholder değerlendirmesi yapılır.
8. Toplam puan hesaplanır.
9. Sonuç Spring'e döner.
10. Spring sonucu ister saklar ister doğrudan istemciye döner.

## 12. Gerçek OCR / NLP Entegrasyonu Geldiğinde Neler Değişecek?

En önemli avantaj, API katmanını bozmadan iç servisi değiştirebilmenizdir.

Değiştirilecek yerler:

- `app/services/ocr_service.py`
- `app/services/nlp_service.py`

Muhtemel geliştirmeler:

- Tesseract, PaddleOCR, EasyOCR veya bulut OCR servisleri
- LLM tabanlı cevap değerlendirme
- soru anahtarına göre dinamik puanlama
- rubric bazlı değerlendirme
- çok sayfalı PDF/görsel desteği
- el yazısı tanıma

Bu geliştirmeler yapılırken mümkün olduğunca şu dosyaları sabit tutmak avantaj sağlar:

- `app/api/routes/exam.py`
- `app/schemas/exam.py`

Böylece Spring entegrasyonu kırılmaz.

## 13. Lokal Kurulum

### Gereksinimler

- Python 3.11+ önerilir
- `pip`

### Paket kurulumu

```bash
pip install -r requirements.txt
```

### Uygulamayı çalıştırma

```bash
uvicorn app.main:app --reload
```

Alternatif entrypoint:

```bash
uvicorn main:app --reload
```

Uygulama varsayılan olarak şu adreste çalışır:

- `http://127.0.0.1:8000`

## 14. API Dokümantasyonu

FastAPI varsayılan dokümantasyon ekranları kullanılabilir:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

Spring ekibi veya frontend ekibi, entegrasyon sırasında bu sayfalar üzerinden endpoint sözleşmesini hızlıca test edebilir.

## 15. Örnek İstekler

### `curl` ile

```bash
curl -X POST "http://127.0.0.1:8000/process-exam" \
  -H "accept: application/json" \
  -F "image=@sample_exam.jpg;type=image/jpeg"
```

### PowerShell ile

```powershell
$form = @{
  image = Get-Item ".\sample_exam.jpg"
}

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/process-exam" `
  -Method Post `
  -Form $form
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

## 16. Geliştirme Notları

- Logging yapılandırması `app/main.py` içinde yapılır.
- Özel exception handler'lar `app/core/exceptions.py` içinde toplanmıştır.
- Response modelleri Pydantic ile tanımlanmıştır.
- Route katmanı ile servis katmanı birbirinden ayrıldığı için kod kolay genişletilebilir.

## 17. Kısa Sonuç

Bu repo, sınav görsellerini işlemek için tasarlanmış, Spring Boot ile entegre çalışan bir FastAPI servisidir. Şu anda placeholder mantıkla çalışır; ancak endpoint yapısı, hata sözleşmesi ve response modeli gerçek üretim entegrasyonuna uygun olacak şekilde kurgulanmıştır.

En önemli fikir şudur:

- **Frontend -> Spring Boot -> FastAPI** ana akış olmalı
- FastAPI işleme motoru gibi davranmalı
- Spring ana sistem olarak kalmalı
- Gerçek OCR/NLP eklendiğinde dış API kontratı bozulmamalı
