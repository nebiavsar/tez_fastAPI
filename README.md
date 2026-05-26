# tez_fastAPI — AI Tabanlı Otomatik Sınav Değerlendirme (ASAG)

> Lisans tez projesi — Türkçe el yazısı sınav kâğıtlarını **otomatik okuyup**, öğretmenin
> cevap kâğıdı ile karşılaştırarak **soru bazında semantik puan üreten** AI sistemi.

Bu repo, sistemin **Python/FastAPI** motorunu içerir: OCR (Qwen2.5-VL-7B Vision-Language
Model) + NLP semantik karşılaştırma (fine-tuned SBERT) + tip bazlı modüler skorlama.
Yönetim, kimlik doğrulama ve veritabanı katmanı [tez_springBootAPI](https://github.com/nebiavsar)
deposunda; Android istemcisi ayrı depoda.

---

## İçindekiler

1. [Sorun ve çözüm](#1-sorun-ve-çözüm)
2. [Sistem mimarisi](#2-sistem-mimarisi)
3. [Uçtan uca işlem akışı](#3-uçtan-uca-işlem-akışı)
4. [OCR katmanı (FastAPI)](#4-ocr-katmanı-fastapi)
5. [NLP katmanı — tip bazlı skorlama](#5-nlp-katmanı--tip-bazlı-skorlama)
6. [Standart sınav kâğıdı formatı](#6-standart-sınav-kâğıdı-formatı)
7. [API sözleşmesi](#7-api-sözleşmesi)
8. [Kullanılan teknolojiler](#8-kullanılan-teknolojiler)
9. [Proje yapısı](#9-proje-yapısı)
10. [Lokal geliştirme](#10-lokal-geliştirme)
11. [Production / demo deployment (Colab + ngrok)](#11-production--demo-deployment-colab--ngrok)
12. [Spring Boot entegrasyonu](#12-spring-boot-entegrasyonu)
13. [Test sonuçları](#13-test-sonuçları)
14. [Sınırlamalar ve gelecek çalışmalar](#14-sınırlamalar-ve-gelecek-çalışmalar)

---

## 1. Sorun ve çözüm

### Sorun

Öğretmenler her hafta onlarca sınav kâğıdı okuyup puanlıyor. Açık uçlu sorularda:
- Manuel okuma zaman alıcı (her sınav ~3-5 dk)
- Öğretmenler arası **subjektivite** (aynı cevap farklı puan alabiliyor)
- El yazısı çözmek yorucu

### Çözümümüz

Üç katmanlı bir AI pipeline:

1. **OCR** — Görüntüden el yazısı + basılı metin çıkar (Qwen2.5-VL-7B VLM)
2. **Tip tespiti** — Her soruyu *çoktan seçmeli*, *boşluk doldurma* veya *açık uçlu* olarak sınıflandır
3. **Tip bazlı skorlama** — Açık uçlular için **SBERT semantik benzerlik**, diğer tipler için **exact / pattern match**

**Tezin merkezi katkısı:** SBERT semantik karşılaştırma sayesinde, klasik OCR'ın
"%100 doğru karakter okumadığı" durumlarda bile **anlamsal benzerlik** yakalanabilir.
Örneğin OCR "Pb(NO₃)₂" yerine "Pb(NO₃)2" yazsa SBERT bunu **doğru cevap** olarak
yakalar — klasik string matching ile sıfır puan alacak vakalarda **kısmi/tam puan** üretir.

---

## 2. Sistem mimarisi

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              KULLANICI KATMANI                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  Android (Kotlin)                                                │      │
│   │  - Kamera ile sınav kâğıdı fotoğrafı çek                         │      │
│   │  - Sınıf/öğrenci/sınav yönetimi UI                               │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                  │                                          │
│                                  │ HTTP (JWT auth)                          │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORKESTRASYON KATMANI                               │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  Spring Boot 3 (Java)        tez_springBootAPI                   │      │
│   │  - Kullanıcı/sınıf/öğrenci/sınav CRUD (PostgreSQL/MySQL)         │      │
│   │  - JWT auth, dosya yönetimi, image servisi                       │      │
│   │  - WebClient ile FastAPI'ye proxy (multipart)                    │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                  │                                          │
│                                  │ POST /process-exam                       │
│                                  │   paperImage + answerKeyImage            │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AI KATMANI (BU REPO)                             │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  FastAPI (Python)            tez_fastAPI                         │      │
│   │  ┌──────────────────────────────────────────────────────────┐    │      │
│   │  │  ExamService  (orkestrasyon + cache)                     │    │      │
│   │  │  ┌─────────────────┐         ┌─────────────────────┐    │    │      │
│   │  │  │ OCRService      │ ───→    │ NLPService          │    │    │      │
│   │  │  │ Qwen2.5-VL-7B   │         │ Strategy Dispatcher │    │    │      │
│   │  │  │ (4-bit quant)   │         │ ┌──────┬──────┬───┐ │    │    │      │
│   │  │  └─────────────────┘         │ │ MC   │ FB   │OE │ │    │    │      │
│   │  │                              │ └──────┴──────┴───┘ │    │    │      │
│   │  │                              │     ↓        ↓  ↓   │    │    │      │
│   │  │                              │  Exact   Token SBERT│    │    │      │
│   │  │                              └─────────────────────┘    │    │      │
│   │  └──────────────────────────────────────────────────────────┘    │      │
│   └──────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Backend ayrımının gerekçesi

| Katman | Sorumluluk | Neden ayrı? |
|---|---|---|
| **Spring Boot** | CRUD, auth, persist, dosya yönetimi | Java'nın olgun enterprise yığını (Spring Security, JPA, validation) |
| **FastAPI** | OCR + NLP inference | Python'un ML ekosistemi (PyTorch, transformers, sentence-transformers) |

Java tarafında PyTorch ekosistemini çoğaltmak çok zahmetli; Python'da Spring kalitesinde
auth/persist altyapısı kurmak da öyle. Mikroservis ayrımı her iki dünyanın güçlü
yanlarını birleştirir.

---

## 3. Uçtan uca işlem akışı

### Tek bir sınav değerlendirme isteği

```
1. Öğretmen Android app'te:
   "Yeni sınav ekle" → öğrenci kâğıdı fotoğrafı çek

2. Android → Spring Boot:
   PUT /api/classes/{groupId}
     multipart: postExamDTO (JSON) + examPhotos[] (bytes)
   JWT auth header

3. Spring Boot:
   - Validate (sınıf var mı, öğretmenin mi)
   - Sınıfa ait aktif "answer key image" varsa al
   - Disk'e kaydet (uploads/exams/...)
   - FastAPI'ye paperImage + answerKeyImage gönder

4. FastAPI ExamService.process_exam():
   a. answer_key_cache.get(SHA256(answerKeyImage)) → cache HIT mi?
      └─ MISS: OCRService.extract(answerKeyImage)
              → Qwen VLM çağrısı (~60-90 sn)
              → Markdown çıkış: "*çoktan seçmeli soru / 1) ... (10p)"
              → parse_s3_markdown() → list[AnswerKeyEntry]
              → cache'e yaz
      └─ HIT: cache'ten 1ms
   b. OCRService.extract(paperImage) → öğrenci cevap listesi
   c. NLPService.evaluate(student_ocr, answer_key):
      Her soru için tip dispatch:
      - MC  → MultipleChoiceScorer (letter exact)
      - FB  → FillBlankScorer (exact + Türkçe normalize + token)
      - OE  → OpenEndedScorer (SBERT semantic cosine similarity)
   d. ExamProcessingResponse JSON döndür

5. Spring Boot:
   - ExamResult entity'sine yaz (extracted_score, per-question detail)
   - ExamSubmission entity'sine yaz (kâğıt görseli referansı)
   - Android'e JSON dön

6. Android: Sonuç sayfasında skor + soru bazlı detay göster
```

### Performans

| Faz | Süre (T4 GPU) |
|---|---|
| Spring Boot multipart receive | <100 ms |
| FastAPI answer key OCR (cache MISS) | 60-90 sn |
| FastAPI answer key OCR (cache HIT) | <1 ms |
| FastAPI student paper OCR | 60-90 sn |
| FastAPI NLP scoring (5 soru) | 1-3 sn |
| Spring Boot persist + response | <500 ms |
| **TOPLAM (cache MISS, ilk öğrenci)** | **~3-4 dakika** |
| **TOPLAM (cache HIT, sonraki öğrenciler)** | **~1.5 dakika** |

Cache stratejisi sayesinde aynı sınıfın 30 öğrencisini puanlamak 30 × 1.5 dk ≈ **45 dakika** sürer.

---

## 4. OCR katmanı (FastAPI)

### Model seçimi: Qwen2.5-VL-7B-Instruct (4-bit quantize)

Tezde önce klasik OCR (EasyOCR, PaddleOCR) denendi:
- ❌ Türkçe el yazısında düşük doğruluk
- ❌ Layout heuristic'leri (Y-axis clustering) kırılgan
- ❌ Sub-question, tablo, çoktan seçmeli karışık layout'larda çöküyor

VLM (Vision-Language Model) çözümü:
- ✅ Layout + OCR + structured extraction **tek geçişte**
- ✅ Türkçe el yazısı + sembollerde (kimya formülleri vb.) belirgin iyileşme
- ✅ Markdown formatında yapılandırılmış çıkış üretir

**Quantization:** 4-bit `nf4` (bitsandbytes) — VRAM kullanımı ~6 GB, Colab T4 (16 GB) ve
RTX 3060+ (8 GB+) ile uyumlu. Kalite kaybı minimal.

**Deterministik mod:** `do_sample=False` — aynı görsel her seferinde aynı çıkışı verir
(tez "reproducibility" iddiası).

### OCR akışı

```python
# app/services/ocr_service.py
class OCRService:
    def extract(self, file_bytes: bytes, filename: str) -> OCRExtractionResult:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        raw_text = self._loader.generate(image, OCR_PROMPT, max_new_tokens=768)
        detected = parse_s3_markdown(raw_text)
        return OCRExtractionResult(
            raw_text=raw_text,
            lines=raw_text.split("\n"),
            detected_answers=detected,
        )
```

### Prompt mühendisliği

Tezde **deneme-yanılma** ile evrildi:
- v1: ham OCR → degeneration (sonsuz tekrar) sorunu
- v2: anti-degeneration parametreleri (`repetition_penalty`, `no_repeat_ngram_size`)
- v3: tip etiketi `[open_ended]` yazdırma denemesi → VLM uyumsuzdu
- v4 (mevcut): section başlığı + puan değerlerini **AYNEN koru** talimatı

Mevcut prompt (özet):
```
KURALLAR:
1. Yıldızlı (*) section başlıklarını AYNEN koru — '*çoktan seçmeli soru',
   '*boşluk doldurma' gibi. Bu başlıklar skorlama tipini belirler.
2. Soruların yanındaki puan değerlerini (10p), (5p) AYNEN yaz.
3. Her soru için TEK BİR blok yaz.
4. Çoktan seçmeli sorularda öğrencinin işaretlediği şıkkı tek harf olarak yaz.
5. Sadece kâğıttaki cevapları yaz, kendi yorumlarını EKLEME.
```

### Markdown parser — `app/ml/ocr_parser.py`

VLM çıkışı (örnek):
```
*çoktan seçmeli soru
**1)** Türkiye'nin coğrafi konumu... (10p)
A) ... B) ... C) ... D) ...

*boşluk doldurma
**1)** Dünya'nın en büyük... ülke? (5p)
Cevap: Arjantin

**1)** Fotosentez nedir? (10p)
Cevap: ...
```

Parser bunu şuna dönüştürür:
```python
[
    OCRDetectedAnswer(
        question_number="mc1",          # mc/fb/oe prefix + numara
        extracted_answer="...",
        question_type=QuestionType.MULTIPLE_CHOICE,
        max_score=10,                   # (10p) parse edildi
    ),
    OCRDetectedAnswer(question_number="fb1", question_type=FILL_BLANK, max_score=5, ...),
    OCRDetectedAnswer(question_number="oe1", question_type=OPEN_ENDED, max_score=10, ...),
]
```

**Compound ID (`mc1`, `fb1`, `oe1`)** sayesinde **tekrarlı numaralar** çakışmaz:
çoktan seçmeli "1" ve boşluk doldurma "1" iki ayrı kayıt.

**Section başlığı semantiği — one-shot:** Her `*section` başlığı sadece **kendinden sonraki bir
soruyu** etiketler. Sonraki sorular için yeni başlık gerekir veya default OPEN_ENDED'a düşer.

### Halüsinasyon ve gürültü filtreleri

| Filtre | Ne yapar |
|---|---|
| `_filter_hallucinations` | `Yanlış cevap:`, `Not:`, `Çözüm:` gibi model yorumlarını ele |
| `_is_empty_answer` | `(boş)` markörünü tanı, atla |
| `_truncate_at_leak` | Bir block içine başka soru numarası sızdıysa kırp |
| OCR typo varyantları | `seçmeli` ↔ `seçimli`, `boşluk` ↔ `boşluktan`, `uçlu` ↔ `uçlulu` |

---

## 5. NLP katmanı — tip bazlı skorlama

### Strategy pattern (`app/scoring/`)

```
QuestionType.MULTIPLE_CHOICE → MultipleChoiceScorer  → letter exact match
QuestionType.FILL_BLANK      → FillBlankScorer       → exact + TR normalize + token intersect
QuestionType.OPEN_ENDED      → OpenEndedScorer       → SBERT cosine similarity
```

Dispatcher (`app/scoring/dispatcher.py`):
```python
def score_answer(*, question_type, expected, student, max_score, sbert_loader):
    if question_type == QuestionType.MULTIPLE_CHOICE:
        scorer = MultipleChoiceScorer()
    elif question_type == QuestionType.FILL_BLANK:
        scorer = FillBlankScorer()
    else:  # OPEN_ENDED + UNKNOWN
        scorer = OpenEndedScorer(sbert_loader=sbert_loader)

    return scorer.score(expected=expected, student=student, max_score=max_score)
```

### `OpenEndedScorer` — SBERT semantic

Fine-tuned SBERT modeli: `model_ai_noted_with_negatives_positives_v2/`
(Türkçe domain-specific olarak `MultipleNegativesRankingLoss` ile eğitildi.)

```python
similarity = sbert_loader.similarity(expected, student)  # 0..1 cosine

if similarity >= 0.85:                  → tam puan,      "Doğru."
elif similarity >= 0.65:                → %60-80 puan,   "Kısmen doğru."
elif similarity >= 0.40:                → %20-40 puan,   "Yakın ama yetersiz."
else:                                   → 0 puan,        "Yanlış."
```

### `FillBlankScorer` — exact + TR normalize

```python
# Türkçe normalize: ı→i, ş→s, ç→c, ğ→g, ö→o, ü→u, lowercase, punctuation strip
e_norm = _normalize(expected)   # "Fiziksel" → "fiziksel"
s_norm = _normalize(student)    # "fızıksel" → "fiziksel" (OCR Türkçe karışıklığı)

if e_norm == s_norm:                                   → tam puan
elif (e_tokens & s_tokens):                            → %70 puan ("token kesişimi")
else:                                                   → 0 puan
```

Eşleştirme cevapları (a→4, b→2) da bu scorer'a düşer — token kesişimi çiftleri sayar.

### `MultipleChoiceScorer` — letter exact

```python
expected_letter = _extract_letter(expected)  # "C" → "C", "Cevap: C" → "C"
student_letter = _extract_letter(student)

if expected_letter == student_letter:        → tam puan
else:                                         → 0 puan
```

---

## 6. Standart sınav kâğıdı formatı

Tezde tanımlanan format spesifikasyonu — öğretmenler sınav kâğıdı hazırlarken
bu kurallara uymalı.

### Section başlıkları

| Başlık | Atadığı tip | Skorlama |
|---|---|---|
| `*çoktan seçmeli soru`, `*çoktan seçmeli`, `*multiple choice` | MULTIPLE_CHOICE | Letter exact |
| `*boşluk doldurma`, `*eşleştirme`, `*fill in the blank`, `*matching` | FILL_BLANK | Exact + normalize |
| `*açık uçlu`, `*klasik`, `*kısa cevaplı`, `*uzun cevaplı` (veya başlık YOK) | OPEN_ENDED | SBERT semantic |

OCR yazım hataları toleranslı: `seçimli`, `boşluktan`, `uçlulu` da kabul edilir.

### Puan değeri

Format: `(Np)` — örnek: `(10p)`, `(5p)`, `(10 puan)`, `(15 pt)`.
Soru başlığından sonra veya soru gövdesinde yer alabilir.

### Numaralandırma

Her section kendi içinde `1)`, `2)`... yazabilir. Parser otomatik olarak
section prefix ekler (`mc1`, `mc2`, `fb1`, `oe1`...).

### Örnek kâğıt (sadeleştirilmiş)

```
Okul Adı
2.dönem 1.yazılı

Ad: ___ Soyad: ___ Sınıf: ___

*çoktan seçmeli soru
1) Türkiye'nin coğrafi konumu? (10p)
A) Devşirme Sistemi
B) İskân Politikası
C) Tımar Sistemi
D) İltizam Sistemi

*çoktan seçmeli soru
2) ... (10p)
A) ... B) ... C) ... D) ...

*boşluk doldurma
1) Dünya'nın en büyük yüzölçümüne sahip ülke ........'dir. (5p)

*açık uçlu
1-) Fotosentez nedir ve önemi nedir? (10p)
Cevap: _________

*açık uçlu
2-) ... (10p)
Cevap: _________
```

---

## 7. API sözleşmesi

### `POST /process-exam`

**Content-Type:** `multipart/form-data`

| Field | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| `paperImage` | UploadFile | ✅ | Öğrencinin doldurduğu sınav kâğıdı (JPEG/PNG/WEBP, max 10 MB) |
| `answerKeyImage` | UploadFile | ⚠️ | Öğretmenin doğru cevap kâğıdı. Verilmezse NLP skor hesaplanmaz, sadece OCR çıkışı döndürülür. |

**Response — 200:**
```json
{
  "score": 35,
  "questions": [
    {
      "questionId": "mc1",
      "questionText": "",
      "extractedAnswer": "B",
      "expectedAnswer": "B",
      "score": 10,
      "feedback": "[multiple_choice] Doğru (B)."
    },
    {
      "questionId": "fb1",
      "extractedAnswer": "Arjantin",
      "expectedAnswer": "Arjantin",
      "score": 5,
      "feedback": "[fill_blank] Doğru."
    },
    {
      "questionId": "oe1",
      "extractedAnswer": "Fotosentez bitkilerin...",
      "expectedAnswer": "Fotosentez bitkilerin ışık enerjisini...",
      "score": 8,
      "feedback": "[open_ended] Kısmen doğru, eksik kavramlar var. (benzerlik: 0.78)"
    }
  ]
}
```

**Error responses:**

| Status | Sebep |
|---|---|
| 400 | Dosya eksik / boş / okuma hatası |
| 413 | Dosya boyutu 10 MB'ı aştı |
| 415 | Desteklenmeyen MIME (sadece image/jpeg, image/png, image/webp) |
| 502 | OCR veya NLP iç hatası (Qwen yüklenmedi, vb.) |
| 500 | Beklenmeyen |

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /docs`

Otomatik üretilen Swagger UI.

---

## 8. Kullanılan teknolojiler

### AI / ML
- **PyTorch 2.7+** (CUDA 11.8) — tensor operasyonları
- **transformers 5.0+** — HuggingFace model API'si
- **bitsandbytes** — 4-bit `nf4` quantization
- **Qwen2.5-VL-7B-Instruct** — Vision-Language Model (Alibaba)
- **sentence-transformers** — SBERT wrapper
- **Fine-tuned SBERT** — Türkçe domain (`MultipleNegativesRankingLoss`)
- **accelerate** — büyük model yükleme

### Backend
- **FastAPI 0.136+** — async REST API
- **uvicorn** — ASGI server
- **Pydantic 2** — schema validation
- **Pillow 11** — image I/O

### Deployment
- **Google Colab T4/L4 GPU** — runtime
- **pyngrok** — public URL tunneling
- **Google Drive** — model cache (449 MB SBERT + ~6 GB Qwen)
- **Git + GitHub** — versiyon kontrolü

### Test
- **pytest** — 44 unit test (parser + scoring + NLP)
- Mock SBERT (CPU-free testler)

---

## 9. Proje yapısı

```
tez_fastAPI/
├── app/
│   ├── main.py                          # FastAPI app factory + lifespan event
│   ├── api/
│   │   ├── router.py
│   │   └── routes/
│   │       └── exam.py                  # POST /process-exam, GET /health
│   ├── core/
│   │   ├── config.py                    # Settings (max_upload_size, allowed_mime)
│   │   └── exceptions.py                # Custom exceptions + handlers
│   ├── ml/                              # ← AI yükleyiciler + parser
│   │   ├── qwen_loader.py               # Qwen2.5-VL singleton, 4-bit load
│   │   ├── sbert_loader.py              # SBERT singleton
│   │   ├── ocr_parser.py                # Markdown → OCRDetectedAnswer
│   │   └── answer_key_cache.py          # SHA256 hash bazlı cache
│   ├── scoring/                         # ← Strategy pattern
│   │   ├── base.py                      # Scorer ABC + ScoreResult
│   │   ├── open_ended.py                # SBERT semantic
│   │   ├── fill_blank.py                # Exact + TR normalize
│   │   ├── multiple_choice.py           # Letter exact
│   │   └── dispatcher.py                # Tip → scorer route
│   ├── schemas/
│   │   ├── common.py                    # ErrorResponse, HealthResponse
│   │   └── exam.py                      # OCRDetectedAnswer, ExamProcessingResponse, ...
│   └── services/
│       ├── exam_service.py              # Orkestrasyon + dosya validation + cache
│       ├── ocr_service.py               # Qwen + prompt + parser glue
│       └── nlp_service.py               # answer_key dict + dispatch
├── model_ai_noted_with_negatives_positives_v2/  # Fine-tuned SBERT (gitignored)
├── notebooks/
│   ├── colab_fastapi_runner.ipynb       # Colab uvicorn + ngrok
│   └── colab_qwen_ocr_spike.ipynb       # Eski spike notebook
├── scripts/
│   ├── setup_ocr.ps1                    # Lokal Windows kurulum
│   └── spike_qwen_vl.py                 # OCR spike CLI script
├── tests/
│   ├── test_ocr_parser.py               # Parser + section detection
│   ├── test_nlp_service.py              # NLPService dispatch
│   └── test_scoring.py                  # 4 scorer + dispatcher
├── requirements.txt
├── main.py                              # uvicorn entrypoint
└── README.md                            # bu dosya
```

---

## 10. Lokal geliştirme

### Sistem gereksinimleri

| Bileşen | Önerilen |
|---|---|
| Python | 3.13 |
| GPU | NVIDIA 8+ GB VRAM (4 GB'ta sıkışık) |
| CUDA | 11.8 |
| Disk | 10 GB boş (model cache) |
| RAM | 16 GB+ |

### Kurulum

```powershell
# Venv
python -m venv .venv
.venv\Scripts\activate

# PyTorch CUDA build (CPU-only çalışmaz)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Diğer
pip install -r requirements.txt

# SBERT modelini temin et (449 MB, git'te yok)
# Google Drive'dan kopyala veya HuggingFace push'tan çek
# Hedef yol: model_ai_noted_with_negatives_positives_v2/
```

### Çalıştırma

```powershell
uvicorn app.main:app --reload
```

Hızlı kontrol:
```powershell
curl http://localhost:8000/health
# {"status":"ok"}
```

Swagger UI: http://localhost:8000/docs

### Test

```powershell
pytest tests/ -v
```

44 test geçmeli. GPU gerek yok (SBERT mock'lanır).

---

## 11. Production / demo deployment (Colab + ngrok)

Sistem **demo amaçlı** Colab GPU + ngrok tunnel ile dışarıya açılır.
Self-hosted GPU server alternatifi tezdeki "future work" kapsamında.

### Setup (tek seferlik)

1. **Google Drive'a iki şey yükle:**
   - `tez_kaynaklari/model_ai_noted_with_negatives_positives_v2/` (SBERT, 449 MB)
   - (İlk Colab çalıştırmasından sonra) `tez_kaynaklari/qwen_cache/` (~6 GB)

2. **ngrok hesap aç:** https://dashboard.ngrok.com/get-started/your-authtoken
   Bedava tier yeterli (session başında URL değişir).

### Her demo başında

1. Colab'da notebook'u aç:
   https://colab.research.google.com/github/nebiavsar/tez_fastAPI/blob/main/notebooks/colab_fastapi_runner.ipynb
2. **Çalışma zamanı → T4 GPU**
3. **Çalışma zamanı → Tümünü çalıştır**
4. Drive izni → onayla
5. ngrok token → yapıştır
6. Bölüm 6 çıktısından `FASTAPI_BASE_URL=...` satırını kopyala

Süre: ilk çalıştırma ~10 dk (Qwen + SBERT yükleme), sonraki çalıştırmalar Drive
cache sayesinde ~2 dk.

### Cleanup

Notebook Bölüm 10 hücresi tunnel + uvicorn'u durdurur.

---

## 12. Spring Boot entegrasyonu

`tez_springBootAPI` lokal makinede çalışır, FastAPI URL'sini env var ile alır.

### Spring Boot setup

**`application.properties`** (env var fallback ile):
```properties
app.fastapi.base-url=${FASTAPI_BASE_URL:http://localhost:8000}
spring.servlet.multipart.max-request-size=25MB
```

**`WebClientConfig.java`** — 5 dakika read timeout (OCR + NLP toplam ~3-4 dk):
```java
HttpClient httpClient = HttpClient.create()
    .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 30_000)
    .responseTimeout(Duration.ofMinutes(5))
    .doOnConnected(c -> c.addHandlerLast(new ReadTimeoutHandler(300, SECONDS)));
```

### Demo akışı

```powershell
# 1. Colab'dan ngrok URL'i kopyaladıktan sonra IntelliJ Run Config:
#    Environment variables → FASTAPI_BASE_URL=https://abcd-1234.ngrok-free.app

# 2. Windows Firewall (ilk seferlik, port 8080 telefondan erişim için):
New-NetFirewallRule -DisplayName "Spring Boot 8080" -Direction Inbound `
  -LocalPort 8080 -Protocol TCP -Action Allow

# 3. PC'nin lokal IP'sini öğren (telefon bu IP'ye konuşur):
ipconfig | findstr "IPv4"
# Örnek: 192.168.1.42

# 4. IntelliJ'de Run → Spring Boot ayağa kalkar
```

### Android (Kotlin)

```kotlin
private const val API_BASE = "http://192.168.1.42:8080/api"   // PC'nin lokal IP'si
// ❌ localhost olmaz, telefonda kendi IP'sine bakar
// Telefon ve PC aynı wifi'de olmalı
```

---

## 13. Test sonuçları

### Vaka 1 — ornek3.jpeg (lise kimya sınavı, el yazısı)

| Soru | Tip | Skor | Detay |
|---|---|---|---|
| 1 | open_ended | 10/10 | Kimya denklemi, SBERT benzerlik 0.96 |
| 2 | open_ended | 7/10 | Fill-in-the-blank içerik, OCR el yazısı zorlandı |
| 3 | open_ended | 10/10 | SBERT benzerlik 0.85 |
| 4 | open_ended | 10/10 | SBERT benzerlik 1.00 |
| 5 | open_ended | 10/10 | SBERT benzerlik 0.91 |
| **TOPLAM** | | **47/50 (%94)** | |

### Vaka 2 — Ahmet Yesevi Ortaokulu (karışık tip sınav)

Section başlıkları + farklı tipler birarada.

| Soru | Tip | Scorer | Skor |
|---|---|---|---|
| mc1 | multiple_choice | MultipleChoiceScorer | 10/10 |
| fb1 | fill_blank | FillBlankScorer | 10/10 |
| oe1 | open_ended | OpenEndedScorer (SBERT) | 10/10 |
| oe2 | open_ended | OpenEndedScorer | 10/10 |
| oe3 | open_ended | OpenEndedScorer | 10/10 |
| **TOPLAM** | | | **50/50** |

(mc2 ve fb2 OCR atladı — VLM tekrarlı section başlıklarını birleştirdi. Bilinen sınır.)

### Unit tests

```
44 passed in 0.41s
  - tests/test_ocr_parser.py: 20 test (section detection, compound IDs, sub-question grouping)
  - tests/test_nlp_service.py: 7 test (dispatch + answer_key mapping)
  - tests/test_scoring.py: 17 test (4 scorer + dispatcher)
```

---

## 14. Sınırlamalar ve gelecek çalışmalar

### Bilinen sınırlamalar

1. **VLM tekrarlı section başlıklarını birleştirebilir** — `*çoktan seçmeli` iki kez
   ardışık yazılırsa, VLM bazen bir tanesini OCR çıkışına almaz. Sonraki soru
   default OPEN_ENDED'a düşer.

2. **MC işaretli şık tespiti zayıf** — Öğrenci A/B/C/D yazarsa OK, ama sadece
   yuvarlağa alıp/karaladıysa Qwen bazen okuyamaz. Future: OpenCV ile koyu pixel
   yoğunluğu karşılaştırması.

3. **Qwen2.5-VL Türkçe el yazısında "orta" kalite** — bazı kelimelerde typo
   yapar ("seçmeli" → "seçimli"). Parser bu varyantları kabul ediyor ama
   kalitesi kullanıcı yazısının zorluğuna bağlı.

4. **Demo deployment kırılgan** — Colab session 12 saatte timeout, ngrok URL
   değişir. Production için self-hosted GPU server veya ngrok Pro gerek.

5. **SBERT skorlama eşikleri tahmini** — `0.85 / 0.65 / 0.40` thresholds
   experimental olarak ayarlandı, geniş bir validation set üzerinde
   optimize edilmedi.

### Future work

- **OpenCV bazlı MC işaretleme tespiti**
- **Daha büyük VLM** (Qwen2.5-VL-32B veya 72B, A100 GPU üzerinde)
- **Soru tipine göre fine-tuned SBERT** (matematik için sembol-aware vs.)
- **Self-hosted production deployment** (Hetzner GPU + Docker)
- **Web frontend** (mevcut Android'in yanına)
- **Validation set ile eşik optimizasyonu**

---

## Lisans

Akademik kullanım — lisans tezi kapsamında. Detay için yazara başvur.

## İletişim

**Yazar:** Nebi Avsar
**GitHub:** [@nebiavsar](https://github.com/nebiavsar)
**Repo:** [tez_fastAPI](https://github.com/nebiavsar/tez_fastAPI)
