"""Qwen2.5-VL'in S3 markdown çıkışını OCRDetectedAnswer listesine parse eder.

Beklenen format (ocr_service.OCR_PROMPT sonucu):

    **1)** Pb(NO₃)₂(suda)+2KI(suda)->PbI₂(k)+2KNO₃(suda)

    **2)** Suyun buharlaşması: Kondensel
       Kağıdın yanması: Kimyasal

    **3)** ...

Parse safety-net'leri (2026-05-22 v2 fix):
  - Block içinde başka soru numarası varsa split eder (4-5 karışıklığı önlemek)
  - Halüsinasyon satırlarını filtreler ("Yanlış cevap:", "Not:", "Doğru cevap:")
  - "**N)** (boş)" → atılır (öğrenci cevap yazmamış)

question_text alanı boş bırakılır — soru metni Spring Boot tarafındaki
GroupAnswerKey'den gelecek (referans cevap ile birlikte).
"""

from __future__ import annotations

import logging
import re

from app.schemas import OCRDetectedAnswer

logger = logging.getLogger(__name__)

# Qwen'in gerçek dünyada ürettiği başlık varyantları (spike sonuçlarından):
#   **1)**   ← klasik markdown bold
#   **3**)   ← model bazen yıldızları sayının etrafına atıyor
#   **5)     ← bazen kapanış yıldızı eksik
_HEADER_INNER = r"\*\*\s*(\d+)\s*\*?\*?\s*\)\s*\*?\*?"
_HEADER_LOOKAHEAD = r"\n\s*\*\*\s*\d+\s*\*?\*?\s*\)\s*\*?\*?"

_ANSWER_BLOCK_RE = re.compile(
    rf"{_HEADER_INNER}\s*(.+?)(?={_HEADER_LOOKAHEAD}|\Z)",
    re.DOTALL,
)

# Fallback: **1)** yoksa "1)" veya "1." başlangıçları
_PLAIN_NUMBER_RE = re.compile(
    r"^\s*(\d+)\s*[\.\)]\s*(.+?)(?=\n\s*\d+\s*[\.\)]|\Z)",
    re.DOTALL | re.MULTILINE,
)

# Bir blok içindeki sızıntı: satır başında başka bir soru numarası gelirse
# o satırdan itibaren her şey o sorudan değildir.
_INLINE_NUMBER_LEAK_RE = re.compile(
    r"\n\s*(\*\*\s*\d+\s*\*?\*?\s*\)\s*\*?\*?|\d+\s*[\.\)])",
    re.MULTILINE,
)

# Halüsinasyon paternleri — model'in kendi yorumladığı satırlar
_HALLUCINATION_PREFIXES = (
    "yanlış cevap",
    "doğru cevap",
    "not:",
    "açıklama:",
    "çözüm:",
    "değerlendirme:",
    "yorum:",
)

# "Boş cevap" markörü (yeni prompt'tan)
_EMPTY_MARKERS = ("(boş)", "(bos)", "[boş]", "[bos]", "(empty)")


def parse_s3_markdown(text: str) -> list[OCRDetectedAnswer]:
    """Qwen S3 prompt çıkışını OCRDetectedAnswer listesine dönüştür."""
    matches = list(_ANSWER_BLOCK_RE.finditer(text))
    if not matches:
        logger.debug("Markdown **N)** formatı bulunamadı, düz numara formatına geçiliyor")
        matches = list(_PLAIN_NUMBER_RE.finditer(text))

    answers: list[OCRDetectedAnswer] = []
    for match in matches:
        question_number = match.group(1).strip()
        body = match.group(2).strip()

        # Safety net 1: blok içinde başka soru numarası sızıntısı varsa kırp
        body = _truncate_at_leak(body)

        # Safety net 2: halüsinasyon satırlarını ele
        body = _filter_hallucinations(body)

        # Safety net 3: boş cevap markörünü tespit et — atla
        if _is_empty_answer(body):
            logger.debug("Soru %s boş cevap olarak atlandı", question_number)
            continue

        # Çok kısa veya boş cevapları atla (gürültü)
        if len(body) < 2:
            continue

        answers.append(
            OCRDetectedAnswer(
                question_number=question_number,
                question_text="",  # Spring Boot tarafından doldurulacak
                extracted_answer=_clean_body(body),
            )
        )

    if not answers:
        logger.warning(
            "S3 çıkışından soru bloğu çıkarılamadı. Ham metin uzunluğu: %d karakter",
            len(text),
        )

    return answers


def _truncate_at_leak(text: str) -> str:
    """Bir blok içindeki sonraki soru numarası başlığında kırp.

    Önceki test: Qwen soru 4'ün bloğuna soru 5'in cevabını yazıyordu. Buradaki
    regex blok içinde sonraki '**M)**' veya 'M)' başlığı bulursa o noktada
    metni keser; sonrası ileriki soruya aittir, ana regex onu zaten ayrı match'le yakalar.
    """
    leak = _INLINE_NUMBER_LEAK_RE.search(text)
    if leak is None:
        return text
    return text[: leak.start()].rstrip()


def _filter_hallucinations(text: str) -> str:
    """Halüsinasyon satırlarını ('Yanlış cevap:', 'Not:' vb.) ele."""
    kept = []
    for line in text.split("\n"):
        stripped_lower = line.strip().lower()
        if any(stripped_lower.startswith(prefix) for prefix in _HALLUCINATION_PREFIXES):
            logger.debug("Halüsinasyon satırı atıldı: %s", line.strip()[:80])
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _is_empty_answer(text: str) -> bool:
    """Öğrenci cevap yazmamış mı (boş markör)?"""
    stripped = text.strip().lower()
    return any(marker in stripped for marker in _EMPTY_MARKERS) and len(stripped) < 30


def _clean_body(text: str) -> str:
    """Cevap metnini temizle — fazla boşluk, leading bullet'ler, vs."""
    lines = [line.strip().lstrip("-*•").strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)
