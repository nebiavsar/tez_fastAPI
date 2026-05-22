"""Qwen2.5-VL'in S3 markdown çıkışını OCRDetectedAnswer listesine parse eder.

Beklenen format (spike_qwen_vl.py S3 prompt sonucu):

    Tabii ki! İşte öğrencinin el ile yazdığı cevaplar:

    **1)** Pb(NO₃)₂(suda)+2KI(suda)->PbI₂(k)+2KNO₃(suda)

    **2)** Suyun buharlaşması: Kondensel
       Kağıdın yanması: Kimyasal
       ...

    **3)** ...

Parse:
    [
        OCRDetectedAnswer(question_number="1", extracted_answer="Pb(NO₃)₂..."),
        OCRDetectedAnswer(question_number="2", extracted_answer="Suyun buharlaşması: Kondensel\nKağıdın yanması: Kimyasal\n..."),
        ...
    ]

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
# Aşağıdaki regex hepsini kabul eder.
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


def parse_s3_markdown(text: str) -> list[OCRDetectedAnswer]:
    """Qwen S3 prompt çıkışını OCRDetectedAnswer listesine dönüştür.

    Önce markdown **N)** formatını dener, bulamazsa düz "N)" / "N." dener.
    """
    matches = list(_ANSWER_BLOCK_RE.finditer(text))
    if not matches:
        logger.debug("Markdown **N)** formatı bulunamadı, düz numara formatına geçiliyor")
        matches = list(_PLAIN_NUMBER_RE.finditer(text))

    answers: list[OCRDetectedAnswer] = []
    for match in matches:
        question_number = match.group(1).strip()
        body = match.group(2).strip()
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


def _clean_body(text: str) -> str:
    """Cevap metnini temizle — fazla boşluk, leading bullet'ler, vs."""
    # Satır başındaki markdown bullet ve fazla boşlukları temizle
    lines = [line.strip().lstrip("-*•").strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)
