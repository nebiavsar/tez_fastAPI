"""Qwen2.5-VL'in OCR çıkışını OCRDetectedAnswer listesine parse eder.

Desteklenen format'lar (Qwen tutarsız üretiyor):
    **1)** ana soru cevabı
    **3**) yıldız konumu farklı
    2a)** sub-question (yeni — 2026-05-22 prompt v2 sonrası)
    3b)** sub-question
    5)   düz format (fallback)

Sub-question'lar ana soru altında a) ... b) ... formatıyla gruplanır:
    "2a)** Karasal\n2b)** Kimyasal"  →  OCRDetectedAnswer(question_number="2", extracted_answer="a) Karasal\nb) Kimyasal")

Safety net'ler (2026-05-22 v2 fix):
  - Block içinde başka soru numarası varsa split eder
  - Halüsinasyon satırlarını filtreler ("Yanlış cevap:", "Not:", ...)
  - "**N)** (boş)" markörünü atar

question_text alanı boş bırakılır — soru metni Spring Boot tarafındaki
GroupAnswerKey'den gelecek.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import NamedTuple

from app.schemas import OCRDetectedAnswer, QuestionType

logger = logging.getLogger(__name__)

# Tip etiketi regex'i — [open_ended], [fill_blank], [matching], [multiple_choice]
# Header'dan hemen sonra opsiyonel olarak gelir.
_TYPE_TAG_RE = re.compile(
    r"^\s*\[\s*(open_ended|fill_blank|matching|multiple_choice)\s*\]\s*",
    re.IGNORECASE,
)

_TYPE_MAP = {
    "open_ended": QuestionType.OPEN_ENDED,
    "fill_blank": QuestionType.FILL_BLANK,
    "matching": QuestionType.MATCHING,
    "multiple_choice": QuestionType.MULTIPLE_CHOICE,
}

# Header regex — hem ana hem sub-question yakalar.
# Örnekler:
#   **1)**  → main='1', sub=''
#   **3**)  → main='3', sub=''
#   2a)**   → main='2', sub='a'
#   5b)     → main='5', sub='b'
#   1)      → main='1', sub=''
#   1.      → main='1', sub=''  (düz fallback, "1." formatı)
# Anchor: satır başı (MULTILINE) — cümle ortasındaki '(1)' gibi şeyleri yakalamasın.
_FULL_HEADER_RE = re.compile(
    r"^[ \t]*\*?\*?\s*(\d+)([a-eçğöşüı]?)\s*\*?\*?\s*[\)\.]\s*\*?\*?",
    re.IGNORECASE | re.MULTILINE,
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

_EMPTY_MARKERS = ("(boş)", "(bos)", "[boş]", "[bos]", "(empty)")


class _Block(NamedTuple):
    main: str            # ana soru numarası ("1", "2", ...)
    sub: str             # sub-letter ("", "a", "b", ...)
    body: str            # cevap metni
    qtype: QuestionType  # VLM tarafından atanan tip


def parse_s3_markdown(text: str) -> list[OCRDetectedAnswer]:
    """Qwen çıkışını OCRDetectedAnswer listesine dönüştür.

    Sub-question'lar ana soru altında gruplanır.
    """
    headers = list(_FULL_HEADER_RE.finditer(text))
    if not headers:
        logger.warning(
            "Hiçbir soru başlığı bulunamadı. Ham metin uzunluğu: %d karakter",
            len(text),
        )
        return []

    # Her header için body'i (sonraki header'a kadar olan metin) çıkar
    blocks: list[_Block] = []
    for i, header in enumerate(headers):
        body_start = header.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end].strip()

        # Tip etiketini çıkar (varsa); body'den ayır
        qtype, body = _extract_type_tag(body)

        body = _filter_hallucinations(body)
        if _is_empty_answer(body):
            logger.debug("Soru %s%s boş cevap, atlandı", header.group(1), header.group(2) or "")
            continue
        # Tek karakter (örn. eşleştirme sayısı "4") geçerli cevap olabilir → < 1 kontrolü
        if len(body.strip()) < 1:
            continue

        blocks.append(
            _Block(
                main=header.group(1),
                sub=(header.group(2) or "").lower(),
                body=_clean_body(body),
                qtype=qtype,
            )
        )

    # Ana soru numarasına göre grupla
    by_main: dict[str, list[_Block]] = defaultdict(list)
    for block in blocks:
        by_main[block.main].append(block)

    # Her gruptan OCRDetectedAnswer oluştur
    answers: list[OCRDetectedAnswer] = []
    for main_num in sorted(by_main.keys(), key=lambda x: int(x)):
        parts = by_main[main_num]
        answer_text = _format_grouped_answer(parts)
        if not answer_text:
            continue
        # Grup içindeki tipler — sub-question'lar farklı tipte olabilir.
        # Çoğunluk tipini al; eşitlikse ilkini, hepsi UNKNOWN'sa OPEN_ENDED default.
        qtype = _resolve_group_type([p.qtype for p in parts])
        answers.append(
            OCRDetectedAnswer(
                question_number=main_num,
                question_text="",
                extracted_answer=answer_text,
                question_type=qtype,
            )
        )

    return answers


def _extract_type_tag(body: str) -> tuple[QuestionType, str]:
    """Body'nin başındaki [tip] etiketini ayır.

    Dönüş: (tip, etiket atılmış body)
    Etiket yoksa: (UNKNOWN, body)
    """
    match = _TYPE_TAG_RE.match(body)
    if match is None:
        return QuestionType.UNKNOWN, body
    tag = match.group(1).lower()
    qtype = _TYPE_MAP.get(tag, QuestionType.UNKNOWN)
    return qtype, body[match.end():]


def _resolve_group_type(types: list[QuestionType]) -> QuestionType:
    """Sub-question'lar farklı tip etiketi taşıyabilir — gruba tek tip seç.

    Sıra: çoğunluk → tek tip varsa o → UNKNOWN varsa OPEN_ENDED'a düşür.
    """
    non_unknown = [t for t in types if t != QuestionType.UNKNOWN]
    if not non_unknown:
        return QuestionType.OPEN_ENDED  # fallback
    # Çoğunluk
    counts: dict[QuestionType, int] = defaultdict(int)
    for t in non_unknown:
        counts[t] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _format_grouped_answer(parts: list[_Block]) -> str:
    """Bir ana sorunun parçalarını tek bir cevap metnine birleştir.

    Tek parça ve sub yoksa → düz body.
    Birden fazla parça veya sub varsa → "a) ... b) ..." formatlı.
    """
    if not parts:
        return ""

    if len(parts) == 1 and not parts[0].sub:
        return parts[0].body

    lines: list[str] = []
    for part in parts:
        if part.sub:
            lines.append(f"{part.sub}) {part.body}")
        else:
            lines.append(part.body)
    return "\n".join(lines)


def _filter_hallucinations(text: str) -> str:
    """Halüsinasyon satırlarını ele."""
    kept = []
    for line in text.split("\n"):
        stripped_lower = line.strip().lower()
        if any(stripped_lower.startswith(prefix) for prefix in _HALLUCINATION_PREFIXES):
            logger.debug("Halüsinasyon satırı atıldı: %s", line.strip()[:80])
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _is_empty_answer(text: str) -> bool:
    stripped = text.strip().lower()
    return any(marker in stripped for marker in _EMPTY_MARKERS) and len(stripped) < 30


def _clean_body(text: str) -> str:
    lines = [line.strip().lstrip("-*•").strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)
