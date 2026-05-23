"""Sınav kâğıdı OCR çıkışını OCRDetectedAnswer listesine parse eder.

**Standart sınav formatı (tez spesifikasyonu, 2026-05-23):**

    *çoktan seçmeli soru               ← Section başlığı (yıldızlı)
    1) Soru metni (10p)                ← Numara + soru + puan
    A) Şık A
    B) Şık B
    ...

    *boşluk doldurma                   ← FB section
    1) Cümle ............'dir. (5p)

    1-) Açık uçlu soru? (10p)          ← Section başlığı yok → OPEN_ENDED
    Cevap: öğrenci yazısı

**3 strateji:**
    *çoktan seçmeli / *multiple choice (+ varyantları)  → MULTIPLE_CHOICE
    *boşluk doldurma / *eşleştirme / *fill in the blank → FILL_BLANK
    (yıldız yok veya *açık uçlu)                         → OPEN_ENDED

**Tekrarlı numara handling (mc1 vs fb1 vs oe1):**
Her section'da numaralandırma 1'den başlayabilir. Parser section prefix
ekleyerek global unique ID üretir:
    *çoktan seçmeli + 1) → 'mc1'
    *boşluk doldurma + 1) → 'fb1'
    (section yok) + 1) → 'oe1'

Bu sayede iki "Soru 1" karıştırılmaz; cevap kâğıdı ile öğrenci kâğıdı
deterministik şekilde eşleşir.

**Puan parse:**
Soru başlığından sonra (10p), (5 puan), (10 pt) gibi formatlar → max_score.
Yoksa default 10.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import NamedTuple

from app.schemas import OCRDetectedAnswer, QuestionType

logger = logging.getLogger(__name__)

# Soru başlığı varyantları:
#   1)            düz
#   **1)**        markdown bold
#   ### 1)        markdown header
#   1.            nokta ile
#   1-)           tire ile (açık uçlu bölümde gözlemlendi)
#   2a)           sub-question
_FULL_HEADER_RE = re.compile(
    r"^[ \t]*#{0,4}\s*\*?\*?\s*(\d+)([a-eçğöşüı]?)\s*\*?\*?\s*[\-]?\s*[\)\.]\s*\*?\*?",
    re.IGNORECASE | re.MULTILINE,
)

# Yıldızlı section başlığı: *çoktan seçmeli soru, *boşluk doldurma, vs.
# Yıldızdan sonra ne yazıyorsa al, classify et.
_STARRED_SECTION_RE = re.compile(
    r"^\s*\*\s*([^*\n]+?)\s*$",
    re.MULTILINE,
)

# Section anahtar kelimeleri — tip eşleştirme
_SECTION_KEYWORDS: list[tuple[QuestionType, tuple[str, ...]]] = [
    (
        QuestionType.MULTIPLE_CHOICE,
        (
            "çoktan seçmeli soru",
            "çoktan seçmeli sorular",
            "çoktan seçmeli",
            "multiple choice questions",
            "multiple choice question",
            "multiple choice",
            "multiple-choice",
            "test soru",
            "test sorular",
        ),
    ),
    (
        QuestionType.FILL_BLANK,
        (
            "boşluk doldurma",
            "bosluk doldurma",
            "eşleştirme sorular",
            "eşleştirme",
            "eslestirme",
            "fill in the blanks",
            "fill in the blank",
            "matching questions",
            "matching",
        ),
    ),
    (
        QuestionType.OPEN_ENDED,
        (
            "açık uçlu",
            "acik uclu",
            "klasik",
            "kısa cevaplı",
            "kisa cevapli",
            "uzun cevaplı",
            "uzun cevapli",
            "open ended",
            "open-ended",
        ),
    ),
]

# Puan değeri: (10p), (5 p), (10puan), (10 puan), (10pt), (10 points)
_POINTS_RE = re.compile(
    r"\(\s*(\d+)\s*(?:p|puan|pt|point|points)\.?\s*\)",
    re.IGNORECASE,
)

# Halüsinasyon paternleri
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

# Section type → question_number prefix
_TYPE_PREFIX: dict[QuestionType, str] = {
    QuestionType.MULTIPLE_CHOICE: "mc",
    QuestionType.FILL_BLANK: "fb",
    QuestionType.OPEN_ENDED: "oe",
}


class _Block(NamedTuple):
    main: str            # ana soru numarası ("1", "2", ...)
    sub: str             # sub-letter ("", "a", "b", ...)
    body: str            # cevap metni
    qtype: QuestionType  # section başlığından atanan tip
    max_score: int       # (Np) formatından parse edilen puan


def _classify_section_text(text: str) -> QuestionType | None:
    """Yıldızlı satırın içeriğine bakıp tipi belirle. Eşleşmezse None."""
    text_lower = text.lower().strip()
    for qtype, keywords in _SECTION_KEYWORDS:
        for kw in keywords:
            if kw in text_lower:
                return qtype
    return None


def _find_starred_sections(text: str) -> list[tuple[int, QuestionType]]:
    """Yıldızlı section başlıklarını bul.

    Her *X satırı için _classify_section_text ile tip belirle.
    Tanımsız yıldızlı satırlar atlanır (örn. *Not:).
    """
    sections: list[tuple[int, QuestionType]] = []
    for match in _STARRED_SECTION_RE.finditer(text):
        section_text = match.group(1)
        qtype = _classify_section_text(section_text)
        if qtype is not None:
            sections.append((match.start(), qtype))
            logger.debug("Section başlığı tespit: '%s' → %s @ pos=%d",
                         section_text, qtype.value, match.start())
    sections.sort(key=lambda x: x[0])
    return sections


def _section_for_header(
    sections: list[tuple[int, QuestionType]],
    headers: list[re.Match[str]],
    header_idx: int,
) -> QuestionType:
    """One-shot section başlığı: her section başlığı SADECE kendinden sonraki
    ilk soruya tip atar.

    Mantık: Verilen sorunun (header_idx) HEMEN ÖNCESİNDE — önceki sorudan
    SONRA — yer alan section başlığını bul. Yoksa default OPEN_ENDED.

    Bu sayede 'sticky section' problemi çözülür:
        *boşluk doldurma
        1) (5p) ... → FB
        *boşluk doldurma
        2) (5p) ... → FB
        1-) (10p) ... → OE  (section başlığı yok, default)
    """
    current_pos = headers[header_idx].start()
    prev_header_end = headers[header_idx - 1].end() if header_idx > 0 else 0

    # Önceki soru ile şu an arasında yer alan section başlığı varsa al
    for section_pos, section_type in sections:
        if prev_header_end <= section_pos < current_pos:
            return section_type
    return QuestionType.OPEN_ENDED


def _extract_points(text: str, default: int = 10) -> int:
    """Metinden (Np), (N puan) gibi puan değerini çıkar. Yoksa default."""
    match = _POINTS_RE.search(text)
    if match:
        return int(match.group(1))
    return default


def parse_s3_markdown(text: str) -> list[OCRDetectedAnswer]:
    """OCR markdown çıkışını OCRDetectedAnswer listesine dönüştür.

    Section başlıklarına göre tip atanır. Question ID compound (mc1/fb1/oe1)
    olarak üretilir. Puan değeri parse edilirse max_score'a atılır.
    """
    headers = list(_FULL_HEADER_RE.finditer(text))
    if not headers:
        logger.warning(
            "Hiçbir soru başlığı bulunamadı. Ham metin uzunluğu: %d karakter",
            len(text),
        )
        return []

    sections = _find_starred_sections(text)

    blocks: list[_Block] = []
    for i, header in enumerate(headers):
        body_start = header.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body_raw = text[body_start:body_end].strip()

        # Tipi belirle: bu sorunun HEMEN ÖNCESİNDEKİ section başlığı (one-shot)
        qtype = _section_for_header(sections, headers, i)

        # Puan: hem header'ın kendi satırı hem body içinde olabilir
        # Header'ın bittiği yer ile satır sonu arasındaki kısma da bak
        header_line_end = text.find("\n", header.end())
        if header_line_end == -1:
            header_line_end = len(text)
        header_remainder = text[header.end():header_line_end]
        max_score = _extract_points(header_remainder, default=0) or _extract_points(body_raw, default=10)

        body = _filter_hallucinations(body_raw)
        if _is_empty_answer(body):
            logger.debug("Soru %s%s boş cevap, atlandı", header.group(1), header.group(2) or "")
            continue
        if len(body.strip()) < 1:
            continue

        blocks.append(
            _Block(
                main=header.group(1),
                sub=(header.group(2) or "").lower(),
                body=_clean_body(body),
                qtype=qtype,
                max_score=max_score,
            )
        )

    # Compound key (qtype_prefix + main) bazlı grupla
    # Aynı section'da sub-question'lar birleşir (mc1a + mc1b → mc1 altında a)..b)..)
    by_key: dict[str, list[_Block]] = defaultdict(list)
    for block in blocks:
        prefix = _TYPE_PREFIX.get(block.qtype, "oe")
        key = f"{prefix}{block.main}"
        by_key[key].append(block)

    answers: list[OCRDetectedAnswer] = []
    # Sıralama: önce qtype (mc < fb < oe), sonra numara
    sort_order = {"mc": 0, "fb": 1, "oe": 2}
    for key in sorted(by_key.keys(), key=lambda k: (sort_order.get(k[:2], 99), int(k[2:]))):
        parts = by_key[key]
        answer_text = _format_grouped_answer(parts)
        if not answer_text:
            continue
        qtype = parts[0].qtype
        # En yüksek max_score (sub-question'lardan)
        max_score = max(p.max_score for p in parts)
        answers.append(
            OCRDetectedAnswer(
                question_number=key,
                question_text="",
                extracted_answer=answer_text,
                question_type=qtype,
                max_score=max_score,
            )
        )

    return answers


def _format_grouped_answer(parts: list[_Block]) -> str:
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
