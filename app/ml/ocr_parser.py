"""Sınav kâğıdı OCR çıkışını OCRDetectedAnswer listesine parse eder.

**Tasarım kararı (2026-05-23 v3):** Soru tipi tespiti **section başlıklarına** dayanır,
artık pattern heuristic'lerine değil. Öğretmen sınav kâğıdı hazırlarken aşağıdaki
başlıkları kullanır:

    Çoktan Seçmeli Sorular         → o bölgedeki sorular MULTIPLE_CHOICE
    Multiple Choice Questions      →                       (aynı)

    Boşluk Doldurma                → o bölgedeki sorular FILL_BLANK
    Eşleştirme                     →                      (aynı)
    Fill in the Blanks             →                      (aynı)
    Matching                       →                      (aynı)

    (başka başlık yok)             → OPEN_ENDED (SBERT)

Soru başlıkları (**1)**, **2a)**, **3)**) bu section'lar arasında dağılır.
Her soru, kendinden önceki en son section'ın tipini alır.

Eski sürümlerde olan ama kaldırılan:
  - VLM '[open_ended]', '[fill_blank]' etiketi yazması (VLM uyumsuzdu)
  - Pattern heuristic'leri (tek harf → MC, "label: value" → FB, vs.)
  - QuestionType.MATCHING (FILL_BLANK ile birleştirildi)

Korunanlar:
  - Sub-question grouping (2a, 2b → "a) ... b) ...")
  - Halüsinasyon satır filtreleme ("Yanlış cevap:", "Not:", vs.)
  - Boş cevap markörü atlama ("**N)** (boş)")
  - Block içinde sızıntı kırpma (4. sorunun bloğuna 5'in içeriği sızarsa)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import NamedTuple

from app.schemas import OCRDetectedAnswer, QuestionType

logger = logging.getLogger(__name__)

# Soru başlığı: **1)**, **3**), 2a)**, 5)
_FULL_HEADER_RE = re.compile(
    r"^[ \t]*\*?\*?\s*(\d+)([a-eçğöşüı]?)\s*\*?\*?\s*[\)\.]\s*\*?\*?",
    re.IGNORECASE | re.MULTILINE,
)

# Section başlıkları — Türkçe + İngilizce
_SECTION_PATTERNS: list[tuple[QuestionType, re.Pattern[str]]] = [
    (
        QuestionType.MULTIPLE_CHOICE,
        re.compile(
            r"\b(multiple[\s\-]+choice(\s+questions?)?|çoktan\s+seçmeli(\s+sorular?)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        QuestionType.FILL_BLANK,
        re.compile(
            r"\b("
            r"boşluk\s+doldurma|"
            r"eşleştirme(\s+sorular[ıi])?|"
            r"fill[\s\-]+in[\s\-]+the[\s\-]+blanks?|"
            r"matching(\s+questions?)?"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]

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
    qtype: QuestionType  # section başlığından atanan tip


def _find_sections(text: str) -> list[tuple[int, QuestionType]]:
    """Metindeki tüm section başlıklarını ve pozisyonlarını bul.

    Dönüş: pozisyona göre sıralı [(start_index, type), ...]
    """
    sections: list[tuple[int, QuestionType]] = []
    for qtype, pattern in _SECTION_PATTERNS:
        for match in pattern.finditer(text):
            sections.append((match.start(), qtype))
    sections.sort(key=lambda x: x[0])
    return sections


def _section_for_position(sections: list[tuple[int, QuestionType]], pos: int) -> QuestionType:
    """Verilen pozisyondan ÖNCE gelen en son section'ı bul.

    Hiçbir section yoksa veya pozisyon ilk section'dan önceyse → OPEN_ENDED (default).
    """
    current = QuestionType.OPEN_ENDED
    for section_pos, section_type in sections:
        if section_pos < pos:
            current = section_type
        else:
            break
    return current


def parse_s3_markdown(text: str) -> list[OCRDetectedAnswer]:
    """OCR markdown çıkışını OCRDetectedAnswer listesine dönüştür.

    Section başlıklarına göre her soruya tip atanır. Sub-question'lar ana soru
    altında a) ... b) ... formatıyla gruplanır.
    """
    headers = list(_FULL_HEADER_RE.finditer(text))
    if not headers:
        logger.warning(
            "Hiçbir soru başlığı bulunamadı. Ham metin uzunluğu: %d karakter",
            len(text),
        )
        return []

    sections = _find_sections(text)
    if sections:
        logger.debug(
            "Section başlıkları bulundu: %s",
            [(s[1].value, s[0]) for s in sections],
        )

    blocks: list[_Block] = []
    for i, header in enumerate(headers):
        body_start = header.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end].strip()

        body = _filter_hallucinations(body)
        if _is_empty_answer(body):
            logger.debug("Soru %s%s boş cevap, atlandı", header.group(1), header.group(2) or "")
            continue
        if len(body.strip()) < 1:
            continue

        # Bu sorunun pozisyonundan önceki en son section → tip
        qtype = _section_for_position(sections, header.start())

        blocks.append(
            _Block(
                main=header.group(1),
                sub=(header.group(2) or "").lower(),
                body=_clean_body(body),
                qtype=qtype,
            )
        )

    # Ana soru numarasına göre grupla (sub-question'lar)
    by_main: dict[str, list[_Block]] = defaultdict(list)
    for block in blocks:
        by_main[block.main].append(block)

    answers: list[OCRDetectedAnswer] = []
    for main_num in sorted(by_main.keys(), key=lambda x: int(x)):
        parts = by_main[main_num]
        answer_text = _format_grouped_answer(parts)
        if not answer_text:
            continue
        # Aynı section içindeki sub-question'lar aynı tipte olur → ilk part'tan al
        qtype = parts[0].qtype
        answers.append(
            OCRDetectedAnswer(
                question_number=main_num,
                question_text="",
                extracted_answer=answer_text,
                question_type=qtype,
            )
        )

    return answers


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
