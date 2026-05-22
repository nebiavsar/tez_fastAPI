"""MATCHING — eşleştirme soruları için pair-by-pair exact match.

Beklenen format (referans veya öğrenci):
    "a→4, b→2, c→9"        (ok yön)
    "a) 4, b) 2, c) 9"     (parantezli)
    "a-4, b-2, c-9"        (tire)
    "a: 4\nb: 2\nc: 9"     (çok satır)

Her doğru çift için max_score / toplam_çift_sayısı kadar puan.
"""

from __future__ import annotations

import re

from app.scoring.base import ScoreResult, Scorer


# "a→4", "a) 4", "a-4", "a:4", "a 4" varyantları
_PAIR_RE = re.compile(
    r"([a-eçğöşüı])\s*[\-\)\:\→\>]+\s*(\w+)",
    re.IGNORECASE,
)


def _parse_pairs(text: str) -> dict[str, str]:
    """Metinden harf→değer çiftlerini çıkar."""
    pairs: dict[str, str] = {}
    for match in _PAIR_RE.finditer(text):
        letter = match.group(1).lower()
        value = match.group(2).strip().lower()
        pairs[letter] = value
    return pairs


class MatchingScorer(Scorer):
    def score(self, *, expected: str, student: str, max_score: int) -> ScoreResult:
        expected_pairs = _parse_pairs(expected)
        student_pairs = _parse_pairs(student)

        if not expected_pairs:
            # Referans parse edilemedi — fallback olarak fill-blank gibi davran
            return ScoreResult(
                score=0,
                feedback="Referans cevap parse edilemedi (eşleştirme formatı bulunamadı).",
                similarity=0.0,
            )

        correct = 0
        total = len(expected_pairs)
        for letter, expected_value in expected_pairs.items():
            if student_pairs.get(letter) == expected_value:
                correct += 1

        score = round(max_score * (correct / total)) if total else 0
        feedback = f"{correct}/{total} eşleşme doğru."

        return ScoreResult(
            score=score,
            feedback=feedback,
            similarity=correct / total if total else 0.0,
        )
