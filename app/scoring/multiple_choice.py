"""MULTIPLE_CHOICE — çoktan seçmeli için exact letter match (A/B/C/D/E)."""

from __future__ import annotations

import re

from app.scoring.base import ScoreResult, Scorer


_LETTER_RE = re.compile(r"[A-E]", re.IGNORECASE)


def _extract_letter(text: str) -> str | None:
    """Metinden tek harf cevabı çıkar (A/B/C/D/E)."""
    match = _LETTER_RE.search(text)
    if match is None:
        return None
    return match.group(0).upper()


class MultipleChoiceScorer(Scorer):
    """Tek harf eşleşme: doğruysa tam puan, değilse 0."""

    def score(self, *, expected: str, student: str, max_score: int) -> ScoreResult:
        expected_letter = _extract_letter(expected)
        student_letter = _extract_letter(student)

        if expected_letter is None:
            return ScoreResult(
                score=0,
                feedback="Referans cevap (doğru şık) parse edilemedi.",
                similarity=0.0,
            )

        if student_letter is None:
            return ScoreResult(
                score=0,
                feedback="Öğrenci işaretlemesi okunamadı.",
                similarity=0.0,
            )

        if student_letter == expected_letter:
            return ScoreResult(
                score=max_score,
                feedback=f"Doğru ({student_letter}).",
                similarity=1.0,
            )

        return ScoreResult(
            score=0,
            feedback=f"Yanlış — öğrenci: {student_letter}, doğru: {expected_letter}.",
            similarity=0.0,
        )
