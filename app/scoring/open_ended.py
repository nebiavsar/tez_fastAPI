"""OPEN_ENDED — SBERT semantic skorlama (klasik açık uçlu sorular)."""

from __future__ import annotations

from app.ml.sbert_loader import SBERTLoader
from app.scoring.base import ScoreResult, Scorer


def similarity_to_score(similarity: float, max_score: int) -> tuple[int, str]:
    """Kosinüs benzerliği → tam sayı puan + öğrenciye geri bildirim.

    Eşik politikası ([[Semantik Skorlama]] vault sayfası):
        ≥ 0.85       → %100 tam puan
        0.65 - 0.85  → %60-80 (lineer interpolasyon)
        0.40 - 0.65  → %20-40 (lineer interpolasyon)
        < 0.40       → 0 puan
    """
    if similarity >= 0.85:
        return max_score, "Doğru."
    if similarity >= 0.65:
        ratio = 0.6 + 0.2 * (similarity - 0.65) / 0.20
        return round(max_score * ratio), "Kısmen doğru, eksik kavramlar var."
    if similarity >= 0.40:
        ratio = 0.2 + 0.2 * (similarity - 0.40) / 0.25
        return round(max_score * ratio), "Yakın ama yetersiz."
    return 0, "Yanlış veya konuyla ilgisiz."


class OpenEndedScorer(Scorer):
    """SBERT kosinüs benzerliğiyle skorlama."""

    def __init__(self, sbert_loader: SBERTLoader) -> None:
        self._loader = sbert_loader

    def score(self, *, expected: str, student: str, max_score: int) -> ScoreResult:
        similarity = self._loader.similarity(reference=expected, student=student)
        score, base_feedback = similarity_to_score(similarity, max_score)
        return ScoreResult(
            score=score,
            feedback=f"{base_feedback} (benzerlik: {similarity:.2f})",
            similarity=similarity,
        )
