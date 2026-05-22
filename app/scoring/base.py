"""Scorer arayüzü ve sonuç tipi."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreResult:
    """Tek bir sorunun skorlanma sonucu."""

    score: int
    feedback: str
    similarity: float | None = None  # debug/log için; bazı scorer'lar None döner


class Scorer(ABC):
    """Bütün soru tipi scorer'larının ortak arayüzü."""

    @abstractmethod
    def score(self, *, expected: str, student: str, max_score: int) -> ScoreResult:
        """Öğrenci cevabını referansa göre skorla.

        Args:
            expected: Öğretmenin doğru cevabı (cevap kâğıdı OCR çıkışı)
            student: Öğrencinin verdiği cevap (sınav kâğıdı OCR çıkışı)
            max_score: Sorunun tam puanı

        Returns:
            ScoreResult: skor, geri bildirim, opsiyonel benzerlik
        """
        raise NotImplementedError
