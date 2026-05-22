"""FILL_BLANK — boşluk doldurma için exact match + Türkçe normalize."""

from __future__ import annotations

import unicodedata

from app.scoring.base import ScoreResult, Scorer


def _normalize(text: str) -> str:
    """Türkçe karakterleri ASCII'ye yaklaştır, lowercase, fazla boşluk temizle.

    Örnek:
        "Fiziksel"      → "fiziksel"
        "FİZİKSEL"      → "fiziksel"
        " kimyasal. "   → "kimyasal"
        "kimyasall"     → "kimyasall"  (yazım hatası korunur — exact match için)
    """
    # NFKD normalize: Türkçe karakterleri ayrıştır
    text = unicodedata.normalize("NFKD", text)
    # Lowercase
    text = text.lower()
    # Türkçe özel haritalama (ı → i, ş → s, ç → c, vb.)
    tr_map = str.maketrans({
        "ı": "i", "İ": "i", "ş": "s", "Ş": "s",
        "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u",
        "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
    })
    text = text.translate(tr_map)
    # Trailing noktalama + fazla boşluk
    text = "".join(c for c in text if c.isalnum() or c.isspace())
    return " ".join(text.split())


class FillBlankScorer(Scorer):
    """Tek kelime / kısa ifade için exact match (normalize sonrası).

    Tam eşleşme → max_score
    Kısmi (öğrenci cevabı referans içinde geçiyor) → max_score * 0.7
    Yoksa → 0
    """

    def score(self, *, expected: str, student: str, max_score: int) -> ScoreResult:
        e_norm = _normalize(expected)
        s_norm = _normalize(student)

        if not s_norm:
            return ScoreResult(score=0, feedback="Cevap boş.", similarity=0.0)

        if e_norm == s_norm:
            return ScoreResult(score=max_score, feedback="Doğru.", similarity=1.0)

        # Kısmi: tokenler kesişiyor mu
        e_tokens = set(e_norm.split())
        s_tokens = set(s_norm.split())
        if e_tokens & s_tokens:
            return ScoreResult(
                score=round(max_score * 0.7),
                feedback="Kısmen doğru — anahtar kelime eşleşti.",
                similarity=len(e_tokens & s_tokens) / max(len(e_tokens), 1),
            )

        return ScoreResult(score=0, feedback="Yanlış.", similarity=0.0)
