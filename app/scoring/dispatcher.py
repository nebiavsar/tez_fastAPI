"""Tip bazlı scorer dispatcher — `score_answer` ana giriş noktası.

3 strateji (2026-05-23 revize, MatchingScorer kaldırıldı):
    OPEN_ENDED      → OpenEndedScorer (SBERT semantic)         — default
    FILL_BLANK      → FillBlankScorer (exact + TR normalize + token kesişimi)
                       Eşleştirme cevapları da burada handle edilir.
    MULTIPLE_CHOICE → MultipleChoiceScorer (letter exact)
"""

from __future__ import annotations

import logging

from app.ml.sbert_loader import SBERTLoader
from app.schemas import QuestionType
from app.scoring.base import ScoreResult
from app.scoring.fill_blank import FillBlankScorer
from app.scoring.multiple_choice import MultipleChoiceScorer
from app.scoring.open_ended import OpenEndedScorer

logger = logging.getLogger(__name__)


def score_answer(
    *,
    question_type: QuestionType,
    expected: str,
    student: str,
    max_score: int,
    sbert_loader: SBERTLoader,
) -> ScoreResult:
    """Tip bazlı dispatch — doğru scorer'ı seç ve çağır.

    UNKNOWN tip OPEN_ENDED gibi işlenir (en yaygın varsayılan).
    """
    effective_type = (
        question_type if question_type != QuestionType.UNKNOWN else QuestionType.OPEN_ENDED
    )

    if effective_type == QuestionType.MULTIPLE_CHOICE:
        scorer = MultipleChoiceScorer()
    elif effective_type == QuestionType.FILL_BLANK:
        scorer = FillBlankScorer()
    else:
        # OPEN_ENDED (varsayılan + bilinmeyen tipler)
        scorer = OpenEndedScorer(sbert_loader=sbert_loader)

    result = scorer.score(expected=expected, student=student, max_score=max_score)
    logger.info(
        "Skor: tip=%s, skor=%d/%d, benzerlik=%s",
        effective_type.value,
        result.score,
        max_score,
        f"{result.similarity:.2f}" if result.similarity is not None else "—",
    )
    return result
