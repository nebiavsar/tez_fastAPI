"""Tip bazlı scorer dispatcher — `score_answer` ana giriş noktası.

Soru tipine bakar, doğru scorer'a yönlendirir.
"""

from __future__ import annotations

import logging

from app.ml.sbert_loader import SBERTLoader
from app.schemas import QuestionType
from app.scoring.base import ScoreResult
from app.scoring.fill_blank import FillBlankScorer
from app.scoring.matching import MatchingScorer
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
    # UNKNOWN → OPEN_ENDED fallback
    effective_type = question_type if question_type != QuestionType.UNKNOWN else QuestionType.OPEN_ENDED

    if effective_type == QuestionType.OPEN_ENDED:
        scorer = OpenEndedScorer(sbert_loader=sbert_loader)
    elif effective_type == QuestionType.FILL_BLANK:
        scorer = FillBlankScorer()
    elif effective_type == QuestionType.MATCHING:
        scorer = MatchingScorer()
    elif effective_type == QuestionType.MULTIPLE_CHOICE:
        scorer = MultipleChoiceScorer()
    else:
        # Edge case — yeni tip eklenirse OPEN_ENDED'a düş
        logger.warning("Bilinmeyen QuestionType '%s', OPEN_ENDED'a düşürülüyor", question_type)
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
