"""Scoring strategy pattern — her soru tipi için ayrı scorer.

Dispatcher (`score_answer`) tip bazlı yönlendirme yapar:
    OPEN_ENDED      → OpenEndedScorer (SBERT semantic)
    FILL_BLANK      → FillBlankScorer (exact match + Türkçe normalize)
    MATCHING        → MatchingScorer (pair-by-pair exact)
    MULTIPLE_CHOICE → MultipleChoiceScorer (letter exact)
"""

from app.scoring.dispatcher import ScoreResult, score_answer

__all__ = ["ScoreResult", "score_answer"]
