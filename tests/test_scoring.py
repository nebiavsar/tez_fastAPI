"""Scoring strategy testleri (v3, 2026-05-23 — 3 strateji).

MatchingScorer kaldırıldı. 3 scorer:
    OpenEndedScorer       — SBERT semantic
    FillBlankScorer       — exact + Türkçe normalize + token kesişimi (eşleştirme dahil)
    MultipleChoiceScorer  — letter exact
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.schemas import QuestionType
from app.scoring import score_answer
from app.scoring.fill_blank import FillBlankScorer
from app.scoring.multiple_choice import MultipleChoiceScorer
from app.scoring.open_ended import OpenEndedScorer, similarity_to_score


# --- OPEN_ENDED (SBERT mock) ---


def test_open_ended_full_score_at_high_similarity() -> None:
    mock = MagicMock()
    mock.similarity.return_value = 0.95
    scorer = OpenEndedScorer(sbert_loader=mock)
    result = scorer.score(expected="ref", student="öğr", max_score=20)
    assert result.score == 20
    assert "Doğru" in result.feedback


def test_open_ended_zero_at_low_similarity() -> None:
    mock = MagicMock()
    mock.similarity.return_value = 0.20
    scorer = OpenEndedScorer(sbert_loader=mock)
    result = scorer.score(expected="ref", student="öğr", max_score=20)
    assert result.score == 0


def test_similarity_to_score_threshold() -> None:
    assert similarity_to_score(0.85, 10)[0] == 10
    assert similarity_to_score(0.50, 10)[0] in (2, 3, 4)
    assert similarity_to_score(0.30, 10)[0] == 0


# --- FILL_BLANK ---


def test_fill_blank_exact_match() -> None:
    scorer = FillBlankScorer()
    result = scorer.score(expected="fiziksel", student="fiziksel", max_score=4)
    assert result.score == 4


def test_fill_blank_case_insensitive() -> None:
    scorer = FillBlankScorer()
    result = scorer.score(expected="FİZİKSEL", student="fiziksel", max_score=4)
    assert result.score == 4


def test_fill_blank_turkish_normalize() -> None:
    scorer = FillBlankScorer()
    result = scorer.score(expected="ışık", student="isik", max_score=4)
    assert result.score == 4


def test_fill_blank_partial_token_match() -> None:
    scorer = FillBlankScorer()
    result = scorer.score(expected="kimyasal", student="kimyasal değişim", max_score=10)
    assert result.score == 7  # token kesişimi → %70 puan


def test_fill_blank_wrong() -> None:
    scorer = FillBlankScorer()
    result = scorer.score(expected="fiziksel", student="kimyasal", max_score=10)
    assert result.score == 0


def test_fill_blank_handles_matching_pairs() -> None:
    """Eşleştirme cevapları artık FillBlankScorer ile değerlendirilir.

    Aynı pair sırası → token kesişimi yüksek → tam veya yüksek puan.
    """
    scorer = FillBlankScorer()
    result = scorer.score(
        expected="a→4, b→2, c→9",
        student="a→4, b→2, c→9",
        max_score=10,
    )
    # Exact match veya çok yüksek partial match
    assert result.score >= 7


# --- MULTIPLE_CHOICE ---


def test_multiple_choice_correct() -> None:
    scorer = MultipleChoiceScorer()
    result = scorer.score(expected="C", student="C", max_score=5)
    assert result.score == 5


def test_multiple_choice_case_insensitive() -> None:
    scorer = MultipleChoiceScorer()
    result = scorer.score(expected="C", student="c", max_score=5)
    assert result.score == 5


def test_multiple_choice_wrong() -> None:
    scorer = MultipleChoiceScorer()
    result = scorer.score(expected="C", student="B", max_score=5)
    assert result.score == 0


def test_multiple_choice_letter_inside_sentence() -> None:
    scorer = MultipleChoiceScorer()
    result = scorer.score(expected="C", student="Cevap C", max_score=5)
    assert result.score == 5


# --- DISPATCHER (3 strateji) ---


def test_dispatcher_routes_open_ended_to_sbert() -> None:
    mock_sbert = MagicMock()
    mock_sbert.similarity.return_value = 0.90
    result = score_answer(
        question_type=QuestionType.OPEN_ENDED,
        expected="ref",
        student="öğr",
        max_score=20,
        sbert_loader=mock_sbert,
    )
    assert result.score == 20
    mock_sbert.similarity.assert_called_once()


def test_dispatcher_routes_fill_blank_without_sbert() -> None:
    mock_sbert = MagicMock()
    result = score_answer(
        question_type=QuestionType.FILL_BLANK,
        expected="fiziksel",
        student="fiziksel",
        max_score=4,
        sbert_loader=mock_sbert,
    )
    assert result.score == 4
    mock_sbert.similarity.assert_not_called()


def test_dispatcher_routes_multiple_choice_without_sbert() -> None:
    mock_sbert = MagicMock()
    result = score_answer(
        question_type=QuestionType.MULTIPLE_CHOICE,
        expected="C",
        student="C",
        max_score=5,
        sbert_loader=mock_sbert,
    )
    assert result.score == 5
    mock_sbert.similarity.assert_not_called()


def test_dispatcher_unknown_falls_back_to_open_ended() -> None:
    mock_sbert = MagicMock()
    mock_sbert.similarity.return_value = 0.90
    result = score_answer(
        question_type=QuestionType.UNKNOWN,
        expected="ref",
        student="öğr",
        max_score=20,
        sbert_loader=mock_sbert,
    )
    assert result.score == 20
    mock_sbert.similarity.assert_called_once()
