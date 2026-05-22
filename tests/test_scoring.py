"""Scoring strategy testleri — GPU/SBERT gerekmez (mock'lanır)."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.schemas import QuestionType
from app.scoring import score_answer
from app.scoring.fill_blank import FillBlankScorer
from app.scoring.matching import MatchingScorer
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
    assert "Yanlış" in result.feedback or "ilgisiz" in result.feedback


def test_similarity_to_score_threshold() -> None:
    assert similarity_to_score(0.85, 10)[0] == 10
    assert similarity_to_score(0.50, 10)[0] in (2, 3, 4)
    assert similarity_to_score(0.30, 10)[0] == 0


# --- FILL_BLANK ---


def test_fill_blank_exact_match() -> None:
    scorer = FillBlankScorer()
    result = scorer.score(expected="fiziksel", student="fiziksel", max_score=4)
    assert result.score == 4
    assert "Doğru" in result.feedback


def test_fill_blank_case_insensitive() -> None:
    scorer = FillBlankScorer()
    result = scorer.score(expected="FİZİKSEL", student="fiziksel", max_score=4)
    assert result.score == 4


def test_fill_blank_turkish_normalize() -> None:
    """ı/i, ş/s, ç/c, ğ/g, ö/o, ü/u normalizasyonu."""
    scorer = FillBlankScorer()
    # OCR "İ" yerine "I" okumuş olabilir
    result = scorer.score(expected="ışık", student="isik", max_score=4)
    # ı→i, ş→s, ı→i → eşleşmeli
    assert result.score == 4


def test_fill_blank_partial_token_match() -> None:
    scorer = FillBlankScorer()
    # öğrenci "kimyasal değişim" yazmış, referans "kimyasal" — anahtar kelime eşleşir
    result = scorer.score(expected="kimyasal", student="kimyasal değişim", max_score=10)
    # Tam eşleşme değil (fazla kelime), token kesişimi var → %70 puan
    assert result.score == 7
    assert "Kısmen" in result.feedback


def test_fill_blank_wrong() -> None:
    scorer = FillBlankScorer()
    result = scorer.score(expected="fiziksel", student="kimyasal", max_score=10)
    assert result.score == 0


# --- MATCHING ---


def test_matching_all_correct() -> None:
    scorer = MatchingScorer()
    expected = "a→4, b→2, c→9"
    student = "a→4, b→2, c→9"
    result = scorer.score(expected=expected, student=student, max_score=20)
    assert result.score == 20


def test_matching_half_correct() -> None:
    scorer = MatchingScorer()
    expected = "a→4, b→2, c→9, d→6"
    student = "a→4, b→3, c→9, d→7"  # 2/4 doğru
    result = scorer.score(expected=expected, student=student, max_score=20)
    assert result.score == 10
    assert "2/4" in result.feedback


def test_matching_different_format_styles() -> None:
    """a) 4, a-4, a:4, a→4 hepsi parse edilebilmeli."""
    scorer = MatchingScorer()
    expected = "a) 4, b) 2, c) 9"
    student = "a-4, b-2, c-9"  # farklı separator
    result = scorer.score(expected=expected, student=student, max_score=15)
    assert result.score == 15


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
    assert "Yanlış" in result.feedback


def test_multiple_choice_letter_inside_sentence() -> None:
    """OCR 'Cevap: C' yazmışsa yine yakalanmalı."""
    scorer = MultipleChoiceScorer()
    result = scorer.score(expected="C", student="Cevap C", max_score=5)
    assert result.score == 5


# --- DISPATCHER ---


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
    """FILL_BLANK SBERT'i çağırmamalı."""
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
