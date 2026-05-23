"""NLPService skorlama mantığı testleri (refactor sonrası, tip bazlı).

SBERT modeli mock'lanır — gerçek model GPU/RAM gerektirir. Burada test edilen:
- Tip bazlı dispatch (open_ended → SBERT, fill_blank → exact, vs.)
- answer_key eşleştirme mantığı (question_number bazlı)
- answer_key=None davranışı (skor=0, ham OCR çıkışı)
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.schemas import (
    AnswerKeyEntry,
    OCRDetectedAnswer,
    OCRExtractionResult,
    QuestionType,
)
from app.services.nlp_service import NLPService


def _ocr_fixture() -> OCRExtractionResult:
    return OCRExtractionResult(
        raw_text="",
        lines=[],
        detected_answers=[
            OCRDetectedAnswer(
                question_number="1",
                question_text="",
                extracted_answer="su buharlaşır",
                question_type=QuestionType.OPEN_ENDED,
            ),
            OCRDetectedAnswer(
                question_number="2",
                question_text="",
                extracted_answer="fiziksel",
                question_type=QuestionType.FILL_BLANK,
            ),
        ],
    )


def test_evaluate_without_answer_key_returns_zero_scores() -> None:
    mock = MagicMock()
    service = NLPService(sbert_loader=mock)
    result = service.evaluate(_ocr_fixture(), answer_key=None)

    assert len(result.questions) == 2
    assert all(q.score == 0 for q in result.questions)
    assert "Referans cevap" in result.questions[0].feedback
    assert result.score == 0
    mock.similarity.assert_not_called()


def test_evaluate_open_ended_uses_sbert() -> None:
    mock = MagicMock()
    mock.similarity.return_value = 0.90  # tam puan eşiğinin üstü

    service = NLPService(sbert_loader=mock)
    answer_key = [
        AnswerKeyEntry(
            question_number="1",
            expected_answer="su buharlaşma örneği",
            question_type=QuestionType.OPEN_ENDED,
            max_score=10,
        ),
    ]

    result = service.evaluate(_ocr_fixture(), answer_key=answer_key)

    # Soru 1: SBERT çağrıldı, tam puan
    q1 = next(q for q in result.questions if q.questionId == "1")
    assert q1.score == 10
    assert mock.similarity.called


def test_evaluate_fill_blank_does_not_call_sbert() -> None:
    """fill_blank exact match yapar, SBERT'i çağırmamalı."""
    mock = MagicMock()

    service = NLPService(sbert_loader=mock)
    answer_key = [
        AnswerKeyEntry(
            question_number="2",
            expected_answer="fiziksel",
            question_type=QuestionType.FILL_BLANK,
            max_score=4,
        ),
    ]

    result = service.evaluate(_ocr_fixture(), answer_key=answer_key)

    q2 = next(q for q in result.questions if q.questionId == "2")
    assert q2.score == 4  # tam eşleşme
    mock.similarity.assert_not_called()


def test_evaluate_missing_answer_key_entry_gets_zero() -> None:
    mock = MagicMock()
    mock.similarity.return_value = 0.95

    service = NLPService(sbert_loader=mock)
    answer_key = [
        AnswerKeyEntry(
            question_number="1",
            expected_answer="ref 1",
            question_type=QuestionType.OPEN_ENDED,
            max_score=10,
        ),
    ]
    # Sadece 1 var, 2 yok

    result = service.evaluate(_ocr_fixture(), answer_key=answer_key)

    by_id = {q.questionId: q for q in result.questions}
    assert by_id["1"].score == 10
    assert by_id["2"].score == 0
    assert "Referans cevap" in by_id["2"].feedback


def test_evaluate_respects_max_score_per_question() -> None:
    mock = MagicMock()
    mock.similarity.return_value = 0.95

    service = NLPService(sbert_loader=mock)
    answer_key = [
        AnswerKeyEntry(
            question_number="1",
            expected_answer="x",
            question_type=QuestionType.OPEN_ENDED,
            max_score=20,
        ),
        AnswerKeyEntry(
            question_number="2",
            expected_answer="fiziksel",
            question_type=QuestionType.FILL_BLANK,
            max_score=5,
        ),
    ]

    result = service.evaluate(_ocr_fixture(), answer_key=answer_key)

    by_id = {q.questionId: q for q in result.questions}
    assert by_id["1"].score == 20
    assert by_id["2"].score == 5
    assert result.score == 25


def test_answer_key_from_ocr_helper() -> None:
    """OCRExtractionResult → AnswerKeyEntry listesi dönüşümü.

    max_score parser'dan gelir (OCRDetectedAnswer.max_score field'ı).
    Parser '(Np)' bulduysa o değeri kullanır, yoksa default (10).
    """
    ocr = OCRExtractionResult(
        raw_text="",
        lines=[],
        detected_answers=[
            OCRDetectedAnswer(
                question_number="oe1",
                question_text="",
                extracted_answer="referans 1",
                question_type=QuestionType.OPEN_ENDED,
                max_score=15,  # Parser (15p) okuduysa
            ),
            OCRDetectedAnswer(
                question_number="fb1",
                question_text="",
                extracted_answer="fiziksel",
                question_type=QuestionType.FILL_BLANK,
                max_score=5,  # Parser (5p) okuduysa
            ),
        ],
    )
    keys = NLPService.answer_key_from_ocr(ocr)

    assert len(keys) == 2
    assert keys[0].question_number == "oe1"
    assert keys[0].expected_answer == "referans 1"
    assert keys[0].question_type == QuestionType.OPEN_ENDED
    assert keys[0].max_score == 15  # Parser'dan gelir
    assert keys[1].question_type == QuestionType.FILL_BLANK
    assert keys[1].max_score == 5


def test_answer_key_from_ocr_unknown_type_falls_back_to_open_ended() -> None:
    """OCR tip atayamadıysa AnswerKeyEntry OPEN_ENDED'a düşmeli."""
    ocr = OCRExtractionResult(
        raw_text="",
        lines=[],
        detected_answers=[
            OCRDetectedAnswer(
                question_number="1",
                question_text="",
                extracted_answer="ref",
                question_type=QuestionType.UNKNOWN,
            ),
        ],
    )
    keys = NLPService.answer_key_from_ocr(ocr)
    assert keys[0].question_type == QuestionType.OPEN_ENDED
