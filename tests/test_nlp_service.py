"""NLPService skorlama mantığı testleri.

SBERT modeli mock'lanır — gerçek model GPU/RAM gerektirir, burada sadece
similarity → score politikasını ve answer_key eşleştirme mantığını doğruluyoruz.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.schemas import AnswerKeyEntry, OCRDetectedAnswer, OCRExtractionResult
from app.services.nlp_service import NLPService, _similarity_to_score


# --- Skor eşiği testleri (saf fonksiyon) ---


def test_similarity_full_score() -> None:
    score, feedback = _similarity_to_score(0.95, max_score=10)
    assert score == 10
    assert "Doğru" in feedback


def test_similarity_partial_high() -> None:
    score, feedback = _similarity_to_score(0.80, max_score=10)
    assert 6 <= score <= 8
    assert "Kısmen" in feedback


def test_similarity_partial_low() -> None:
    score, feedback = _similarity_to_score(0.50, max_score=10)
    assert 2 <= score <= 4
    assert "Yakın" in feedback or "yetersiz" in feedback


def test_similarity_zero() -> None:
    score, feedback = _similarity_to_score(0.20, max_score=10)
    assert score == 0
    assert "Yanlış" in feedback or "ilgisiz" in feedback


def test_similarity_threshold_exact_boundary() -> None:
    # 0.85 → tam puan eşiği
    score, _ = _similarity_to_score(0.85, max_score=10)
    assert score == 10


# --- NLPService.evaluate testleri (mock SBERT) ---


def _ocr_fixture() -> OCRExtractionResult:
    return OCRExtractionResult(
        raw_text="",
        lines=[],
        detected_answers=[
            OCRDetectedAnswer(question_number="1", question_text="", extracted_answer="su buharlaşır"),
            OCRDetectedAnswer(question_number="2", question_text="", extracted_answer="kağıt yanar"),
        ],
    )


def test_evaluate_without_answer_key_returns_zero_scores() -> None:
    """answer_key None ise skor 0, ham OCR ile geri dönüş."""
    mock_loader = MagicMock()
    service = NLPService(sbert_loader=mock_loader)

    result = service.evaluate(_ocr_fixture(), answer_key=None)

    assert len(result.questions) == 2
    assert all(q.score == 0 for q in result.questions)
    assert "Referans cevap" in result.questions[0].feedback
    assert result.score == 0
    # SBERT çağrılmadı
    mock_loader.similarity.assert_not_called()


def test_evaluate_with_high_similarity_gives_full_score() -> None:
    mock_loader = MagicMock()
    mock_loader.similarity.return_value = 0.92  # tam puan eşiğinin üstü

    service = NLPService(sbert_loader=mock_loader)
    answer_key = [
        AnswerKeyEntry(question_number="1", expected_answer="su buharlaşma örneği", max_score=10),
        AnswerKeyEntry(question_number="2", expected_answer="kağıt yanma örneği", max_score=10),
    ]

    result = service.evaluate(_ocr_fixture(), answer_key=answer_key)

    assert len(result.questions) == 2
    assert all(q.score == 10 for q in result.questions)
    assert result.score == 20


def test_evaluate_missing_answer_key_entry_gets_zero() -> None:
    """OCR'da 1 ve 2 var ama answer_key'de sadece 1 → 2. soru skor 0."""
    mock_loader = MagicMock()
    mock_loader.similarity.return_value = 0.95

    service = NLPService(sbert_loader=mock_loader)
    answer_key = [
        AnswerKeyEntry(question_number="1", expected_answer="ref 1", max_score=10),
    ]

    result = service.evaluate(_ocr_fixture(), answer_key=answer_key)

    by_id = {q.questionId: q for q in result.questions}
    assert by_id["1"].score == 10
    assert by_id["2"].score == 0
    assert "Referans cevap" in by_id["2"].feedback


def test_evaluate_respects_max_score_per_question() -> None:
    mock_loader = MagicMock()
    mock_loader.similarity.return_value = 0.95

    service = NLPService(sbert_loader=mock_loader)
    answer_key = [
        AnswerKeyEntry(question_number="1", expected_answer="x", max_score=20),
        AnswerKeyEntry(question_number="2", expected_answer="y", max_score=5),
    ]

    result = service.evaluate(_ocr_fixture(), answer_key=answer_key)

    by_id = {q.questionId: q for q in result.questions}
    assert by_id["1"].score == 20
    assert by_id["2"].score == 5
    assert result.score == 25
