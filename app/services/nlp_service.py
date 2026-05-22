"""NLP skorlama servisi — tip bazlı strategy pattern üzerinden.

OCR çıkışındaki her cevap için question_type'a göre uygun scorer çağrılır:
    OPEN_ENDED      → SBERT semantic
    FILL_BLANK      → exact + Türkçe normalize
    MATCHING        → pair-by-pair exact
    MULTIPLE_CHOICE → letter exact

Önceki sürümlerden değişiklik:
- AnswerKeyEntry yerine OCRDetectedAnswer'lar (cevap kâğıdı OCR'ından gelir)
- Tip bazlı dispatcher ile her soruya doğru skorlama
- SBERT artık tek skorlama yöntemi değil — sadece OPEN_ENDED'ler için
"""

from __future__ import annotations

import logging

from app.ml.sbert_loader import SBERTLoader
from app.schemas import (
    AnswerKeyEntry,
    ExamProcessingResponse,
    OCRDetectedAnswer,
    OCRExtractionResult,
    QuestionItem,
    QuestionType,
)
from app.scoring import score_answer

logger = logging.getLogger(__name__)


class NLPService:
    """Tip bazlı semantik/exact skorlama orkestratörü."""

    def __init__(self, sbert_loader: SBERTLoader) -> None:
        self._loader = sbert_loader

    def evaluate(
        self,
        student_ocr: OCRExtractionResult,
        answer_key: list[AnswerKeyEntry] | None = None,
    ) -> ExamProcessingResponse:
        """Öğrenci OCR çıkışını referans cevaplarla karşılaştırıp skorla.

        `answer_key` None ise: skor hesaplanamaz, score=0 + ham OCR cevabı.
        """
        logger.info(
            "NLP evaluate başlıyor: %d öğrenci cevabı, answer_key=%s",
            len(student_ocr.detected_answers),
            "VAR" if answer_key else "YOK",
        )

        key_by_number: dict[str, AnswerKeyEntry] = {}
        if answer_key:
            key_by_number = {entry.question_number: entry for entry in answer_key}

        questions: list[QuestionItem] = []
        total_score = 0

        for ocr_item in student_ocr.detected_answers:
            key_entry = key_by_number.get(ocr_item.question_number)

            if key_entry is None:
                questions.append(
                    QuestionItem(
                        questionId=ocr_item.question_number,
                        questionText=ocr_item.question_text,
                        extractedAnswer=ocr_item.extracted_answer,
                        expectedAnswer="",
                        score=0,
                        feedback="Referans cevap (answer_key) bu soru için bulunamadı.",
                    )
                )
                continue

            # Soru tipi: öğrenci OCR'ından gelir; UNKNOWN ise answer_key'inkini dene
            effective_type = ocr_item.question_type
            if effective_type == QuestionType.UNKNOWN:
                effective_type = key_entry.question_type

            result = score_answer(
                question_type=effective_type,
                expected=key_entry.expected_answer,
                student=ocr_item.extracted_answer,
                max_score=key_entry.max_score,
                sbert_loader=self._loader,
            )
            total_score += result.score

            logger.info(
                "Soru %s (%s): skor=%d/%d",
                ocr_item.question_number,
                effective_type.value,
                result.score,
                key_entry.max_score,
            )

            questions.append(
                QuestionItem(
                    questionId=ocr_item.question_number,
                    questionText=ocr_item.question_text,
                    extractedAnswer=ocr_item.extracted_answer,
                    expectedAnswer=key_entry.expected_answer,
                    score=result.score,
                    feedback=f"[{effective_type.value}] {result.feedback}",
                )
            )

        return ExamProcessingResponse(questions=questions, score=total_score)

    @staticmethod
    def answer_key_from_ocr(ocr_result: OCRExtractionResult, default_max_score: int = 10) -> list[AnswerKeyEntry]:
        """Cevap kâğıdı OCR çıkışını AnswerKeyEntry listesine dönüştür.

        Cevap kâğıdı görseli OCR'dan geçince OCRDetectedAnswer listesi alırız.
        Bunu NLPService için AnswerKeyEntry formatına çeviriyoruz.

        max_score şu an default — gelecekte cevap kâğıdında puan etiketi
        ([5p], [10p] gibi) OCR ile çıkarılabilir.
        """
        return [
            AnswerKeyEntry(
                question_number=item.question_number,
                expected_answer=item.extracted_answer,
                question_type=item.question_type if item.question_type != QuestionType.UNKNOWN else QuestionType.OPEN_ENDED,
                max_score=default_max_score,
            )
            for item in ocr_result.detected_answers
        ]
