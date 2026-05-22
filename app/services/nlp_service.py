"""NLP semantik skorlama servisi — SBERT fine-tune ile öğrenci-referans karşılaştırma.

ADR (vault): [[NLPService]] — fine-tuned SBERT (model_ai_noted_with_negatives_positives_v2)
ile öğrenci cevabı ve öğretmen referans cevabı arasındaki kosinüs benzerliği.

Önceki placeholder (modül-seviye `SentenceTransformer("sentence_transformers_model_id")`
çağrısı import sırasında patlatıyordu) kaldırıldı. Model artık [[SBERTLoader]]
singleton'u üzerinden lifespan'da yüklenir.

Skor politikası — bkz [[Semantik Skorlama]]:
    benzerlik ≥ 0.85    → %100 puan, "Doğru."
    0.65 ≤ b < 0.85     → %60-80 puan, "Kısmen doğru, eksik kavramlar var."
    0.40 ≤ b < 0.65     → %20-40 puan, "Yakın ama yetersiz."
    b < 0.40            → %0 puan, "Yanlış / konuyla ilgisiz."
"""

from __future__ import annotations

import logging

from app.ml.sbert_loader import SBERTLoader
from app.schemas import (
    AnswerKeyEntry,
    ExamProcessingResponse,
    OCRExtractionResult,
    QuestionItem,
)

logger = logging.getLogger(__name__)


def _similarity_to_score(similarity: float, max_score: int) -> tuple[int, str]:
    """Kosinüs benzerliği → tam sayı puan + öğrenciye geri bildirim."""
    if similarity >= 0.85:
        return max_score, "Doğru."
    if similarity >= 0.65:
        ratio = 0.6 + 0.2 * (similarity - 0.65) / 0.20  # 0.65→%60, 0.85→%80
        return round(max_score * ratio), "Kısmen doğru, eksik kavramlar var."
    if similarity >= 0.40:
        ratio = 0.2 + 0.2 * (similarity - 0.40) / 0.25  # 0.40→%20, 0.65→%40
        return round(max_score * ratio), "Yakın ama yetersiz."
    return 0, "Yanlış veya konuyla ilgisiz."


class NLPService:
    """Öğrenci cevabı vs referans cevap semantik karşılaştırma."""

    def __init__(self, sbert_loader: SBERTLoader) -> None:
        self._loader = sbert_loader

    def evaluate(
        self,
        ocr_result: OCRExtractionResult,
        answer_key: list[AnswerKeyEntry] | None = None,
    ) -> ExamProcessingResponse:
        """OCR çıkışını referans cevap ile karşılaştırıp skor üret.

        `answer_key` None ise: skor hesaplanamaz, score=0 ve "Referans cevap yok" feedback'i döner.
        Bu, Spring Boot tarafının henüz answer_key payload'ı göndermediği geçiş döneminde
        OCR çıkışını test etmek için kullanışlı.
        """
        logger.info(
            "NLP evaluate başlıyor: %d OCR cevabı, answer_key=%s",
            len(ocr_result.detected_answers),
            "VAR" if answer_key else "YOK",
        )

        # answer_key'i question_number → entry sözlüğüne çevir
        key_by_number: dict[str, AnswerKeyEntry] = {}
        if answer_key:
            key_by_number = {entry.question_number: entry for entry in answer_key}

        questions: list[QuestionItem] = []
        total_score = 0

        for ocr_item in ocr_result.detected_answers:
            key_entry = key_by_number.get(ocr_item.question_number)

            if key_entry is None:
                # Referans yok — skor hesaplama, ham OCR çıkışını döndür
                questions.append(
                    QuestionItem(
                        questionId=ocr_item.question_number,
                        questionText=ocr_item.question_text,
                        extractedAnswer=ocr_item.extracted_answer,
                        expectedAnswer="",
                        score=0,
                        feedback="Referans cevap (answer_key) sağlanmadı.",
                    )
                )
                continue

            similarity = self._loader.similarity(
                reference=key_entry.expected_answer,
                student=ocr_item.extracted_answer,
            )
            score, feedback = _similarity_to_score(similarity, key_entry.max_score)
            total_score += score

            logger.info(
                "Soru %s: benzerlik=%.3f → skor=%d/%d",
                ocr_item.question_number,
                similarity,
                score,
                key_entry.max_score,
            )

            questions.append(
                QuestionItem(
                    questionId=ocr_item.question_number,
                    questionText=ocr_item.question_text,
                    extractedAnswer=ocr_item.extracted_answer,
                    expectedAnswer=key_entry.expected_answer,
                    score=score,
                    feedback=f"{feedback} (benzerlik: {similarity:.2f})",
                )
            )

        return ExamProcessingResponse(questions=questions, score=total_score)
