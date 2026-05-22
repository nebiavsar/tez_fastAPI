"""Exam orchestration service.

Akış:
1. paperImage (öğrenci) + answerKeyImage (öğretmen cevap kâğıdı) doğrula
2. answerKeyImage cache'te var mı? Yoksa OCR'dan geçir, cache'le.
3. paperImage OCR'dan geçir.
4. NLPService.evaluate ile tip-bazlı skorlama.
5. ExamProcessingResponse döndür.
"""

import logging

from fastapi import UploadFile, status

from app.core.config import Settings
from app.core.exceptions import FileValidationError, ProcessingError, UpstreamServiceError
from app.ml.answer_key_cache import AnswerKeyCache
from app.schemas import ExamProcessingResponse
from app.services.nlp_service import NLPService
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)


class ExamService:
    def __init__(
        self,
        *,
        settings: Settings,
        ocr_service: OCRService,
        nlp_service: NLPService,
        answer_key_cache: AnswerKeyCache,
    ) -> None:
        self.settings = settings
        self.ocr_service = ocr_service
        self.nlp_service = nlp_service
        self.answer_key_cache = answer_key_cache

    async def process_exam(
        self,
        paper_image: UploadFile | None,
        answer_key_image: UploadFile | None,
    ) -> ExamProcessingResponse:
        """İki görselden öğrenci sınavını değerlendir."""
        # 1. Öğrenci kâğıdı doğrula
        paper_bytes, paper_filename = await self._read_and_validate_file(
            paper_image, label="paperImage"
        )
        logger.info("Exam processing for student paper: '%s'", paper_filename)

        # 2. Cevap kâğıdı OCR (cache'li)
        answer_key_entries = None
        if answer_key_image is not None:
            ak_bytes, ak_filename = await self._read_and_validate_file(
                answer_key_image, label="answerKeyImage"
            )

            cached = self.answer_key_cache.get(ak_bytes)
            if cached is not None:
                ak_ocr_answers = cached
            else:
                logger.info("Answer key OCR başlıyor: '%s'", ak_filename)
                try:
                    ak_ocr_result = self.ocr_service.extract(
                        file_bytes=ak_bytes,
                        filename=ak_filename,
                    )
                except Exception as exc:
                    logger.exception("Answer key OCR failed", exc_info=exc)
                    raise UpstreamServiceError(
                        "Answer key OCR processing failed.",
                        error="answer_key_ocr_failed",
                    ) from exc
                ak_ocr_answers = ak_ocr_result.detected_answers
                self.answer_key_cache.put(ak_bytes, ak_ocr_answers)

            # AnswerKeyEntry listesine dönüştür
            from app.schemas import OCRExtractionResult

            ak_extraction = OCRExtractionResult(
                raw_text="",
                lines=[],
                detected_answers=ak_ocr_answers,
            )
            answer_key_entries = NLPService.answer_key_from_ocr(ak_extraction)
            logger.info("Answer key hazır: %d soru", len(answer_key_entries))

        # 3. Öğrenci kâğıdı OCR
        try:
            student_ocr = self.ocr_service.extract(
                file_bytes=paper_bytes,
                filename=paper_filename,
            )
        except Exception as exc:
            logger.exception("Student paper OCR failed", exc_info=exc)
            raise UpstreamServiceError(
                "Student paper OCR processing failed.",
                error="ocr_processing_failed",
            ) from exc

        # 4. NLP skorlama
        try:
            evaluation = self.nlp_service.evaluate(student_ocr, answer_key=answer_key_entries)
        except Exception as exc:
            logger.exception("NLP evaluation failed", exc_info=exc)
            raise UpstreamServiceError(
                "NLP processing failed.",
                error="nlp_processing_failed",
            ) from exc

        logger.info(
            "Exam processing tamamlandı: '%s', toplam skor: %s",
            paper_filename,
            evaluation.score,
        )
        return evaluation

    async def _read_and_validate_file(
        self,
        upload: UploadFile | None,
        *,
        label: str,
    ) -> tuple[bytes, str]:
        if upload is None:
            raise FileValidationError(
                f"{label} dosyası gerekli.",
                error=f"missing_{label}",
            )

        filename = upload.filename or f"{label}-image"
        content_type = (upload.content_type or "").lower()

        if content_type not in self.settings.allowed_image_content_types:
            raise FileValidationError(
                f"{label}: desteklenmeyen dosya tipi. "
                "İzinli tipler: image/jpeg, image/jpg, image/png, image/webp.",
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                error="unsupported_file_type",
            )

        try:
            file_bytes = await upload.read()
        except Exception as exc:
            logger.exception("Dosya okuma hatası: '%s'", filename, exc_info=exc)
            raise ProcessingError(
                f"{label} dosyası okunamadı.",
                error="file_read_error",
            ) from exc
        finally:
            await upload.close()

        if not file_bytes:
            raise FileValidationError(
                f"{label} boş.",
                error=f"empty_{label}",
            )

        if len(file_bytes) > self.settings.max_upload_size_bytes:
            raise FileValidationError(
                f"{label} boyut limitini aşıyor.",
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                error="file_too_large",
            )

        return file_bytes, filename
