"""Exam orchestration service."""

import json
import logging

from fastapi import UploadFile, status
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import FileValidationError, ProcessingError, UpstreamServiceError
from app.schemas import AnswerKeyEntry, ExamProcessingResponse
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
    ) -> None:
        self.settings = settings
        self.ocr_service = ocr_service
        self.nlp_service = nlp_service

    async def process_exam(
        self,
        image: UploadFile | None,
        answer_key_json: str | None = None,
    ) -> ExamProcessingResponse:
        file_bytes, filename = await self._read_and_validate_file(image)
        logger.info("Starting exam processing for '%s'", filename)

        answer_key = self._parse_answer_key(answer_key_json)

        try:
            ocr_result = self.ocr_service.extract(file_bytes=file_bytes, filename=filename)
        except Exception as exc:  # pragma: no cover - reserved for future integration failures
            logger.exception("OCR processing failed for '%s'", filename, exc_info=exc)
            raise UpstreamServiceError("OCR processing failed.", error="ocr_processing_failed") from exc

        try:
            evaluation = self.nlp_service.evaluate(ocr_result, answer_key=answer_key)
        except Exception as exc:  # pragma: no cover - reserved for future integration failures
            logger.exception("NLP processing failed for '%s'", filename, exc_info=exc)
            raise UpstreamServiceError("NLP processing failed.", error="nlp_processing_failed") from exc

        logger.info(
            "Completed exam processing for '%s' with total score %s",
            filename,
            evaluation.score,
        )
        return evaluation

    @staticmethod
    def _parse_answer_key(answer_key_json: str | None) -> list[AnswerKeyEntry] | None:
        """Spring Boot'tan multipart text part olarak gelen JSON'ı parse et."""
        if not answer_key_json:
            return None
        try:
            raw = json.loads(answer_key_json)
            if not isinstance(raw, list):
                raise FileValidationError(
                    "answer_key JSON dizi (list) olmalı.",
                    error="invalid_answer_key",
                )
            return [AnswerKeyEntry.model_validate(item) for item in raw]
        except (json.JSONDecodeError, ValidationError) as exc:
            raise FileValidationError(
                f"answer_key parse edilemedi: {exc}",
                error="invalid_answer_key",
            ) from exc

    async def _read_and_validate_file(self, image: UploadFile | None) -> tuple[bytes, str]:
        if image is None:
            raise FileValidationError("Image file is required.", error="missing_file")

        filename = image.filename or "uploaded-image"
        content_type = (image.content_type or "").lower()

        if content_type not in self.settings.allowed_image_content_types:
            raise FileValidationError(
                "Unsupported file type. Allowed types: image/jpeg, image/jpg, image/png, image/webp.",
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                error="unsupported_file_type",
            )

        try:
            file_bytes = await image.read()
        except Exception as exc:
            logger.exception("Failed to read uploaded file '%s'", filename, exc_info=exc)
            raise ProcessingError("Failed to read uploaded file.", error="file_read_error") from exc
        finally:
            await image.close()

        if not file_bytes:
            raise FileValidationError("Uploaded image is empty.", error="empty_file")

        if len(file_bytes) > self.settings.max_upload_size_bytes:
            raise FileValidationError(
                "Uploaded image exceeds the maximum allowed size.",
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                error="file_too_large",
            )

        return file_bytes, filename
