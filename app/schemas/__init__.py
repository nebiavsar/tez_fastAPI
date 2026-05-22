"""Schema exports."""

from app.schemas.common import ErrorResponse, HealthResponse
from app.schemas.exam import (
    AnswerKeyEntry,
    ExamProcessingResponse,
    OCRDetectedAnswer,
    OCRExtractionResult,
    QuestionItem,
)

__all__ = [
    "AnswerKeyEntry",
    "ErrorResponse",
    "ExamProcessingResponse",
    "HealthResponse",
    "OCRDetectedAnswer",
    "OCRExtractionResult",
    "QuestionItem",
]
