"""Routes for exam processing."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.config import Settings, get_settings
from app.ml.qwen_loader import QwenVLLoader, get_qwen_loader
from app.ml.sbert_loader import SBERTLoader, get_sbert_loader
from app.schemas import ErrorResponse, ExamProcessingResponse, HealthResponse
from app.services.exam_service import ExamService
from app.services.nlp_service import NLPService
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["exam-processing"])


def get_ocr_service(
    qwen_loader: Annotated[QwenVLLoader, Depends(get_qwen_loader)],
) -> OCRService:
    return OCRService(qwen_loader=qwen_loader)


def get_nlp_service(
    sbert_loader: Annotated[SBERTLoader, Depends(get_sbert_loader)],
) -> NLPService:
    return NLPService(sbert_loader=sbert_loader)


def get_exam_service(
    settings: Annotated[Settings, Depends(get_settings)],
    ocr_service: Annotated[OCRService, Depends(get_ocr_service)],
    nlp_service: Annotated[NLPService, Depends(get_nlp_service)],
) -> ExamService:
    return ExamService(
        settings=settings,
        ocr_service=ocr_service,
        nlp_service=nlp_service,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def get_health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/process-exam",
    response_model=ExamProcessingResponse,
    summary="Process an uploaded exam image",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def process_exam(
    exam_service: Annotated[ExamService, Depends(get_exam_service)],
    image: UploadFile | None = File(default=None),
    answer_key: Annotated[
        str | None,
        Form(
            description=(
                "Opsiyonel: Öğretmenin doğru cevap kâğıdının JSON dizisi. "
                'Format: \'[{"question_number":"1","expected_answer":"...","max_score":10}, ...]\'. '
                "Verilmezse NLP skor hesaplanmaz, sadece OCR çıkışı dönderilir."
            ),
        ),
    ] = None,
) -> ExamProcessingResponse:
    logger.info("Received /process-exam request (answer_key=%s)", "var" if answer_key else "yok")
    response = await exam_service.process_exam(image=image, answer_key_json=answer_key)
    logger.info("Returning /process-exam response with %s questions", len(response.questions))
    return response
