"""Routes for exam processing."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import Settings, get_settings
from app.ml.answer_key_cache import AnswerKeyCache, get_answer_key_cache
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
    answer_key_cache: Annotated[AnswerKeyCache, Depends(get_answer_key_cache)],
) -> ExamService:
    return ExamService(
        settings=settings,
        ocr_service=ocr_service,
        nlp_service=nlp_service,
        answer_key_cache=answer_key_cache,
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
    summary="Process an uploaded exam image with optional answer key image",
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
    paperImage: UploadFile = File(
        ...,
        description="Öğrenci tarafından çözülmüş sınav kâğıdı fotoğrafı",
    ),
    answerKeyImage: UploadFile | None = File(
        default=None,
        description=(
            "Öğretmenin doğru cevap kâğıdı fotoğrafı. "
            "Verilmezse NLP skorlama yapılmaz, sadece OCR çıkışı dönderilir. "
            "Aynı cevap kâğıdı tekrar gönderildiğinde cache'ten okunur."
        ),
    ),
) -> ExamProcessingResponse:
    logger.info(
        "Received /process-exam (paperImage='%s', answerKeyImage='%s')",
        paperImage.filename,
        answerKeyImage.filename if answerKeyImage else None,
    )
    response = await exam_service.process_exam(
        paper_image=paperImage,
        answer_key_image=answerKeyImage,
    )
    logger.info(
        "Returning /process-exam response: %d questions, total score=%d",
        len(response.questions),
        response.score,
    )
    return response
