"""FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.ml.qwen_loader import get_qwen_loader
from app.ml.sbert_loader import get_sbert_loader


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama startup'ında ML modellerini bir kez yükle, shutdown'da temizle.

    Yüklenen modeller:
        - Qwen2.5-VL-7B (OCR) — cold start ~5-10 dk (ilk), ~30-60 sn (cache'li)
        - SBERT fine-tune (NLP) — CPU'da ~5-10 sn

    Her HTTP isteğinde yeniden yüklemek YASAK — singleton kullanılır.
    Donanım hatası olursa uygulama yine başlar (degraded mod), ama
    /process-exam çağrıları 500 döner.
    """
    logger = logging.getLogger(__name__)

    qwen = get_qwen_loader()
    sbert = get_sbert_loader()

    logger.info("Lifespan startup: Qwen2.5-VL yükleniyor...")
    try:
        qwen.load()
    except RuntimeError as exc:
        logger.error("Qwen yüklenemedi: %s — degraded modda devam", exc)

    logger.info("Lifespan startup: SBERT yükleniyor...")
    try:
        sbert.load()
    except RuntimeError as exc:
        logger.error("SBERT yüklenemedi: %s — degraded modda devam", exc)

    logger.info("Lifespan startup tamamlandı, sunucu hazır")

    yield

    logger.info("Lifespan shutdown: modeller unload ediliyor...")
    qwen.unload()
    sbert.unload()


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="FastAPI processing service used by Spring Boot for OCR exam evaluation.",
        lifespan=lifespan,
    )

    application.include_router(api_router)
    register_exception_handlers(application)
    return application


app = create_app()
