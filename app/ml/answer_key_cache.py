"""Cevap kâğıdı OCR sonucu için in-memory cache.

Aynı cevap kâğıdı görseli birden fazla öğrenci sınavı için tekrar gönderilir
(öğretmen sınıf başına bir cevap kâğıdı yükler). Her sınavda baştan OCR
yapmak yerine, hash bazlı cache ile aynı görsel ikinci kez geldiğinde
saklı sonucu döndürürüz.

- Key: SHA256(answer_key_image_bytes)
- Value: list[OCRDetectedAnswer]
- Storage: in-memory dict (uvicorn restart'ında sıfırlanır — gerçek
  production için Redis veya benzeri persistent store düşünülebilir)
"""

from __future__ import annotations

import hashlib
import logging
import threading
from functools import lru_cache

from app.schemas import OCRDetectedAnswer

logger = logging.getLogger(__name__)


class AnswerKeyCache:
    """Thread-safe in-memory cache."""

    def __init__(self) -> None:
        self._store: dict[str, list[OCRDetectedAnswer]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _hash(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def get(self, image_bytes: bytes) -> list[OCRDetectedAnswer] | None:
        key = self._hash(image_bytes)
        with self._lock:
            cached = self._store.get(key)
        if cached is not None:
            logger.info("AnswerKey cache HIT: %s... (%d soru)", key[:8], len(cached))
        else:
            logger.info("AnswerKey cache MISS: %s...", key[:8])
        return cached

    def put(self, image_bytes: bytes, answers: list[OCRDetectedAnswer]) -> None:
        key = self._hash(image_bytes)
        with self._lock:
            self._store[key] = answers
        logger.info("AnswerKey cache SET: %s... (%d soru)", key[:8], len(answers))

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


@lru_cache(maxsize=1)
def get_answer_key_cache() -> AnswerKeyCache:
    """FastAPI DI için singleton."""
    return AnswerKeyCache()
