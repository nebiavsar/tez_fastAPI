"""SBERT (Sentence-BERT) yükleyici — semantik benzerlik için fine-tuned model.

Singleton — Qwen ile aynı pattern, lifespan startup'ta yüklenir.

Model: `tez_fastAPI/model_ai_noted_with_negatives_positives_v2/`
    - Kullanıcının manuel etiketli veri + pozitif/negatif çiftlerle fine-tune ettiği SBERT.
    - Detay: vault'taki [[SBERT Modeli]] sayfası.

CPU'da çalışır — SBERT inference 384-dim embedding üretir, GPU şart değil.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model yolu — tez_fastAPI/ kökünden göreli.
# Path(__file__).resolve().parents[2] = tez_fastAPI/
MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "model_ai_noted_with_negatives_positives_v2"
)


class SBERTLoader:
    """SBERT singleton — startup'ta yüklen, encode/similarity ile kullan."""

    def __init__(self) -> None:
        self._model: "SentenceTransformer | None" = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Modeli RAM'e yükle. CPU'da ~5-10 sn."""
        if self.is_loaded:
            logger.info("SBERT zaten yüklü, atlanıyor")
            return

        if not MODEL_DIR.exists():
            raise RuntimeError(
                f"SBERT model klasörü bulunamadı: {MODEL_DIR}. "
                "tez_fastAPI/model_ai_noted_with_negatives_positives_v2/ var mı?"
            )

        # Lazy import
        from sentence_transformers import SentenceTransformer

        logger.info("SBERT yükleniyor: %s", MODEL_DIR)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(str(MODEL_DIR))
        elapsed = time.perf_counter() - t0
        logger.info("SBERT yüklendi (%.1f sn)", elapsed)

    def unload(self) -> None:
        self._model = None
        logger.info("SBERT unload edildi")

    def similarity(self, reference: str, student: str) -> float:
        """İki cümle arasındaki kosinüs benzerliği (0..1)."""
        if not self.is_loaded:
            raise RuntimeError(
                "SBERTLoader.load() çağrılmadı. Lifespan event'inde startup'ta yüklenmeli."
            )

        embeddings = self._model.encode([reference, student], convert_to_tensor=True)
        # SentenceTransformer.similarity → 2x2 matris; [0][1] = referans ↔ öğrenci
        sim_matrix = self._model.similarity(embeddings, embeddings)
        score = float(sim_matrix[0][1].item())
        # Çok küçük negatif değerleri (numerik gürültü) 0'a çek
        return max(0.0, min(1.0, score))


@lru_cache(maxsize=1)
def get_sbert_loader() -> SBERTLoader:
    return SBERTLoader()
