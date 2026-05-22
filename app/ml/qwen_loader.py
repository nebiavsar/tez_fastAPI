"""Qwen2.5-VL-7B-Instruct yükleyici.

Singleton — uygulama başlangıcında lifespan event ile bir kez yüklenir.
Her HTTP isteğinde aynı model/processor referansı kullanılır.

ADR-005: 4-bit quantize, T4 16 GB veya 8+ GB lokal VRAM gerekir.
"""

from __future__ import annotations

import io
import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# Image token sınırı — VRAM tutumlu kalmak için.
# 1280 patch = ~1003520 px, T4 16 GB için rahat.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# Generation parametreleri — spike_qwen_vl.py'da degeneration'ı önleyen ayarlar.
GENERATION_KWARGS = {
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 4,
    "do_sample": True,
    "temperature": 0.1,
    "top_p": 0.9,
}


class QwenVLLoader:
    """Qwen2.5-VL-7B singleton — startup'ta yüklen, generate() ile kullan."""

    def __init__(self) -> None:
        self._model: "Qwen2_5_VLForConditionalGeneration | None" = None
        self._processor: "AutoProcessor | None" = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        """Modeli VRAM'e yükle. Cold start ~30-60 sn (cache'liyse) / ~10 dk (ilk indirme)."""
        if self.is_loaded:
            logger.info("Qwen model zaten yüklü, atlanıyor")
            return

        # Lazy import — torch/transformers import'u uzun, ML kullanılmayan testlerde gereksiz.
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA yok — Qwen2.5-VL-7B CPU'da çalıştırılamaz. "
                "Colab T4 veya 8+ GB VRAM lokal GPU gerekir."
            )

        logger.info("Qwen2.5-VL-7B yükleniyor (4-bit nf4 quantize)...")
        t0 = time.perf_counter()

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            device_map="cuda:0",
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )

        elapsed = time.perf_counter() - t0
        vram_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info(
            "Qwen yüklendi (%.1f sn, VRAM: %.2f GB)",
            elapsed,
            vram_gb,
        )

    def unload(self) -> None:
        """VRAM'i temizle (shutdown veya manuel reset için)."""
        if not self.is_loaded:
            return
        import torch

        self._model = None
        self._processor = None
        torch.cuda.empty_cache()
        logger.info("Qwen unload edildi, VRAM temizlendi")

    def generate(
        self,
        image: "PILImage",
        prompt: str,
        max_new_tokens: int = 768,
    ) -> str:
        """Görüntü + prompt ile model'i çağır, decode edilmiş metni döndür."""
        if not self.is_loaded:
            raise RuntimeError(
                "QwenVLLoader.load() çağrılmadı. "
                "FastAPI lifespan event'inde startup'ta yüklenmeli."
            )

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        n_input = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **GENERATION_KWARGS,
            )
        elapsed = time.perf_counter() - t0
        n_gen = outputs.shape[-1] - n_input
        logger.info(
            "Qwen generate: %.1f sn, %d girdi token, %d üretilen token",
            elapsed,
            n_input,
            n_gen,
        )

        decoded = self._processor.batch_decode(
            outputs[:, n_input:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return decoded


@lru_cache(maxsize=1)
def get_qwen_loader() -> QwenVLLoader:
    """FastAPI DI için singleton accessor."""
    return QwenVLLoader()
