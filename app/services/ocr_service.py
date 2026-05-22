"""OCR servisi — Qwen2.5-VL-7B üzerinden el yazısı sınav okuma.

ADR-005: Qwen2.5-VL-7B-Instruct (4-bit quantize), Colab T4 GPU veya 8+ GB VRAM lokal.
Mimari:
    UploadFile bytes → PIL.Image → Qwen S3 prompt → markdown → parse → OCRExtractionResult

Önceki placeholder (sabit demo cevaplar döndüren) kaldırıldı.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

from app.ml.ocr_parser import parse_s3_markdown
from app.ml.qwen_loader import QwenVLLoader
from app.schemas import OCRExtractionResult

logger = logging.getLogger(__name__)


# S3 prompt — spike_qwen_vl.py'da en sağlam sonuç veren prompt.
# Öğrencinin el yazısı cevaplarını soru numarası ile eşleştirip listeler;
# header/footer/marj notlarını eler.
OCR_PROMPT = (
    "Bu görüntüde basılı (matbu) sorulara öğrenci tarafından elle yazılmış cevaplar var. "
    "Sadece öğrencinin el yazısıyla yazdığı cevapları soru numarasıyla eşleştirip listele. "
    'Format: "**N)** cevap metni" şeklinde her satır ayrı. '
    "Öğrenci ismi, okul, marj notları gibi şeyleri yazma."
)


class OCRService:
    """Görüntüden öğrenci cevaplarını çıkaran servis."""

    def __init__(self, qwen_loader: QwenVLLoader) -> None:
        self._loader = qwen_loader

    def extract(self, file_bytes: bytes, filename: str) -> OCRExtractionResult:
        """Sınav görüntüsünden cevapları çıkar."""
        logger.info("OCR başlıyor: %s (%d bayt)", filename, len(file_bytes))

        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as exc:
            logger.exception("Görüntü açılamadı: %s", filename)
            raise

        logger.info("Görüntü boyutu: %sx%s", image.size[0], image.size[1])

        raw_text = self._loader.generate(
            image=image,
            prompt=OCR_PROMPT,
            max_new_tokens=768,
        )

        detected = parse_s3_markdown(raw_text)
        logger.info(
            "OCR tamamlandı: %s soru bloğu çıkarıldı",
            len(detected),
        )

        return OCRExtractionResult(
            raw_text=raw_text,
            lines=raw_text.split("\n"),
            detected_answers=detected,
        )
