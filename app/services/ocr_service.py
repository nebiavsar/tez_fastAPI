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


# OCR prompt v2 (2026-05-22 fix) — önceki sürümde 3 sorun vardı:
#   1) Soru 4-5 karışıklığı: model bir bloğa diğer sorunun cevabını dahil ediyordu
#   2) Halüsinasyon: model "Yanlış cevap:", "Not:" gibi kendi yorumlarını ekliyordu
#   3) Soru atlama: bazen son soruya hiç ulaşmıyordu (soru 5 kayboldu)
#
# Yeni prompt explicit kurallarla bu üçünü hedefliyor. Üç ana ilke:
#   - Tek-blok-tek-soru disiplini ("başka soru numarasını blok içinde tekrar etme")
#   - Sadece-öğrenci-yazısı disiplini ("kendi yorumunu/değerlendirmeni ekleme")
#   - Tam kapsam disiplini ("tüm soruları sırayla işle, atlama")
OCR_PROMPT = (
    "Bu görüntü bir öğrencinin çözdüğü Türkçe sınav kâğıdıdır. "
    "Görevin: ÖĞRENCİNİN EL YAZISIYLA yazdığı cevapları çıkarmak.\n\n"
    "KURALLAR:\n"
    "1. Her soru için TEK BİR blok yaz. Format kesinlikle şu olmalı:\n"
    "   **1)** [öğrencinin cevabı]\n"
    "   **2)** [öğrencinin cevabı]\n"
    "   ...\n"
    "2. Bir blok içinde başka soru numarası (3, 4 vb.) ASLA tekrar etme.\n"
    "3. Sadece öğrencinin yazdıklarını yaz. 'Yanlış cevap:', 'Not:', 'Çözüm:', "
    "'Doğru cevap:' gibi kendi yorumlarını veya değerlendirmelerini EKLEME.\n"
    "4. Öğrenci ismi, okul, sınıf, marj notları gibi sınav dışı şeyleri YOK SAY.\n"
    "5. Tüm soruları sırayla işle, hiçbirini atlama.\n"
    "6. Eğer bir sorunun cevabını öğrenci yazmamışsa: '**N)** (boş)' yaz."
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
