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


# OCR prompt v3 (2026-05-22 evening fix) — sub-question + tip tespiti.
#
# Önceki v2 sürümü el yazısı + halüsinasyon önleme için iyiydi, ama:
#   - Soru TİPİ (open_ended / fill_blank / matching / multiple_choice) ayrımı yoktu
#   - SBERT her soruya uygulanıyordu — eşleştirme/çoktan seçmeli için yanlış araç
#
# v3 her soru/sub-question başına bir tip etiketi yazdırıyor. Parser bunu yakalıyor.
# Skorlama dispatcher (app/scoring/) tipe göre doğru scorer'a yönlendiriyor.
OCR_PROMPT = (
    "Bu görüntü bir Türkçe sınav kâğıdıdır. Sınav kâğıdı öğrenci tarafından doldurulmuş "
    "VEYA öğretmenin doğru cevap kâğıdı olabilir. Her durumda görevin: "
    "her sorunun cevabını ve TİPİNİ çıkarmaktır.\n\n"
    "FORMAT (kesinlikle bu — başka şekilde yazma):\n"
    "   **1)** [open_ended] cevap metni\n"
    "   **2a)** [fill_blank] cevap metni\n"
    "   **3)** [matching] a→4, b→2, c→9\n"
    "   **4)** [multiple_choice] C\n\n"
    "TİP TANIMLARI:\n"
    "- open_ended: açık uçlu klasik soru, cümle veya denklem cevabı\n"
    "- fill_blank: boşluk doldurma, tek kelime veya kısa ifade (örn. 'fiziksel')\n"
    "- matching: eşleştirme (a→4 gibi harf-sayı/kelime çiftleri)\n"
    "- multiple_choice: çoktan seçmeli, tek harf cevap (A/B/C/D/E)\n\n"
    "KURALLAR:\n"
    "1. Her soru için TEK BİR blok yaz. Bir blok içinde başka soru numarası ASLA tekrar etme.\n"
    "2. Tip etiketini köşeli parantez içinde yaz: [open_ended], [fill_blank] vb.\n"
    "3. Sadece kâğıttaki cevapları yaz. 'Yanlış cevap:', 'Not:', 'Çözüm:' gibi "
    "kendi yorumlarını EKLEME.\n"
    "4. Öğrenci ismi, okul, sınıf, marj notları gibi sınav dışı şeyleri YOK SAY.\n"
    "5. Tüm soruları sırayla işle, hiçbirini atlama.\n"
    "6. Cevap yoksa: '**N)** [open_ended] (boş)' yaz."
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
