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


# OCR prompt v4 (2026-05-23) — tip etiketi atıldı.
#
# Eski v3 sürümünde modele "[open_ended], [fill_blank] gibi tip yaz" diyorduk.
# Canlı testte VLM bu kuralı tutmadı (tip etiketi yazmadı). Yerine artık
# **section başlıklarına** dayanan deterministik tip tespiti yapıyoruz
# (app/ml/ocr_parser._SECTION_PATTERNS).
#
# Bu prompt sadece **temiz cevap çıkarımı** ister: her soru için bir blok,
# halüsinasyon yok, sınav dışı şey yok. Section başlıkları sınav kâğıdında
# zaten yazılı olduğu için VLM doğal olarak onları da OCR çıkışına ekler.
OCR_PROMPT = (
    "Bu görüntü bir Türkçe sınav kâğıdıdır. Sınav kâğıdı öğrenci tarafından doldurulmuş "
    "VEYA öğretmenin doğru cevap kâğıdı olabilir.\n\n"
    "Görevin: her sorunun cevabını ve sınav yapısını çıkarmak.\n\n"
    "FORMAT (kesinlikle bu — başka şekilde yazma):\n"
    "   *çoktan seçmeli soru\n"
    "   **1)** cevap (10p)\n"
    "   *boşluk doldurma\n"
    "   **1)** cevap (5p)\n"
    "   **2)** cevap metni (10p)\n\n"
    "KURALLAR:\n"
    "1. **Yıldızlı (*) section başlıklarını AYNEN koru** — '*çoktan seçmeli soru', "
    "'*boşluk doldurma', '*eşleştirme', '*açık uçlu' gibi. Bu başlıklar sorunun "
    "skorlama tipini belirler.\n"
    "2. **Soruların yanındaki puan değerlerini (10p), (5p) AYNEN yaz** — final "
    "puanlamada kullanılır.\n"
    "3. Her soru için TEK BİR blok yaz. Bir blok içinde başka soru numarası tekrar etme.\n"
    "4. Çoktan seçmeli sorularda öğrenci genelde bir şıkkı yuvarlağa alır veya karalar. "
    "Bu durumda işaretli şıkkı tek harf olarak yaz: '**N)** C'. Eğer öğrenci hiç "
    "işaretlememişse '**N)** (boş)' yaz.\n"
    "5. Sadece kâğıttaki cevapları yaz. 'Yanlış cevap:', 'Not:', 'Çözüm:' gibi "
    "kendi yorumlarını EKLEME.\n"
    "6. Öğrenci ismi, okul, sınıf, marj notları gibi sınav dışı şeyleri YOK SAY.\n"
    "7. Tüm soruları sırayla işle, hiçbirini atlama."
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

        # Debug için ham OCR çıkışını log'a yaz — tip etiketleri var mı görmek için.
        # Production'da DEBUG seviyesine düşürülebilir; şimdilik teşhis için INFO.
        logger.info(
            "OCR ham çıkış (%s):\n--- BAŞ ---\n%s\n--- SON ---",
            filename,
            raw_text,
        )

        detected = parse_s3_markdown(raw_text)
        logger.info(
            "OCR tamamlandı: %s soru bloğu çıkarıldı | tipler: %s",
            len(detected),
            [f"{a.question_number}={a.question_type.value}" for a in detected],
        )

        return OCRExtractionResult(
            raw_text=raw_text,
            lines=raw_text.split("\n"),
            detected_answers=detected,
        )
