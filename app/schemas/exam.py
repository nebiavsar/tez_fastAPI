"""Schemas for exam processing."""

from enum import Enum

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Soru tipleri — sınav kâğıdı section başlıklarından deterministik olarak tespit edilir.

    Tasarım kararı (2026-05-23): VLM otomatik tip tespiti kaldırıldı (güvenilmez sonuçlar
    veriyordu). Yerine, öğretmen sınav kâğıdı hazırlarken section başlıkları kullanır:
    - "Çoktan Seçmeli Sorular" → o bölgedeki tüm sorular MULTIPLE_CHOICE
    - "Boşluk Doldurma" / "Eşleştirme" → FILL_BLANK
    - Başlık yok → OPEN_ENDED (SBERT semantic)

    Tipler:
    - OPEN_ENDED: cümle/denklem cevap, SBERT semantic skorlama (DEFAULT)
    - FILL_BLANK: kısa cevap / eşleştirme, exact match + Türkçe normalize + token kesişimi
    - MULTIPLE_CHOICE: çoktan seçmeli (A/B/C/D), exact letter match
    - UNKNOWN: parser tip atayamadı → dispatcher OPEN_ENDED'a düşürür
    """

    OPEN_ENDED = "open_ended"
    FILL_BLANK = "fill_blank"
    MULTIPLE_CHOICE = "multiple_choice"
    UNKNOWN = "unknown"


class QuestionItem(BaseModel):
    questionId: str
    questionText: str
    extractedAnswer: str
    expectedAnswer: str
    score: int
    feedback: str


class ExamProcessingResponse(BaseModel):
    questions: list[QuestionItem] = Field(default_factory=list)
    score: int


class OCRDetectedAnswer(BaseModel):
    question_number: str
    question_text: str
    extracted_answer: str
    question_type: QuestionType = QuestionType.UNKNOWN


class OCRExtractionResult(BaseModel):
    raw_text: str
    lines: list[str] = Field(default_factory=list)
    detected_answers: list[OCRDetectedAnswer] = Field(default_factory=list)


class AnswerKeyEntry(BaseModel):
    """Bir soru için öğretmenin referans cevabı + tipi + tam puanı.

    Cevap kâğıdı görseli OCR'dan geçtikten sonra üretilir
    (artık doğrudan JSON olarak Spring Boot'tan gelmez).
    """

    question_number: str
    expected_answer: str
    question_type: QuestionType = QuestionType.OPEN_ENDED
    max_score: int = 10
