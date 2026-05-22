"""Schemas for exam processing."""

from enum import Enum

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Soru tipleri — VLM tarafından otomatik tespit edilir, skorlama metodunu belirler.

    - OPEN_ENDED: cümle-cümle klasik soru, SBERT semantic skorlama
    - FILL_BLANK: boşluk doldurma, tek kelime / kısa ifade, exact match (Türkçe normalize)
    - MATCHING: eşleştirme (a→4, b→2), pair-by-pair exact match
    - MULTIPLE_CHOICE: çoktan seçmeli (A/B/C/D), exact letter match
    - UNKNOWN: VLM tip atayamadı (fallback olarak OPEN_ENDED gibi davranılır)
    """

    OPEN_ENDED = "open_ended"
    FILL_BLANK = "fill_blank"
    MATCHING = "matching"
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
