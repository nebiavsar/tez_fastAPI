"""Schemas for exam processing."""

from pydantic import BaseModel, Field


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


class OCRExtractionResult(BaseModel):
    raw_text: str
    lines: list[str] = Field(default_factory=list)
    detected_answers: list[OCRDetectedAnswer] = Field(default_factory=list)


class AnswerKeyEntry(BaseModel):
    """Bir soru için öğretmenin referans cevabı + tam puanı.

    Spring Boot tarafından `/process-exam` multipart payload'una eklenir.
    Kaynak: GroupAnswerKey entity'sinin parse edilmiş hâli.
    """

    question_number: str
    expected_answer: str
    max_score: int = 10
