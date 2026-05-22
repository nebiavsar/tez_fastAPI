"""Common response schemas."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")


class ErrorResponse(BaseModel):
    detail: str = Field(examples=["Unsupported file type."])
    error: str = Field(examples=["unsupported_file_type"])
