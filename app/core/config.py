"""Application configuration."""

from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "OCR Exam Processing Service"
    app_version: str = "0.1.0"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    allowed_image_content_types: set[str] = field(
        default_factory=lambda: {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp",
        }
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
