"""Application exceptions and handlers."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas import ErrorResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(self, detail: str, *, status_code: int, error: str) -> None:
        self.detail = detail
        self.status_code = status_code
        self.error = error
        super().__init__(detail)


class FileValidationError(AppError):
    def __init__(self, detail: str, *, status_code: int = status.HTTP_400_BAD_REQUEST, error: str) -> None:
        super().__init__(detail, status_code=status_code, error=error)


class UpstreamServiceError(AppError):
    def __init__(self, detail: str, *, error: str) -> None:
        super().__init__(
            detail,
            status_code=status.HTTP_502_BAD_GATEWAY,
            error=error,
        )


class ProcessingError(AppError):
    def __init__(self, detail: str = "Internal processing error.", *, error: str = "internal_error") -> None:
        super().__init__(
            detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error=error,
        )


def _error_response(status_code: int, detail: str, error: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(detail=detail, error=error).model_dump(),
    )


async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return _error_response(exc.status_code, exc.detail, exc.error)


async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return _error_response(exc.status_code, detail, "http_error")


async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    detail = first_error.get("msg", "Request validation failed.")
    return _error_response(status.HTTP_400_BAD_REQUEST, detail, "validation_error")


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application error", exc_info=exc)
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Internal processing error.",
        "internal_error",
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
