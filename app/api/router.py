"""Central API router."""

from fastapi import APIRouter

from app.api.routes.exam import router as exam_router

api_router = APIRouter()
api_router.include_router(exam_router)
