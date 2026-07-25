"""FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload --port 8000
Interactive API docs: http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import analyze, report
from .services.pipeline import MODELS_ARE_STUBS

app = FastAPI(
    title="AI Deepfake Protection API",
    version="0.1.0",
    description=(
        "Free, public deepfake analysis for images, audio and video.\n\n"
        "Results are probabilistic estimates, never proof. Uploaded files are "
        "deleted as soon as analysis finishes; only the derived result is kept, "
        "and only for 24 hours."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(report.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, object]:
    return {"status": "ok", "models_are_stubs": MODELS_ARE_STUBS}


@app.get("/limits", tags=["meta"])
def limits() -> dict[str, object]:
    """Upload limits, so the frontend can validate before sending 25MB uphill."""
    return {
        "image": {"max_bytes": settings.max_image_bytes},
        "audio": {
            "max_bytes": settings.max_audio_bytes,
            "max_seconds": settings.max_audio_seconds,
        },
        "video": {
            "max_bytes": settings.max_video_bytes,
            "max_seconds": settings.max_video_seconds,
        },
    }
