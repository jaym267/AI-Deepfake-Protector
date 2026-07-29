"""FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload --port 8000
Interactive API docs: http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .ratelimit import RateLimitMiddleware
from .routers import analyze, report
from .services.pipeline import REAL_MODELS

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

# Order matters, and it is the reverse of how it reads. `add_middleware`
# inserts at the front of the stack, so the LAST one added is the OUTERMOST.
# CORS must therefore be added last, so that it wraps the rate limiter and
# still attaches headers to a 429 the limiter returns without calling through.
# Get this backwards and a rate-limited browser client sees an opaque CORS
# failure instead of the "please wait N seconds" message.
# Asserted by test_rate_limited_response_still_has_cors_headers.
app.add_middleware(RateLimitMiddleware)

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
    """Liveness plus which detectors are real.

    ``models_are_stubs`` is kept for the frontend, which uses it only to decide
    whether to warn that nothing here is a genuine result yet. ``models`` is the
    per-detector truth, which is what anyone debugging actually needs between
    steps 3 and 6 while the four models land one at a time.
    """
    from .services.models.artifacts import load_probe
    from .services.models.image_model import ARTIFACT_NAME

    image_ready = load_probe(ARTIFACT_NAME) is not None
    models = dict(REAL_MODELS)
    # A real model whose weights are missing is not real on *this* server.
    models["image"] = models["image"] and image_ready

    return {
        "status": "ok",
        "models_are_stubs": not all(models.values()),
        "models": models,
    }


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
