"""FastAPI application entry point.

Run locally:
    uvicorn app.main:app --reload --port 8000
Interactive API docs: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .bodylimit import BodySizeLimitMiddleware
from .config import settings
from .ratelimit import RateLimitMiddleware
from .routers import analyze, report
from .services.pipeline import REAL_MODELS
from .storage import store

logger = logging.getLogger(__name__)


async def _purge_expired_results() -> None:
    """Drop analysis results past their TTL, forever, on an interval.

    Expiry was previously checked only when a job was looked up, which collects
    nothing in the normal access pattern: a client polls until the result is
    ready, reads it, and never asks again. Nothing then triggers the check, so
    every analysis retained its record — and its InternalScores — for the
    lifetime of the process.

    Runs in the app's event loop rather than a thread because it is a dictionary
    scan under a lock every few minutes, not real work. It moves to the worker
    alongside the job store when that becomes Redis.
    """
    while True:
        await asyncio.sleep(settings.purge_interval_seconds)
        try:
            dropped = store.purge_expired()
            if dropped:
                logger.info("Purged %d expired analysis result(s)", dropped)
        except Exception:  # noqa: BLE001 - a sweep failure must not kill the loop
            logger.exception("Expiry sweep failed; will retry next interval")


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    task = asyncio.create_task(_purge_expired_results())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    lifespan=lifespan,
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
# CORS must therefore be added last, so that it wraps the other two and still
# attaches headers to a 429 or 413 they return without calling through. Get this
# backwards and a rate-limited or oversized browser upload sees an opaque CORS
# failure instead of the message explaining what happened.
# Asserted by test_rate_limited_response_still_has_cors_headers and
# test_oversize_body_rejection_has_cors_headers.
#
# The body ceiling is innermost of the three, so the two cheap checks run first.
# All middleware still runs before routing, and therefore before FastAPI parses
# the multipart body — which is the only thing that matters here, since parsing
# is what writes the body to disk.
app.add_middleware(BodySizeLimitMiddleware)

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
