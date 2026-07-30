"""Analysis endpoints.

All three POST endpoints follow the same shape: validate, stage the upload under
a size cap, run the models, delete the file, return 202 with an id to poll.

Why 202 + polling rather than returning the result inline: once the real models
land, a 60-second video runs three networks plus a fusion step. That does not
fit inside a request the client should hold open, and the frontend would have to
be rebuilt around it later. The stub already runs through the same async path so
nothing above it changes when inference gets slow.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from ..errors import UndecodableUpload, UploadTooLarge
from ..schemas import (
    AnalysisAccepted,
    AnalysisStatus,
    AnalysisStatusResponse,
    MediaKind,
)
from ..services.models.base import ModelUnavailable
from ..services.pipeline import run_analysis
from ..storage import store
from ..uploads import staged_upload

router = APIRouter(prefix="/analyze", tags=["analyze"])


def _handle(request: Request, kind: MediaKind, upload: UploadFile) -> AnalysisAccepted:
    job = store.create(media_kind=kind, filename=upload.filename)

    # The staged file is deleted when this context manager exits, on every path.
    # Analysis runs inline for now — the stubs are fast enough that faking a
    # queue would add moving parts without adding realism. The response contract
    # is already the async one, so step 3 moves this body into a worker without
    # the client noticing.
    try:
        with staged_upload(kind, upload, request) as staged:
            job.status = AnalysisStatus.RUNNING
            store.update(job)
            result, internal = run_analysis(kind, staged.path, has_audio=staged.has_audio)
    except HTTPException:
        store.delete(job.analysis_id)
        raise
    except ModelUnavailable as exc:
        # The server is misconfigured — weights missing. Not the caller's fault,
        # and explicitly not a reason to fall back to a placeholder verdict.
        store.delete(job.analysis_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from None
    except UploadTooLarge as exc:
        # Valid media, but more than this service will spend memory decoding —
        # a decompression bomb, or simply a very large photograph.
        store.delete(job.analysis_id)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from None
    except UndecodableUpload as exc:
        # Decoding failed: the declared content type and the actual bytes
        # disagree. That is a bad upload, so 400 rather than a 500.
        #
        # Matched on this type specifically, not on ValueError. The broad version
        # reflected str(exc) from *any* ValueError — including ones raised deep
        # inside PIL, numpy or torch, whose text can carry filesystem paths — into
        # a 400 body, and let an internal invariant failure (for example
        # VideoAuthenticator.fuse receiving no signals) present as a user error.
        store.delete(job.analysis_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None
    except Exception:
        job.status = AnalysisStatus.FAILED
        job.error = "Analysis failed while processing this file."
        store.update(job)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed while processing this file.",
        ) from None

    result.analysis_id = job.analysis_id
    job.result = result
    job.internal = internal
    job.status = AnalysisStatus.COMPLETE
    store.update(job)

    return AnalysisAccepted(
        analysis_id=job.analysis_id,
        status=job.status,
        poll_url=str(request.url_for("get_analysis", analysis_id=job.analysis_id)),
    )


@router.post(
    "/image",
    response_model=AnalysisAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyse an image (max 5MB)",
)
def analyze_image(request: Request, file: UploadFile = File(...)) -> AnalysisAccepted:
    """Routing: the Image model's output is the final verdict, with no fusion step."""
    return _handle(request, MediaKind.IMAGE, file)


@router.post(
    "/audio",
    response_model=AnalysisAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyse an audio file (max 10MB / ~2 min)",
)
def analyze_audio(request: Request, file: UploadFile = File(...)) -> AnalysisAccepted:
    """Routing: the Audio model's output is the final verdict, with no fusion step."""
    return _handle(request, MediaKind.AUDIO, file)


@router.post(
    "/video",
    response_model=AnalysisAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyse a video (max 25MB / ~60s)",
)
def analyze_video(request: Request, file: UploadFile = File(...)) -> AnalysisAccepted:
    """Routing: Image + Audio + Raw Frames, fused by the Video Authenticator."""
    return _handle(request, MediaKind.VIDEO, file)


@router.get(
    "/{analysis_id}",
    response_model=AnalysisStatusResponse,
    name="get_analysis",
    summary="Poll an analysis by id",
)
def get_analysis(analysis_id: str) -> AnalysisStatusResponse:
    job = store.get(analysis_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis with that id. Results expire after 24 hours.",
        )
    # job.internal is intentionally not included — see schemas.py.
    return AnalysisStatusResponse(
        analysis_id=job.analysis_id,
        status=job.status,
        result=job.result,
        error=job.error,
    )
