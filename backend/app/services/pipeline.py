"""Analysis orchestration: routing, banding, and the public/internal split.

Routing follows the architecture exactly:

  VIDEO  -> Image + Audio + Raw Frames -> Video Authenticator -> final verdict
  IMAGE  -> Image model's output is the final verdict
  AUDIO  -> Audio model's output is the final verdict

The uploaded file is read only inside ``run_analysis``. By the time this function
returns, the caller deletes it (see uploads.staged_upload); nothing here retains
a path or a copy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..config import settings
from ..schemas import (
    AnalysisJob,
    AnalysisResult,
    AnalysisStatus,
    ConfidenceBand,
    InternalScores,
    MediaKind,
    Verdict,
)
from ..storage import store
from .disclaimer import RESULT_DISCLAIMER
from .models import AudioModel, ImageModel, RawFramesModel, VideoAuthenticator
from .models.base import DetectorOutput

image_model = ImageModel()
audio_model = AudioModel()
raw_frames_model = RawFramesModel()
video_authenticator = VideoAuthenticator()

#: Flip to False in step 6, once every model in the routing table is real.
MODELS_ARE_STUBS = True


def _band_verdict(score: float) -> Verdict:
    """Map P(manipulated) onto a verdict band.

    Four bands, never two. A binary real/fake badge would misrepresent what the
    models can actually support, and 'uncertain' has to be a visible outcome
    rather than being rounded into one of the confident answers.

    Thresholds are provisional and must be re-derived from validation-set ROC
    curves once real models exist (steps 3-6) — picking them by eye is exactly
    how a detector ends up confidently wrong.
    """
    if score < 0.25:
        return Verdict.LIKELY_AUTHENTIC
    if score < 0.50:
        return Verdict.UNCERTAIN
    if score < 0.75:
        return Verdict.POSSIBLY_MANIPULATED
    return Verdict.LIKELY_MANIPULATED


def _band_confidence(score: float, signal_count: int) -> ConfidenceBand:
    """Coarse confidence: how far from the fence, and how many signals agreed.

    Returned as a band rather than a number on purpose — a precise percentage
    reads as more authoritative than the underlying model warrants, and is a
    more useful gradient for anyone tuning a fake to slip under a threshold.
    """
    margin = abs(score - 0.5) * 2  # 0.0 at the fence, 1.0 at either extreme
    if signal_count >= 3:
        margin += 0.1
    if margin >= 0.6:
        return ConfidenceBand.HIGH
    if margin >= 0.3:
        return ConfidenceBand.MODERATE
    return ConfidenceBand.LOW


def _has_audio_track(path: Path) -> bool:
    """Whether the video carries a usable audio track.

    TODO(step 5): probe the container with ffprobe. Until then assume audio is
    present; the authenticator already renormalises its weights when a signal is
    missing, so the plumbing for the False case is in place and exercised.
    """
    return True


def run_analysis(media_kind: MediaKind, path: Path) -> tuple[AnalysisResult, InternalScores]:
    """Run the appropriate models for this media type and band the result."""
    analysed_at = datetime.now(timezone.utc)

    image_out: DetectorOutput | None = None
    audio_out: DetectorOutput | None = None
    frames_out: DetectorOutput | None = None

    if media_kind is MediaKind.IMAGE:
        image_out = image_model.analyze(path)
        final = image_out
        signals = ["image"]

    elif media_kind is MediaKind.AUDIO:
        audio_out = audio_model.analyze(path)
        final = audio_out
        signals = ["audio"]

    else:  # VIDEO — all three upstream models, then the authenticator.
        image_out = image_model.analyze(path)
        frames_out = raw_frames_model.analyze(path)
        if _has_audio_track(path):
            audio_out = audio_model.analyze(path)
        final = video_authenticator.fuse(image_out, audio_out, frames_out)
        signals = [
            name
            for name, out in (
                ("image", image_out),
                ("audio", audio_out),
                ("raw_frames", frames_out),
            )
            if out is not None
        ]
        signals.append("video_authenticator")

    upstream_count = sum(1 for out in (image_out, audio_out, frames_out) if out is not None)

    result = AnalysisResult(
        analysis_id="",  # filled in by the caller, which owns the job record
        media_kind=media_kind,
        verdict=_band_verdict(final.score),
        confidence=_band_confidence(final.score, upstream_count),
        evidence=final.evidence,
        signals_used=signals,
        analysed_at=analysed_at,
        media_deleted=settings.delete_upload_after_analysis,
        disclaimer=RESULT_DISCLAIMER,
        is_mock=MODELS_ARE_STUBS,
    )

    internal = InternalScores(
        image_score=image_out.score if image_out else None,
        audio_score=audio_out.score if audio_out else None,
        raw_frames_score=frames_out.score if frames_out else None,
        fused_score=final.score,
        notes=final.notes,
    )
    return result, internal


def execute_job(analysis_id: str, media_kind: MediaKind, path: Path) -> None:
    """Background entry point.

    Runs inside a FastAPI BackgroundTask for now. NOTE: this must move to a real
    worker queue (arq/Celery/RQ) before real model inference lands in step 3 —
    GPU inference inside the web process will block the event loop and cannot be
    scaled or retried independently.
    """
    job = store.get(analysis_id)
    if job is None:
        return

    job.status = AnalysisStatus.RUNNING
    store.update(job)

    try:
        result, internal = run_analysis(media_kind, path)
        result.analysis_id = analysis_id
        job.result = result
        job.internal = internal
        job.status = AnalysisStatus.COMPLETE
    except Exception as exc:  # noqa: BLE001 - surface a safe message, log the rest
        job.status = AnalysisStatus.FAILED
        # Deliberately generic: internal exception text can leak paths and model
        # details. The full traceback belongs in server logs, not the response.
        job.error = "Analysis failed while processing this file."
        _log_failure(analysis_id, exc)

    store.update(job)


def _log_failure(analysis_id: str, exc: Exception) -> None:
    import logging

    logging.getLogger(__name__).exception("analysis %s failed: %s", analysis_id, exc)


def peek_internal(job: AnalysisJob) -> InternalScores | None:
    """Internal scores, gated behind the disclosure setting.

    Returns None in any configuration intended for public deployment.
    """
    if not settings.expose_internal_scores:
        return None
    return job.internal
