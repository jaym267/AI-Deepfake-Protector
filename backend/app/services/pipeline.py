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
    AnalysisResult,
    ConfidenceBand,
    InternalScores,
    MediaKind,
    Verdict,
)
from .disclaimer import RESULT_DISCLAIMER
from .models import AudioModel, ImageModel, RawFramesModel, VideoAuthenticator
from .models.artifacts import Thresholds
from .models.base import DetectorOutput

image_model = ImageModel()
audio_model = AudioModel()
raw_frames_model = RawFramesModel()
video_authenticator = VideoAuthenticator()

#: Which detectors are trained models rather than placeholders. Reported by
#: /health. Per-model rather than one global flag, because between steps 3 and 6
#: a single boolean is necessarily wrong about something.
REAL_MODELS: dict[str, bool] = {
    "image": True,  # step 3 — frozen CLIP + linear probe
    "audio": False,  # step 4
    "raw_frames": False,  # step 5
    "video_authenticator": False,  # step 6
}

#: Used only by detectors that have no calibration of their own — i.e. the ones
#: that are still stubs. Picked by eye, which is exactly why they are not
#: allowed anywhere near a trained model's output (D4).
PLACEHOLDER_THRESHOLDS = Thresholds(
    authentic_below=0.25,
    possible_above=0.50,
    manipulated_above=0.75,
)


def _band_verdict(score: float, thresholds: Thresholds | None) -> Verdict:
    """Map P(manipulated) onto a verdict band.

    Four bands, never two. A binary real/fake badge would misrepresent what the
    models can support, and 'uncertain' has to stay a visible outcome rather than
    being rounded into one of the confident answers.

    Boundaries come from the detector that produced the score, because they are
    only meaningful relative to that detector's score distribution. The image
    probe's are derived from its validation ROC and pinned to explicit error
    rates (see ml/train.py:derive_thresholds); stub detectors fall back to the
    placeholders above.
    """
    bounds = thresholds or PLACEHOLDER_THRESHOLDS
    if score < bounds.authentic_below:
        return Verdict.LIKELY_AUTHENTIC
    if score < bounds.possible_above:
        return Verdict.UNCERTAIN
    if score < bounds.manipulated_above:
        return Verdict.POSSIBLY_MANIPULATED
    return Verdict.LIKELY_MANIPULATED


def _band_confidence(
    score: float,
    signal_count: int,
    thresholds: Thresholds | None,
) -> ConfidenceBand:
    """Coarse confidence: how far from the decision fence, and how many signals agreed.

    Returned as a band rather than a number on purpose — a precise percentage
    reads as more authoritative than the underlying model warrants, and is a
    more useful gradient for anyone tuning a fake to slip under a threshold.

    The fence is the detector's own equal-error point, not a hardcoded 0.5.
    Those coincide only by accident, and on a calibrated model the difference
    decides whether a borderline result is shown as 'moderate' or 'low'.
    """
    bounds = thresholds or PLACEHOLDER_THRESHOLDS
    fence = bounds.possible_above
    # Normalise distance from the fence by whichever side the score falls on, so
    # an asymmetric calibration doesn't make one direction look more certain
    # purely because its threshold sits closer to the end of the range.
    span = (1.0 - fence) if score >= fence else fence
    margin = abs(score - fence) / max(span, 1e-6)

    if signal_count >= 3:
        margin += 0.1
    if margin >= 0.6:
        return ConfidenceBand.HIGH
    if margin >= 0.3:
        return ConfidenceBand.MODERATE
    return ConfidenceBand.LOW


def run_analysis(
    media_kind: MediaKind,
    path: Path,
    has_audio: bool | None = None,
) -> tuple[AnalysisResult, InternalScores]:
    """Run the appropriate models for this media type and band the result.

    ``has_audio`` comes from the container probe already performed during upload
    validation (``uploads.StagedUpload``). It used to be a local helper that
    returned a hardcoded ``True`` with a TODO to "probe the container with
    ffprobe" — while PyAV was already probing every video upload two frames up
    the stack and reporting exactly this. The consequence was that a silent video
    had a hash-derived audio stub folded into fusion at weight 0.20 instead of
    being renormalised away, and the False branch the authenticator implements
    was never reachable.

    ``None`` means unknown, which is treated as present: that is the old
    behaviour, and it only applies to callers that do not have a probe result.
    """
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
        # Not image_model.analyze(): that decodes a still image and would reject
        # an MP4 outright. Scoring a video with the frame-level model needs frame
        # extraction, which arrives in step 5.
        image_out = image_model.analyze_video_frames(path)
        frames_out = raw_frames_model.analyze(path)
        if has_audio is not False:
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

    # Mock if *anything* that fed this verdict was a placeholder. A video whose
    # frame analysis is real but whose audio and fusion are stubs is still a
    # result nobody should act on, and the banner has to say so.
    contributing = [out for out in (image_out, audio_out, frames_out, final) if out is not None]
    is_mock = any(out.is_stub for out in contributing)

    result = AnalysisResult(
        analysis_id="",  # filled in by the caller, which owns the job record
        media_kind=media_kind,
        verdict=_band_verdict(final.score, final.thresholds),
        confidence=_band_confidence(final.score, upstream_count, final.thresholds),
        evidence=final.evidence,
        signals_used=signals,
        analysed_at=analysed_at,
        media_deleted=settings.delete_upload_after_analysis,
        disclaimer=RESULT_DISCLAIMER,
        is_mock=is_mock,
    )

    internal = InternalScores(
        image_score=image_out.score if image_out else None,
        audio_score=audio_out.score if audio_out else None,
        raw_frames_score=frames_out.score if frames_out else None,
        fused_score=final.score,
        notes=final.notes,
    )
    return result, internal


# NOTE (step 6): analysis still runs inline inside the request handler, which is
# what D1 warned about. `execute_job` used to sit here as a background entry
# point for a queue that was never wired up — nothing called it, so it was a
# plausible-looking piece of infrastructure that would have gone stale silently.
# Deleted rather than left to rot. The move to a real worker (arq/Celery/RQ) is
# tracked as its own piece of work; the response contract (202 + poll) is already
# the one a queue needs, so nothing above this changes when it lands.
#
# `peek_internal` was removed for the same reason. It gated InternalScores behind
# `settings.expose_internal_scores`, and no route ever called it — which means the
# setting was documented in D3 as the enforcement mechanism for the two-tier
# disclosure rule while enforcing nothing. The actual enforcement is structural
# and stronger: `AnalysisStatusResponse` has no field capable of carrying these
# scores, so there is no code path that could serialise them and no flag anyone
# can flip in production to start leaking them.
