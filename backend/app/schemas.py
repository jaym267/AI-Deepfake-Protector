"""API schemas.

Two-tier disclosure model
-------------------------
Everything in this file is split into what a client may see and what stays on
the server:

* ``AnalysisResult``      — public. Verdict band, a coarse confidence, and
                            plain-language evidence. This is what the results
                            dashboard renders.
* ``InternalScores``      — server-side only. Per-model numeric scores and the
                            thresholds behind the verdict. Never serialised into
                            a response unless ``settings.expose_internal_scores``
                            is explicitly turned on for local debugging.

The split exists because full per-model scores would turn the public site into a
free evaluation harness for anyone tuning a deepfake to evade detection.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class MediaKind(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class Verdict(str, Enum):
    """Deliberately banded, never binary.

    There is no "authentic" or "fake" value on purpose. The tool reports what the
    evidence supports, and "uncertain" is a legitimate, common outcome.
    """

    LIKELY_AUTHENTIC = "likely_authentic"
    UNCERTAIN = "uncertain"
    POSSIBLY_MANIPULATED = "possibly_manipulated"
    LIKELY_MANIPULATED = "likely_manipulated"


class ConfidenceBand(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class Severity(str, Enum):
    INFO = "info"
    NOTABLE = "notable"
    STRONG = "strong"


class EvidenceItem(BaseModel):
    """One human-readable finding.

    Plain-language explanation is the point of the product, so ``summary`` is
    required and must read as a sentence a non-expert understands. Numeric model
    output never appears here.

    In step 7 an LLM rewrites ``summary`` from the internal technical outputs;
    until then these strings come from fixed templates.
    """

    code: str = Field(description="Stable machine identifier, e.g. 'lip_sync_drift'.")
    summary: str = Field(description="Plain-language finding shown to the user.")
    severity: Severity = Severity.NOTABLE

    # Optional localisation of the finding within the media.
    start_seconds: float | None = None
    end_seconds: float | None = None
    region: str | None = Field(
        default=None,
        description="Human-readable area, e.g. 'around the mouth'. Not pixel coords.",
    )


class AnalysisResult(BaseModel):
    """Public-facing result. Safe to return to any client."""

    analysis_id: str
    media_kind: MediaKind
    verdict: Verdict
    confidence: ConfidenceBand

    evidence: list[EvidenceItem] = Field(default_factory=list)

    # Which of the four models actually contributed. For an image-only upload
    # this is just ["image"]; for video it is all three plus the authenticator.
    signals_used: list[str] = Field(default_factory=list)

    analysed_at: datetime
    media_deleted: bool = Field(
        default=True,
        description="True once the uploaded file has been erased from the server.",
    )

    disclaimer: str

    # Set while the detection models are still stubs, so the frontend can render
    # an unmistakable "not a real result" banner during steps 1-2.
    is_mock: bool = False


class InternalScores(BaseModel):
    """Server-side only. Never returned to a public client.

    Retained so the Video Authenticator (step 6) has its inputs, and so the
    report generator (step 7) has real numbers to translate into prose.
    """

    image_score: float | None = None
    audio_score: float | None = None
    raw_frames_score: float | None = None
    fused_score: float | None = None
    notes: dict[str, float | str] = Field(default_factory=dict)


class AnalysisJob(BaseModel):
    """Full server-side record. ``result`` is the only part clients ever see."""

    analysis_id: str
    media_kind: MediaKind
    status: AnalysisStatus
    created_at: datetime
    filename: str | None = None
    result: AnalysisResult | None = None
    internal: InternalScores | None = None
    error: str | None = None


class AnalysisAccepted(BaseModel):
    """202 response body. The client polls ``poll_url`` until status is terminal."""

    analysis_id: str
    status: AnalysisStatus
    poll_url: str


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    result: AnalysisResult | None = None
    error: str | None = None


class ReportRequest(BaseModel):
    """A report is requested by id, not by resubmitting the result.

    Letting a client post back an arbitrary result body would mean anyone could
    mint an official-looking evidence report saying whatever they wanted, which
    is exactly the document this project tells people to send to a platform or
    an employer. The server regenerates it from its own stored record.
    """

    analysis_id: str
    format: Literal["pdf"] = "pdf"


class ReportResponse(BaseModel):
    analysis_id: str
    format: Literal["pdf"]
    status: Literal["ready", "not_implemented"]
    download_url: str | None = None
    detail: str | None = None
