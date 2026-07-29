"""Common contract for the four detection models.

Every model in this package — Image, Audio, Raw Frames, Video Authenticator —
implements ``Detector``. Steps 3-6 replace each stub body with a real fine-tuned
backbone without changing this interface or anything upstream of it.

Score convention: ``score`` is P(manipulated), in [0.0, 1.0]. 0.0 means nothing
suspicious found; 1.0 means strong evidence of manipulation. Every model uses
this same direction so the authenticator can combine them coherently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ...schemas import EvidenceItem
from .artifacts import Thresholds


class ModelUnavailable(RuntimeError):
    """A real model was asked for but its artifact isn't present.

    Raised rather than falling back to a placeholder score. A detector that
    silently degrades to guessing is worse than one that admits it is offline:
    the caller cannot tell the difference, but the person reading the verdict
    acts on it either way.
    """


@dataclass
class DetectorOutput:
    score: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    notes: dict[str, float | str] = field(default_factory=dict)

    #: False once this detector is a trained model rather than a placeholder.
    #: Propagated to ``AnalysisResult.is_mock`` so the UI banner tracks reality
    #: per model, instead of one global flag that stays wrong for three of the
    #: four steps it takes to replace them all.
    is_stub: bool = True

    #: Band boundaries calibrated for *this* detector's score distribution.
    #: None means the caller must fall back to placeholders (D4).
    thresholds: Thresholds | None = None


class Detector(Protocol):
    name: str

    def analyze(self, path: Path) -> DetectorOutput: ...


def stable_pseudo_score(path: Path, salt: str) -> float:
    """Deterministic stand-in score derived from the file's own bytes.

    Used only by the stubs. Being deterministic per (file, model) means the same
    upload always demos the same way and the frontend can be tested against a
    stable fixture — unlike random(), which would make the results dashboard
    impossible to eyeball.

    Deleted in steps 3-6 as each real model lands.
    """
    digest = hashlib.sha256()
    digest.update(salt.encode())
    with path.open("rb") as handle:
        # Head only: enough to vary per file, cheap for a 25MB video.
        digest.update(handle.read(64 * 1024))
    bucket = int.from_bytes(digest.digest()[:4], "big")
    return round((bucket % 10_000) / 10_000, 4)
