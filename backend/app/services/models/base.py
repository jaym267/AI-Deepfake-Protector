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


@dataclass
class DetectorOutput:
    score: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    notes: dict[str, float | str] = field(default_factory=dict)


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
