"""Raw Frames model — STUB (real implementation is build step 5).

Video-only temporal analysis. This is a separate model from the Image model
because its entire signal lives in how frames relate to each other over time —
information a single still frame cannot carry:

  * flicker and frame-to-frame instability in the face region
  * unnatural blink rate or blink duration
  * lighting and shadow direction that shifts inconsistently between frames
  * head/jaw motion that doesn't move continuously

Planned approach (step 5): per-frame embeddings from the same frozen CLIP
backbone the image model uses, then a probe over *temporal statistics* —
frame-to-frame cosine distance, embedding variance, magnitude of change between
consecutive frames. Flicker is literally instability in that embedding sequence.
Trained on DFDC (free Kaggle account, accept the competition rules).

Note on what that will and won't measure: whole-frame temporal statistics capture
global instability, NOT blink rate or jaw motion. Those need face detection and
landmark tracking, and the evidence this model emits must not claim them unless
they are actually being measured.

This stub emits NO findings of its own. Earlier versions invented them, including
a fabricated blink-rate claim and timestamps (4.0s-7.0s) for footage it never
decoded. Removed per D14 — `is_mock` marks a result as a placeholder, but it does
not make a fabricated finding inside that result acceptable.
"""

from __future__ import annotations

from pathlib import Path

from ...schemas import EvidenceItem, Severity
from .base import DetectorOutput, stable_pseudo_score

#: See audio_model.NOT_IMPLEMENTED_NOTE — same reasoning, including why the code
#: names this model specifically. Video results also carry findings from the other
#: stubs, so this one rarely stands alone, but it must still be truthful about
#: what produced it and distinguishable from the ones beside it.
NOT_IMPLEMENTED_NOTE = EvidenceItem(
    code="raw_frames_model_not_implemented",
    summary=(
        "Frame-by-frame video analysis isn't available yet. This result is a "
        "placeholder and says nothing about the video you uploaded."
    ),
    severity=Severity.INFO,
)


class RawFramesModel:
    name = "raw_frames"

    def analyze(self, path: Path) -> DetectorOutput:
        score = stable_pseudo_score(path, salt="raw_frames")
        return DetectorOutput(
            score=score,
            evidence=[NOT_IMPLEMENTED_NOTE],
            notes={"backbone": "stub", "frames_sampled": "unknown"},
            is_stub=True,
            thresholds=None,
        )
