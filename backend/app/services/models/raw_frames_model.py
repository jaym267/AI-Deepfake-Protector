"""Raw Frames model — STUB (real implementation is build step 5).

Video-only temporal analysis. This is a separate model from the Image model
because its entire signal lives in how frames relate to each other over time —
information a single still frame cannot carry:

  * flicker and frame-to-frame instability in the face region
  * unnatural blink rate or blink duration
  * lighting and shadow direction that shifts inconsistently between frames
  * head/jaw motion that doesn't move continuously

Planned approach (step 5): a sequence model over per-frame features from a
pretrained video/vision backbone, fine-tuned on DFDC (open with a free Kaggle
account). Runs on aligned face crops from the same MTCNN preprocessing the Image
model uses, so both models see a consistent view of the subject.
"""

from __future__ import annotations

from pathlib import Path

from ...schemas import EvidenceItem, Severity
from .base import DetectorOutput, stable_pseudo_score


class RawFramesModel:
    name = "raw_frames"

    def analyze(self, path: Path) -> DetectorOutput:
        score = stable_pseudo_score(path, salt="raw_frames")

        evidence: list[EvidenceItem] = []
        if score > 0.6:
            evidence.append(
                EvidenceItem(
                    code="temporal_flicker",
                    summary=(
                        "The face flickers and shifts slightly between frames "
                        "instead of moving smoothly, which is a common sign of "
                        "a face being generated frame by frame."
                    ),
                    severity=Severity.STRONG,
                    start_seconds=4.0,
                    end_seconds=7.0,
                )
            )
        if score > 0.45:
            evidence.append(
                EvidenceItem(
                    code="blink_rate_anomaly",
                    summary=(
                        "The person blinks far less often than someone normally "
                        "would over this length of footage."
                    ),
                    severity=Severity.NOTABLE,
                )
            )
        if not evidence:
            evidence.append(
                EvidenceItem(
                    code="stable_motion",
                    summary=(
                        "Movement, lighting and blinking stay consistent across "
                        "the whole clip."
                    ),
                    severity=Severity.INFO,
                )
            )

        return DetectorOutput(
            score=score,
            evidence=evidence,
            notes={"backbone": "stub", "frames_sampled": "unknown"},
        )
