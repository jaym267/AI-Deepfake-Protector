"""Image model — STUB (real implementation is build step 3).

Scope, per the architecture: this model covers two different artifact families
that happen to share an input type.

  (a) Manipulated / face-swapped real photographs — blending seams, warped
      facial geometry, inconsistent skin texture at swap boundaries.
  (b) Fully AI-generated images — diffusion/GAN fingerprints, frequency-domain
      artifacts, implausible fine detail (hands, teeth, jewellery, text).

These are distinct signals and will likely need separate heads on a shared
backbone, trained on different data:
  * (a) from face-manipulation data — currently gated behind institutional
        access (FaceForensics++, Celeb-DF), so this head is the weaker one until
        an advisor is available.
  * (b) from GenImage / Tiny-GenImage, which is open access and can be started
        immediately. NOTE: GenImage is CC BY-NC-SA 4.0 — non-commercial.

Preprocessing (step 3): MTCNN face detection + alignment before the (a) head.
When no face is present, only the (b) head is meaningful.

When run against a video, this model scores sampled individual frames — the
per-frame spatial view. Temporal behaviour is deliberately not its job; that
belongs to raw_frames_model.
"""

from __future__ import annotations

from pathlib import Path

from ...schemas import EvidenceItem, Severity
from .base import DetectorOutput, stable_pseudo_score


class ImageModel:
    name = "image"

    def analyze(self, path: Path) -> DetectorOutput:
        score = stable_pseudo_score(path, salt="image")

        evidence: list[EvidenceItem] = []
        if score > 0.6:
            evidence.append(
                EvidenceItem(
                    code="face_blend_boundary",
                    summary=(
                        "The edges of the face don't blend naturally into the "
                        "hair and neck, which often happens when one face has "
                        "been swapped onto another."
                    ),
                    severity=Severity.STRONG,
                    region="around the jaw and hairline",
                )
            )
        if score > 0.4:
            evidence.append(
                EvidenceItem(
                    code="synthetic_texture",
                    summary=(
                        "Skin texture is unusually smooth and repetitive, a "
                        "pattern common in images produced by AI generators."
                    ),
                    severity=Severity.NOTABLE,
                    region="cheeks and forehead",
                )
            )
        if not evidence:
            evidence.append(
                EvidenceItem(
                    code="no_spatial_artifacts",
                    summary=(
                        "No blending seams or generator-style texture patterns "
                        "were found in this image."
                    ),
                    severity=Severity.INFO,
                )
            )

        return DetectorOutput(
            score=score,
            evidence=evidence,
            notes={"backbone": "stub", "faces_detected": "unknown"},
        )
