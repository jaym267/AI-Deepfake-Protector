"""Image model — REAL (build step 3), covering one of its two intended heads.

Architecture: a frozen CLIP ViT-B/32 vision tower with a logistic-regression
probe over its 512-d output. The backbone is not fine-tuned. Training code and
the reasoning for that choice are in ml/; the short version is that fine-tuning
a CNN on a fixed set of generators produces a detector that learns those
generators' fingerprints and degrades badly on new ones, while a probe over
frozen CLIP features holds up far better on generators it never saw. Since every
real upload comes from a generator newer than the training data, that trade is
the whole game.

Scope — this head detects **fully AI-generated images** only.

It does *not* detect face swaps or localised edits to genuine photographs. That
is head (a) in the brief, and it needs face-manipulation training data
(FaceForensics++, Celeb-DF) which is gated behind institutional access nobody on
this project currently has (docs/DECISIONS.md D6). An unedited photograph with
one swapped face is largely a real photograph, and this model will usually call
it authentic. That limitation is surfaced to the user as evidence rather than
buried in a docstring, because a false "no signs of manipulation" on a face-swap
is exactly the failure that gets someone disbelieved.

Consequently there is no MTCNN face detection here. The brief lists it as step 3
preprocessing, but it is preprocessing for head (a); running a face detector to
inform a whole-image synthetic-content classifier would cost time and tell it
nothing.

On evidence: a linear probe over a global image embedding produces one number.
It has no spatial localisation — no heatmap, no per-region attribution. The
evidence below therefore describes the strength and direction of a whole-image
signal and nothing more. Earlier stub versions of this file emitted claims like
"blending seams around the jaw and hairline"; those were fabricated, and
inventing specific findings the model cannot support would poison the one thing
this product is for.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...config import settings
from ...errors import UndecodableUpload, UploadTooLarge
from ...schemas import EvidenceItem, Severity
from .artifacts import Thresholds, embed_image, load_probe
from .base import DetectorOutput, ModelUnavailable

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "image_synthetic_probe"

#: Formats this model will decode, matching the Content-Type allowlist in
#: uploads.py. That allowlist checks a header the client controls; this checks
#: what the bytes actually are, which is the same reasoning media_probe.py
#: applies to audio and video containers. Without it the allowlist was
#: decorative: PIL happily decoded GIF, TIFF and BMP uploads declared as
#: image/png, so the real decoder surface was every format PIL supports rather
#: than the three that were reviewed.
DECODABLE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})

#: Always attached. Neither gap is an edge case: the first is half the intended
#: model, and the second is the difference between the calibration's promise and
#: what it can deliver on a generator released after training.
SCOPE_NOTE = EvidenceItem(
    code="scope_synthetic_only",
    summary=(
        "This check looks for images generated entirely by AI. It is not able "
        "to detect a real photograph that has had a face swapped in or a region "
        "edited, so it cannot rule that out. It is also weaker against image "
        "generators newer than the ones it was trained on, and new generators "
        "appear constantly."
    ),
    severity=Severity.INFO,
)


class ImageModel:
    name = "image"

    def analyze(self, path: Path) -> DetectorOutput:
        probe = load_probe(ARTIFACT_NAME)
        if probe is None:
            raise ModelUnavailable(
                "The image model is not available on this server. "
                "Its trained weights are missing."
            )

        features = self._embed(path, probe.backbone_id)

        score = probe.score(features)
        return DetectorOutput(
            score=score,
            evidence=self._evidence(score, probe.thresholds),
            notes={
                "backbone": probe.backbone_id,
                "probe_version": probe.version,
                "head": "synthetic_only",
            },
            is_stub=False,
            thresholds=probe.thresholds,
        )

    def _embed(self, path: Path, backbone_id: str):
        """Decode an image safely, then embed it.

        Three checks before any pixel buffer is allocated, in the order that
        keeps the cheapest first:

        1. ``Image.open`` reads the header only, so the format is known before
           anything is decoded.
        2. The format is checked against the allowlist — the declared
           Content-Type is not evidence, since the client writes it.
        3. Decoded dimensions are checked before ``load()``. This is the
           decompression-bomb guard, and it is not implied by the byte cap: a
           140KB PNG decodes to 144 million pixels, and ``.convert("RGB")``
           downstream triples the allocation. PIL's own ceiling only warns below
           2x its default while still allocating, and raises
           ``DecompressionBombError`` above it — which is not an ``OSError``, so
           it escaped the old handler and surfaced as a 500 rather than a 400.
        """
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(path) as image:
                image_format = (image.format or "").upper()
                if image_format not in DECODABLE_FORMATS:
                    raise UndecodableUpload(
                        f"This file is {image_format or 'an unrecognised format'}, "
                        "which this tool doesn't read. Please upload a JPEG, PNG "
                        "or WebP image."
                    )

                width, height = image.size
                if width * height > settings.max_image_pixels:
                    megapixels = (width * height) / 1_000_000
                    allowed = settings.max_image_pixels / 1_000_000
                    raise UploadTooLarge(
                        f"This image is {width}x{height} ({megapixels:.0f} "
                        f"megapixels), over the {allowed:.0f} megapixel limit. "
                        "Resizing it will let it through."
                    )

                image.load()
                return embed_image(image, backbone_id)
        except Image.DecompressionBombError as exc:
            # Backstop: PIL's own guard, in case its accounting and ours ever
            # disagree. Same class of problem, so the same status code.
            raise UploadTooLarge(
                "This image is too large to process safely. "
                "Resizing it will let it through."
            ) from exc
        except (UnidentifiedImageError, OSError) as exc:
            # Content-type said image, decoding disagreed. Truncated uploads and
            # renamed files both land here.
            raise UndecodableUpload("This file could not be read as an image.") from exc

    def analyze_video_frames(self, path: Path) -> DetectorOutput:
        """Per-frame spatial scoring for a video — STILL A STUB (step 5).

        The real method is to decode the video, sample frames, embed each one
        with the same frozen backbone, and pool the per-frame scores. What is
        missing is only the decoding: that needs ffmpeg/PyAV, which arrives with
        the Raw Frames model in step 5, and adding a binary media dependency
        here purely to reach it early would be the wrong order.

        Kept as a separate method rather than branching inside ``analyze`` so
        that the real image path cannot silently fall through to a placeholder.
        The output is flagged ``is_stub``, which keeps every video verdict marked
        as mock — correct, since the other three video models are stubs too.
        """
        from .base import stable_pseudo_score

        score = stable_pseudo_score(path, salt="image")
        return DetectorOutput(
            score=score,
            evidence=[],  # A placeholder must not manufacture findings.
            notes={"backbone": "stub", "reason": "frame extraction lands in step 5"},
            is_stub=True,
            thresholds=None,
        )

    def _evidence(self, score: float, thresholds: Thresholds) -> list[EvidenceItem]:
        """Describe the signal honestly, in one finding plus the scope caveat."""
        if score >= thresholds.manipulated_above:
            finding = EvidenceItem(
                code="synthetic_signature_strong",
                summary=(
                    "The visual statistics of this image closely match those of "
                    "images produced by AI generators, and differ clearly from "
                    "camera photographs."
                ),
                severity=Severity.STRONG,
            )
        elif score >= thresholds.possible_above:
            finding = EvidenceItem(
                code="synthetic_signature_moderate",
                summary=(
                    "Parts of this image's visual statistics lean towards AI "
                    "generation, but not decisively. Heavy compression, "
                    "screenshots and re-saved images can all produce this."
                ),
                severity=Severity.NOTABLE,
            )
        elif score >= thresholds.authentic_below:
            finding = EvidenceItem(
                code="synthetic_signature_weak",
                summary=(
                    "This image sits between the two patterns the check can "
                    "tell apart, so it gives no clear answer either way."
                ),
                severity=Severity.NOTABLE,
            )
        else:
            finding = EvidenceItem(
                code="no_synthetic_signature",
                summary=(
                    "This image's visual statistics match camera photographs "
                    "rather than the AI generators this check knows about. On "
                    "images from a generator it hasn't seen before, this check "
                    "misses a substantial share of fakes, so this is weak "
                    "evidence that something is genuine rather than strong "
                    "evidence."
                ),
                severity=Severity.INFO,
            )
        return [finding, SCOPE_NOTE]
