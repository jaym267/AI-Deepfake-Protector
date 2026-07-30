"""Video Authenticator — weighted-average fallback (real training is build step 6).

Takes the outputs of all three upstream models run against the *same* video —
Image, Audio, Raw Frames — and produces the single video-level verdict.

Target design (step 6): a trained stacking meta-classifier that learns how to
weigh the three inputs, e.g. logistic regression or a small gradient-boosted
model fitted on held-out predictions over labelled video. Learned weighting
matters because the three signals are not equally trustworthy in all conditions:
a heavily compressed clip degrades the spatial signal much faster than the audio
one, and a fixed formula cannot express that.

What is implemented here is the explicitly permitted MVP fallback: a fixed
weighted average. It is a placeholder, not the design. ``is_trained`` marks which
one produced a given verdict so results are never misreported as model-learned
when they came from the fallback.

FUTURE EXTENSION (deferred, do not build now): a second stacked meta-layer above
this authenticator was discussed and explicitly deferred. This module is the top
of the ensemble for the current phase.
"""

from __future__ import annotations

from ...schemas import EvidenceItem
from .base import DetectorOutput

# Fallback weights. Raw Frames leads because temporal artifacts are the hardest
# for current generators to suppress across a whole clip; audio is weighted
# lowest because plenty of genuine video has poor or absent audio.
# These numbers are an informed starting point, NOT a tuned result — they are
# replaced wholesale by learned weights in step 6.
FALLBACK_WEIGHTS: dict[str, float] = {
    "raw_frames": 0.45,
    "image": 0.35,
    "audio": 0.20,
}


class VideoAuthenticator:
    name = "video_authenticator"

    #: False until a meta-classifier is actually trained and loaded (step 6).
    #: This is the single source of truth for whether fusion is real: it is
    #: propagated onto every DetectorOutput as ``is_stub`` rather than being a
    #: second, separately-maintained flag that could drift out of agreement with
    #: it. Step 6 flips this one value.
    is_trained = False

    def fuse(
        self,
        image: DetectorOutput | None,
        audio: DetectorOutput | None,
        raw_frames: DetectorOutput | None,
    ) -> DetectorOutput:
        available = {
            "image": image,
            "audio": audio,
            "raw_frames": raw_frames,
        }
        present = {k: v for k, v in available.items() if v is not None}
        if not present:
            raise ValueError("Video authenticator requires at least one upstream signal.")

        # Renormalise over whatever ran. A video with no audio track drops the
        # audio term rather than scoring it as 0.0, which would bias every silent
        # video toward "authentic".
        total_weight = sum(FALLBACK_WEIGHTS[k] for k in present)
        fused = sum(FALLBACK_WEIGHTS[k] * out.score for k, out in present.items()) / total_weight

        evidence: list[EvidenceItem] = []
        for out in present.values():
            evidence.extend(out.evidence)

        # Strongest findings first, so the report leads with what matters.
        severity_rank = {"strong": 0, "notable": 1, "info": 2}
        evidence.sort(key=lambda e: severity_rank.get(e.severity.value, 3))

        notes: dict[str, float | str] = {
            "fusion_method": "trained" if self.is_trained else "weighted_average_fallback",
            "signals_present": ",".join(sorted(present)),
        }

        return DetectorOutput(
            score=round(fused, 4),
            evidence=evidence,
            notes=notes,
            # Derived from is_trained, not hardcoded, so step 6 cannot flip one
            # and forget the other. The fallback is a placeholder by definition,
            # so a verdict it produced must not present as model-learned.
            is_stub=not self.is_trained,
            # Fixed weights have no calibrated score distribution behind them, so
            # there are no meaningful band boundaries to offer (D11). The pipeline
            # falls back to PLACEHOLDER_THRESHOLDS until step 6 derives real ones.
            thresholds=None,
        )
