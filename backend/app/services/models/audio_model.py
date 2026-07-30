"""Audio model — STUB (real implementation is build step 4).

Standalone voice-clone / synthetic-speech detector. It must work on its own, not
only as a sub-component of video analysis: a voicemail or a recorded scam call
is a first-class upload, and for those this model's output *is* the final
verdict with nothing downstream of it.

Planned approach (step 4): a frozen speech backbone (WavLM) with a linear probe,
trained on ASVspoof 2019 LA, mirroring the image model's architecture. Scoring
happens over overlapping windows rather than the whole clip, which means this
model — unlike the image probe — will be able to report genuine timestamps.

This stub emits NO findings of its own. Earlier versions invented them, including
fabricated timestamps ("the voice carries a faint synthetic texture", 2.0s-5.5s).
Those were removed: they are the practice D14 prohibits, and `is_mock` marking a
result as placeholder does not make a fabricated finding inside it acceptable.
"""

from __future__ import annotations

from pathlib import Path

from ...schemas import EvidenceItem, Severity
from .base import DetectorOutput, stable_pseudo_score

#: The one finding a stub is entitled to make: that it isn't a model yet.
#: Not an empty list, because the API contract requires every result to carry at
#: least one finding and an audio-only upload has nothing else to show.
#:
#: The code names the model, rather than being a shared "model_not_implemented".
#: A video result collects findings from both this stub and the raw-frames one,
#: and `code` is the identity of a finding — the dashboard keys its list on it, so
#: a shared value made two distinct findings collide on every video upload.
NOT_IMPLEMENTED_NOTE = EvidenceItem(
    code="audio_model_not_implemented",
    summary=(
        "Audio analysis isn't available yet. This result is a placeholder and "
        "says nothing about the recording you uploaded."
    ),
    severity=Severity.INFO,
)


class AudioModel:
    name = "audio"

    def analyze(self, path: Path) -> DetectorOutput:
        score = stable_pseudo_score(path, salt="audio")
        return DetectorOutput(
            score=score,
            evidence=[NOT_IMPLEMENTED_NOTE],
            notes={"backbone": "stub", "speech_detected": "unknown"},
            is_stub=True,
            thresholds=None,
        )
