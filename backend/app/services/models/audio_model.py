"""Audio model — STUB (real implementation is build step 4).

Standalone voice-clone / synthetic-speech detector. It must work on its own, not
only as a sub-component of video analysis: a voicemail or a recorded scam call
is a first-class upload, and for those this model's output *is* the final
verdict with nothing downstream of it.

Planned approach (step 4): fine-tune a pretrained speech backbone (Wav2Vec2 or
similar) on ASVspoof + WaveFake, both open to individuals. Signals of interest
are vocoder artifacts, unnatural prosody and breath patterns, and spectral
discontinuities at splice points.
"""

from __future__ import annotations

from pathlib import Path

from ...schemas import EvidenceItem, Severity
from .base import DetectorOutput, stable_pseudo_score


class AudioModel:
    name = "audio"

    def analyze(self, path: Path) -> DetectorOutput:
        score = stable_pseudo_score(path, salt="audio")

        evidence: list[EvidenceItem] = []
        if score > 0.6:
            evidence.append(
                EvidenceItem(
                    code="vocoder_artifacts",
                    summary=(
                        "The voice carries a faint synthetic texture typical of "
                        "speech produced by a voice-cloning tool rather than "
                        "recorded from a person."
                    ),
                    severity=Severity.STRONG,
                    start_seconds=2.0,
                    end_seconds=5.5,
                )
            )
        if score > 0.4:
            evidence.append(
                EvidenceItem(
                    code="absent_breath",
                    summary=(
                        "The speaker never pauses to breathe where a person "
                        "normally would, which is common in generated speech."
                    ),
                    severity=Severity.NOTABLE,
                )
            )
        if not evidence:
            evidence.append(
                EvidenceItem(
                    code="natural_speech_pattern",
                    summary=(
                        "Breathing, pacing and background noise in this "
                        "recording look like a genuine recording of a person."
                    ),
                    severity=Severity.INFO,
                )
            )

        return DetectorOutput(
            score=score,
            evidence=evidence,
            notes={"backbone": "stub", "speech_detected": "unknown"},
        )
