"""Container inspection: duration and stream-type validation.

Closes D5 item 4. Until now the byte caps were the only real limit, and they are
a poor proxy for cost: a 25MB H.264 file can be ten minutes long, and step 5 will
run three models across its frames. Duration is what actually bounds the work.

Uses PyAV rather than shelling out to ``ffprobe`` — it ships its own ffmpeg
libraries, so there is no system binary to install, no subprocess to sanitise,
and no parsing of another program's stdout. Step 5 needs PyAV for frame
extraction regardless.

Two checks, both cheap because they read container metadata rather than decoding:

1. **Duration** against the per-kind cap.
2. **Stream type** — an "audio" upload must actually contain an audio stream.
   This is a stronger check than the declared Content-Type, which the client
   controls, and it catches a renamed file that the allowlist would wave through.

Security note: this hands attacker-controlled bytes to ffmpeg's demuxers, which
have a real CVE history. The exposure is unavoidable — the product's whole job is
analysing uploaded media, and step 5 decodes these files anyway — but it argues
for keeping PyAV patched and, before any public deploy, running analysis in a
sandbox rather than in the API process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .schemas import MediaKind

logger = logging.getLogger(__name__)

#: PyAV reports container duration in microseconds (ffmpeg's AV_TIME_BASE).
AV_TIME_BASE = 1_000_000


class UnreadableMedia(ValueError):
    """The container could not be parsed, or lacks the stream it should have."""


@dataclass(frozen=True)
class ProbeResult:
    duration_seconds: float | None
    has_video: bool
    has_audio: bool


def probe(path: Path) -> ProbeResult:
    """Read container metadata. Raises UnreadableMedia if it cannot be parsed."""
    import av
    import av.error

    try:
        with av.open(str(path)) as container:
            has_video = bool(container.streams.video)
            has_audio = bool(container.streams.audio)
            duration = _duration_seconds(container)
    except (av.error.FFmpegError, ValueError, OSError) as exc:
        raise UnreadableMedia(
            "This file could not be read as media. It may be corrupt, "
            "incomplete, or not the format its name suggests."
        ) from exc

    return ProbeResult(duration_seconds=duration, has_video=has_video, has_audio=has_audio)


def _duration_seconds(container) -> float | None:
    """Container duration, falling back to the longest stream.

    Some WebM and fragmented MP4 files carry no container-level duration, but
    their streams still declare one. Returns None when nothing does, which the
    caller must treat as a rejection rather than as "fine" — see enforce_limits.
    """
    if container.duration is not None:
        return float(container.duration) / AV_TIME_BASE

    longest: float | None = None
    for stream in container.streams:
        if stream.duration is not None and stream.time_base is not None:
            seconds = float(stream.duration * stream.time_base)
            longest = seconds if longest is None else max(longest, seconds)
    return longest


def enforce_limits(kind: MediaKind, path: Path, max_seconds: float | None) -> ProbeResult:
    """Validate a staged upload's container, or raise UnreadableMedia.

    Images are skipped: they have no duration, and PIL already decodes them in
    the image model, so a second parser here would add attack surface for nothing.
    """
    if kind is MediaKind.IMAGE:
        return ProbeResult(duration_seconds=None, has_video=False, has_audio=False)

    result = probe(path)

    if kind is MediaKind.VIDEO and not result.has_video:
        raise UnreadableMedia(
            "This file doesn't contain a video track. If it's an audio "
            "recording, use the audio option instead."
        )
    if kind is MediaKind.AUDIO and not result.has_audio:
        raise UnreadableMedia("This file doesn't contain an audio track.")

    if max_seconds is None:
        return result

    if result.duration_seconds is None:
        # Unknown length is rejected rather than waved through. A file whose
        # duration cannot be determined could be arbitrarily long, and the whole
        # point of this check is to bound the work step 5 will do on it.
        raise UnreadableMedia(
            "The length of this file could not be determined, so it can't be "
            "accepted. Re-saving or re-encoding it usually fixes this."
        )

    if result.duration_seconds > max_seconds:
        actual = _human_duration(result.duration_seconds)
        allowed = _human_duration(max_seconds)
        raise DurationExceeded(
            f"This {kind.value} is {actual} long, over the {allowed} limit. "
            "Limits are strict while this is in development."
        )

    return result


class DurationExceeded(UnreadableMedia):
    """Parsed fine, but longer than the cap. Separated so the router can map it
    to 413 (too large) rather than 400 (malformed)."""


def _human_duration(seconds: float) -> str:
    """Phrase a duration the way a person would say it.

    Switching to minutes at exactly 60 produced "this video is 1.1 minutes long,
    over the 1.0 minutes limit", which is both clumsy and harder to act on than
    the same fact in seconds. Round numbers lose the pointless ".0" too.
    """
    if seconds < 90:
        return f"{seconds:.0f} seconds"

    minutes = seconds / 60
    if abs(minutes - round(minutes)) < 0.05:
        return f"{round(minutes)} minutes"
    return f"{minutes:.1f} minutes"
