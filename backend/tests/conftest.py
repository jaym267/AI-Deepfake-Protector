"""Shared fixtures.

Two things changed the shape of these tests once duration limits and rate
limiting landed:

**Media fixtures have to be real.** Before, `b"\\x00\\x00\\x00\\x20ftypmp42"`
plus zero padding was enough, because nothing parsed it. The duration check
opens every audio and video upload with ffmpeg, so fixtures are now genuinely
encoded files. They are tiny (a 64x64 two-second clip is ~2KB) and generated at
import, so nothing binary is committed.

**Rate limiting is off unless a test asks for it.** The suite makes far more
requests per minute than a human would, so leaving it on would make unrelated
tests fail with 429 depending on execution order — the sort of flake that gets a
security control disabled rather than debugged.
"""

from __future__ import annotations

import io
import wave

import pytest

from app.config import settings
from app.ratelimit import limiter


@pytest.fixture(autouse=True)
def disable_rate_limiting(monkeypatch):
    """Off by default. `enable_rate_limiting` opts back in."""
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def enable_rate_limiting(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    limiter.reset()
    yield
    limiter.reset()


def make_png(colour: tuple[int, int, int] = (120, 90, 60), size: int = 64) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def make_wav(seconds: float, rate: int = 8000) -> bytes:
    """A real PCM WAV. Uses the stdlib rather than PyAV — simpler, and it keeps
    at least one fixture independent of the library under test."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def make_mp4(seconds: float, fps: int = 10, size: int = 64) -> bytes:
    """A real H.264 MP4 of the requested duration."""
    import av
    import numpy as np

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = stream.height = size
    stream.pix_fmt = "yuv420p"

    for index in range(int(seconds * fps)):
        frame = av.VideoFrame.from_ndarray(
            np.full((size, size, 3), index % 255, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue()
