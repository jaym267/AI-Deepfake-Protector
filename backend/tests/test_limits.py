"""Duration limits and rate limiting — D5 items 3 and 4.

Both are abuse controls on a free, unauthenticated endpoint, so the tests here
care about what happens when someone is *not* being well behaved.
"""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.ratelimit import Rule, SlidingWindowLimiter

from .conftest import make_mp4, make_png, make_wav

client = TestClient(app)


def _upload(kind: str, data: bytes, content_type: str, name: str = "sample"):
    return client.post(
        f"/analyze/{kind}",
        files={"file": (name, io.BytesIO(data), content_type)},
    )


# --- Duration limits --------------------------------------------------------


def test_video_within_duration_is_accepted():
    assert _upload("video", make_mp4(2), "video/mp4").status_code == 202


def test_audio_within_duration_is_accepted():
    assert _upload("audio", make_wav(3), "audio/wav").status_code == 202


def test_overlong_video_is_rejected():
    """65s against a 60s cap, at a file size well under the 25MB byte cap —
    so this can only be the duration check firing."""
    data = make_mp4(65)
    assert len(data) < settings.max_video_bytes

    response = _upload("video", data, "video/mp4")
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "60 seconds" in detail and "65 seconds" in detail


def test_overlong_audio_is_rejected():
    data = make_wav(130)
    assert len(data) < settings.max_audio_bytes

    response = _upload("audio", data, "audio/wav")
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "2.2 minutes" in detail and "2 minutes" in detail


def test_audio_endpoint_rejects_a_file_with_no_audio_track():
    """A silent video renamed to .wav passes the Content-Type allowlist, because
    the client chooses that header. The container does not lie."""
    response = client.post(
        "/analyze/audio",
        files={"file": ("actually_video.wav", io.BytesIO(make_mp4(2)), "audio/wav")},
    )
    assert response.status_code == 400
    assert "audio track" in response.json()["detail"]


def test_corrupt_container_is_400_not_500():
    response = _upload("video", b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 2048, "video/mp4")
    assert response.status_code == 400
    assert "could not be read" in response.json()["detail"]


def test_duration_check_does_not_run_on_images():
    """Images have no duration, and a second parser on untrusted bytes would be
    attack surface for nothing — PIL already decodes them downstream."""
    assert _upload("image", make_png(), "image/png").status_code in (202, 503)


# --- Rate limiter unit behaviour --------------------------------------------


def test_limiter_allows_up_to_the_limit_then_blocks():
    limiter = SlidingWindowLimiter()
    rule = Rule(limit=3, window_seconds=60, name="t")

    assert all(limiter.check("a", rule, now=100.0) is None for _ in range(3))
    retry = limiter.check("a", rule, now=100.0)
    assert retry is not None and 0 < retry <= 60


def test_limiter_is_per_client():
    limiter = SlidingWindowLimiter()
    rule = Rule(limit=1, window_seconds=60, name="t")

    assert limiter.check("a", rule, now=100.0) is None
    assert limiter.check("a", rule, now=100.0) is not None
    # A different client must be unaffected.
    assert limiter.check("b", rule, now=100.0) is None


def test_window_slides():
    limiter = SlidingWindowLimiter()
    rule = Rule(limit=2, window_seconds=60, name="t")

    assert limiter.check("a", rule, now=0.0) is None
    assert limiter.check("a", rule, now=30.0) is None
    assert limiter.check("a", rule, now=45.0) is not None
    # The first hit has aged out by t=61.
    assert limiter.check("a", rule, now=61.0) is None


def test_rejected_requests_do_not_extend_the_window():
    """Otherwise a client that keeps retrying locks itself out forever, which
    punishes an impatient user far more than an attacker."""
    limiter = SlidingWindowLimiter()
    rule = Rule(limit=1, window_seconds=60, name="t")

    assert limiter.check("a", rule, now=0.0) is None
    for moment in (10.0, 20.0, 30.0, 50.0):
        assert limiter.check("a", rule, now=moment) is not None
    # The single allowed hit was at t=0, so t=61 must be clear.
    assert limiter.check("a", rule, now=61.0) is None


def test_retry_after_counts_down():
    limiter = SlidingWindowLimiter()
    rule = Rule(limit=1, window_seconds=60, name="t")

    limiter.check("a", rule, now=0.0)
    assert limiter.check("a", rule, now=10.0) == pytest.approx(50.0)
    assert limiter.check("a", rule, now=59.0) == pytest.approx(1.0)


# --- Rate limiter over HTTP --------------------------------------------------


def test_burst_limit_returns_429_with_retry_after(enable_rate_limiting):
    png = make_png()
    codes = [
        _upload("image", png, "image/png").status_code
        for _ in range(settings.analyze_burst_limit + 3)
    ]
    assert 429 in codes, f"burst limit never fired: {codes}"

    blocked = _upload("image", png, "image/png")
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0
    assert "wait" in blocked.json()["detail"].lower()


def test_rate_limited_response_still_has_cors_headers(enable_rate_limiting):
    """Middleware order regression guard.

    If the limiter is registered outside CORS it short-circuits before CORS can
    add headers, and a browser reports an opaque network error instead of the
    'please wait' message. `add_middleware` inserts at the front of the stack,
    so CORS has to be added *last* to end up outermost.
    """
    origin = settings.cors_origins[0]
    png = make_png()
    for _ in range(settings.analyze_burst_limit + 2):
        response = client.post(
            "/analyze/image",
            files={"file": ("s.png", io.BytesIO(png), "image/png")},
            headers={"Origin": origin},
        )
        if response.status_code == 429:
            assert response.headers.get("access-control-allow-origin") == origin
            return
    pytest.fail("rate limit never triggered")


def test_reads_are_limited_far_more_generously_than_uploads(enable_rate_limiting):
    """The frontend polls ~1/s while a job runs (D1); a write-tuned limit would
    break normal use."""
    assert settings.read_limit > settings.analyze_burst_limit * 10
    for _ in range(settings.analyze_burst_limit + 5):
        assert client.get("/health").status_code == 200


def test_limiting_is_disabled_by_the_setting():
    """The autouse fixture turns it off, so a long test run cannot self-throttle."""
    assert settings.rate_limit_enabled is False
    png = make_png()
    for _ in range(settings.analyze_burst_limit + 5):
        assert _upload("image", png, "image/png").status_code != 429


def test_forwarded_for_is_ignored_unless_trusted(enable_rate_limiting, monkeypatch):
    """Any client can set X-Forwarded-For. Honouring it on a directly exposed
    server would make the limiter bypassable with one curl flag."""
    monkeypatch.setattr(settings, "trust_forwarded_for", False)
    png = make_png()

    saw_429 = False
    for index in range(settings.analyze_burst_limit + 3):
        response = client.post(
            "/analyze/image",
            files={"file": ("s.png", io.BytesIO(png), "image/png")},
            # A new "source" address every time.
            headers={"X-Forwarded-For": f"10.0.0.{index}"},
        )
        if response.status_code == 429:
            saw_429 = True
            break
    assert saw_429, "spoofed X-Forwarded-For bypassed the limiter"


def test_client_eviction_bounds_memory():
    """A spoofed-IP flood must not grow the limiter without bound."""
    from app.ratelimit import MAX_TRACKED_CLIENTS

    limiter = SlidingWindowLimiter()
    rule = Rule(limit=5, window_seconds=60, name="t")
    now = time.monotonic()
    for index in range(MAX_TRACKED_CLIENTS + 500):
        limiter.check(f"client-{index}", rule, now=now)

    assert len(limiter._hits) <= MAX_TRACKED_CLIENTS + 500
