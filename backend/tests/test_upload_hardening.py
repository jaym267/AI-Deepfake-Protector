"""Resource-exhaustion controls on the upload path.

Everything here is about a request that is *valid enough to be accepted* but
expensive enough to hurt: a body larger than the cap, an image whose compressed
size is tiny and whose decoded size is not, a container that is not the format it
claims. These are the cases the byte cap alone does not cover, and each one was a
live hole before the test beside it existed.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.bodylimit import DEFAULT_MAX_BODY, MULTIPART_OVERHEAD, max_body_for
from app.config import settings
from app.main import app
from app.services.models.artifacts import load_probe
from app.services.models.image_model import ARTIFACT_NAME

from .conftest import make_mp4, make_png

client = TestClient(app)

needs_image_model = pytest.mark.skipif(
    load_probe(ARTIFACT_NAME) is None,
    reason="models/image_synthetic_probe.json missing — run `python -m ml.train`",
)


def _png_of_pixels(width: int, height: int) -> bytes:
    """A solid-colour PNG. Compresses ~1000:1, which is the whole problem."""
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # needed to *create* the fixture
    try:
        buffer = io.BytesIO()
        Image.new("L", (width, height), 0).save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        Image.MAX_IMAGE_PIXELS = previous


# --- Body ceiling ------------------------------------------------------------


def test_body_ceiling_is_per_route():
    """Applying the video cap everywhere would leave /analyze/image five times
    cheaper to attack than its own configured limit allows."""
    assert max_body_for("/analyze/image") == settings.max_image_bytes + MULTIPART_OVERHEAD
    assert max_body_for("/analyze/video") == settings.max_video_bytes + MULTIPART_OVERHEAD
    assert max_body_for("/analyze/image") < max_body_for("/analyze/video")
    # Nothing else accepts a large body.
    assert max_body_for("/report") == DEFAULT_MAX_BODY
    assert max_body_for("/health") == DEFAULT_MAX_BODY


def test_oversize_body_is_rejected_before_it_is_buffered():
    """The regression this suite exists for.

    Previously a 40MB body to /analyze/image returned a correct 413 with all
    41,943,048 bytes already spooled to a temp file on disk, because Starlette
    parses the whole multipart body before the endpoint runs. The 413 was never
    the problem; the disk write behind it was.

    The assertion is that the *endpoint never runs*, which is the only way to know
    nothing was parsed.
    """
    from app.routers import analyze as analyze_router

    entered = []
    original = analyze_router.staged_upload

    def spy(*args, **kwargs):
        entered.append(True)
        return original(*args, **kwargs)

    analyze_router.staged_upload = spy
    try:
        payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * (settings.max_image_bytes * 4)
        response = client.post(
            "/analyze/image",
            files={"file": ("big.png", io.BytesIO(payload), "image/png")},
        )
    finally:
        analyze_router.staged_upload = original

    assert response.status_code == 413
    assert not entered, "handler ran, so the body was parsed and spooled to disk"


def test_body_ceiling_leaves_room_for_a_file_at_the_cap():
    """The ceiling covers multipart framing, so a file at exactly the byte cap
    must still get through to the endpoint rather than being cut off by the
    envelope around it."""
    at_cap = b"\x89PNG\r\n\x1a\n" + b"\x00" * (settings.max_image_bytes - 8)
    assert len(at_cap) == settings.max_image_bytes

    response = client.post(
        "/analyze/image",
        files={"file": ("at-cap.png", io.BytesIO(at_cap), "image/png")},
    )
    # Reaches the endpoint and is rejected there for not being a decodable image,
    # rather than being refused by the middleware.
    assert response.status_code == 400


def test_small_bodies_are_still_capped_on_non_upload_routes():
    oversize_json = {"analysis_id": "x" * (DEFAULT_MAX_BODY + 1)}
    assert client.post("/report", json=oversize_json).status_code == 413


def test_oversize_body_rejection_has_cors_headers():
    """Middleware order regression guard, same reasoning as the 429 case: CORS is
    registered outermost so it wraps a 413 the body limiter returns without
    calling through. Otherwise a browser upload that is simply too big reports an
    opaque network error instead of the size message."""
    origin = settings.cors_origins[0]
    payload = b"\x00" * (settings.max_image_bytes * 4)
    response = client.post(
        "/analyze/image",
        files={"file": ("big.png", io.BytesIO(payload), "image/png")},
        headers={"Origin": origin},
    )
    assert response.status_code == 413
    assert response.headers.get("access-control-allow-origin") == origin


# --- Decompression bombs ------------------------------------------------------


@needs_image_model
@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_decompression_bomb_within_the_byte_cap_is_rejected():
    """144 megapixels in a ~140KB file — comfortably inside the 5MB byte cap.

    Accepted with a 202 before this check existed, decoded in full, then tripled
    by the RGB conversion on the way into CLIP.
    """
    bomb = _png_of_pixels(12_000, 12_000)
    assert len(bomb) < settings.max_image_bytes, "fixture must be inside the byte cap"

    response = client.post(
        "/analyze/image", files={"file": ("bomb.png", io.BytesIO(bomb), "image/png")}
    )
    assert response.status_code == 413
    assert "megapixel" in response.json()["detail"].lower()


@needs_image_model
def test_larger_bomb_is_413_not_500():
    """Above 2x PIL's own ceiling it raises DecompressionBombError, which is not
    an OSError and so escaped the old handler as a 500."""
    bomb = _png_of_pixels(20_000, 20_000)
    response = client.post(
        "/analyze/image", files={"file": ("bomb.png", io.BytesIO(bomb), "image/png")}
    )
    assert response.status_code == 413


@needs_image_model
def test_an_ordinary_large_photograph_still_works():
    """The guard has to sit above real cameras. 24MP is a full-frame DSLR."""
    assert settings.max_image_pixels > 24_000_000
    photo = _png_of_pixels(6000, 4000)
    response = client.post(
        "/analyze/image", files={"file": ("photo.png", io.BytesIO(photo), "image/png")}
    )
    assert response.status_code == 202


# --- Format allowlist ---------------------------------------------------------


@needs_image_model
@pytest.mark.parametrize("fmt", ["GIF", "TIFF", "BMP"])
def test_declared_content_type_does_not_choose_the_decoder(fmt):
    """The Content-Type allowlist checks a header the client writes. PIL will open
    whatever the bytes actually are, so before this check the real decoder surface
    was every format PIL supports — including TIFF and BMP paths with a worse CVE
    record than the three that were reviewed.

    Same principle media_probe.py already applied to audio and video containers.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (1, 2, 3)).save(buffer, format=fmt)

    response = client.post(
        "/analyze/image",
        files={"file": ("actually_" + fmt.lower() + ".png", buffer, "image/png")},
    )
    assert response.status_code == 400
    assert fmt.lower() in response.json()["detail"].lower()


@needs_image_model
@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_allowed_formats_still_decode(fmt):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (9, 9, 9)).save(buffer, format=fmt)
    response = client.post(
        "/analyze/image", files={"file": (f"s.{fmt.lower()}", buffer, "image/png")}
    )
    assert response.status_code == 202


# --- Probe reuse --------------------------------------------------------------


def test_silent_video_drops_the_audio_signal():
    """`_has_audio_track` returned a hardcoded True with a TODO to probe the
    container, while PyAV was already probing every upload two frames up the
    stack. A silent video therefore had a hash-derived audio stub folded into
    fusion at weight 0.20 instead of being renormalised away."""
    silent = make_mp4(2)  # video stream only, no audio track
    response = client.post(
        "/analyze/video", files={"file": ("silent.mp4", io.BytesIO(silent), "video/mp4")}
    )
    assert response.status_code == 202

    result = client.get(f"/analyze/{response.json()['analysis_id']}").json()["result"]
    assert "audio" not in result["signals_used"], result["signals_used"]
    assert result["signals_used"] == ["image", "raw_frames", "video_authenticator"]


def test_image_upload_reports_no_audio_state():
    """Images are never probed, so has_audio stays unknown rather than False."""
    from app.schemas import MediaKind
    from app.uploads import staged_upload

    class _FakeUpload:
        filename = "s.png"
        content_type = "image/png"
        file = io.BytesIO(make_png())

    with staged_upload(MediaKind.IMAGE, _FakeUpload()) as staged:
        assert staged.has_audio is None
