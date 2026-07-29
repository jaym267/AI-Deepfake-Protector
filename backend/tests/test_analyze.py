"""API contract tests.

These assert the *contract* the frontend is built against — verdict bands,
disclosure rules, error codes — not detection quality. Model quality is measured
in ml/evaluate.py against a held-out validation set; a unit test that asserted an
accuracy figure here would be asserting against 35 000 cached feature vectors it
does not have.

Since step 3 the image path runs a real model, so image tests need
``models/image_synthetic_probe.json`` (committed) and the CLIP backbone in the
Hugging Face cache (downloaded on first use). They skip rather than fail if the
artifact is absent, so a contract regression stays distinguishable from a
missing-weights environment.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.models.artifacts import load_probe
from app.services.models.image_model import ARTIFACT_NAME

client = TestClient(app)

WAV_BYTES = b"RIFF" + b"\x00" * 2048
MP4_BYTES = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 2048

#: Bytes that pass the content-type allowlist but are not a decodable image.
CORRUPT_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048

HAS_IMAGE_MODEL = load_probe(ARTIFACT_NAME) is not None
needs_image_model = pytest.mark.skipif(
    not HAS_IMAGE_MODEL,
    reason="models/image_synthetic_probe.json missing — run `python -m ml.train`",
)


def real_png(colour: tuple[int, int, int] = (120, 90, 60), size: int = 64) -> bytes:
    """A small but genuinely decodable PNG.

    The step-1 stub hashed raw bytes, so a PNG magic number followed by zeros was
    enough. The real model decodes and embeds the image, so fixtures have to be
    actual images now.
    """
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buffer, format="PNG")
    return buffer.getvalue()


PNG_BYTES = real_png()


def _upload(kind: str, data: bytes, content_type: str, name: str = "sample"):
    return client.post(
        f"/analyze/{kind}",
        files={"file": (name, io.BytesIO(data), content_type)},
    )


def _result(response) -> dict:
    analysis_id = response.json()["analysis_id"]
    return client.get(f"/analyze/{analysis_id}").json()["result"]


# --- Meta -------------------------------------------------------------------


def test_health_reports_per_model_state():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    # Audio, raw frames and the authenticator are still stubs (steps 4-6), so
    # the aggregate flag stays true even though the image model is real.
    assert body["models_are_stubs"] is True
    assert body["models"]["image"] is HAS_IMAGE_MODEL
    assert body["models"]["audio"] is False


def test_limits_match_config():
    body = client.get("/limits").json()
    assert body["image"]["max_bytes"] == settings.max_image_bytes
    assert body["video"]["max_seconds"] == settings.max_video_seconds


# --- Contract ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "data", "ctype"),
    [
        pytest.param("image", PNG_BYTES, "image/png", marks=needs_image_model),
        ("audio", WAV_BYTES, "audio/wav"),
        ("video", MP4_BYTES, "video/mp4"),
    ],
)
def test_analyze_accepts_and_completes(kind, data, ctype):
    response = _upload(kind, data, ctype)
    assert response.status_code == 202
    assert response.json()["status"] == "complete"

    result = _result(response)
    assert result["media_kind"] == kind
    assert result["verdict"] in {
        "likely_authentic",
        "uncertain",
        "possibly_manipulated",
        "likely_manipulated",
    }
    assert result["confidence"] in {"low", "moderate", "high"}
    assert result["evidence"], "every result must carry at least one finding"
    assert result["disclaimer"]
    assert result["media_deleted"] is True


def test_routing_matches_architecture():
    """Image/audio use one model; video fuses all three via the authenticator."""
    if HAS_IMAGE_MODEL:
        assert _result(_upload("image", PNG_BYTES, "image/png"))["signals_used"] == ["image"]

    assert _result(_upload("audio", WAV_BYTES, "audio/wav"))["signals_used"] == ["audio"]
    assert _result(_upload("video", MP4_BYTES, "video/mp4"))["signals_used"] == [
        "image",
        "audio",
        "raw_frames",
        "video_authenticator",
    ]


def test_internal_scores_are_never_returned():
    """The two-tier disclosure rule, asserted at the wire."""
    raw = client.get(
        f"/analyze/{_upload('video', MP4_BYTES, 'video/mp4').json()['analysis_id']}"
    ).text
    for leaked in ("image_score", "raw_frames_score", "fused_score", "internal"):
        assert leaked not in raw


# --- is_mock tracks the models that actually ran -----------------------------


@needs_image_model
def test_real_image_result_is_not_flagged_mock():
    assert _result(_upload("image", PNG_BYTES, "image/png"))["is_mock"] is False


def test_video_result_is_still_flagged_mock():
    """Audio, raw frames and fusion are stubs, so a video verdict is not real."""
    assert _result(_upload("video", MP4_BYTES, "video/mp4"))["is_mock"] is True


def test_audio_result_is_still_flagged_mock():
    assert _result(_upload("audio", WAV_BYTES, "audio/wav"))["is_mock"] is True


# --- Real image model behaviour ---------------------------------------------


@needs_image_model
def test_image_result_always_declares_its_scope():
    """The face-swap blind spot must reach the user, not just the docstring."""
    codes = {item["code"] for item in _result(_upload("image", PNG_BYTES, "image/png"))["evidence"]}
    assert "scope_synthetic_only" in codes


@needs_image_model
def test_image_evidence_claims_no_localisation():
    """A whole-image probe cannot point at a region, so it must not pretend to."""
    for item in _result(_upload("image", PNG_BYTES, "image/png"))["evidence"]:
        assert item["region"] is None
        assert item["start_seconds"] is None


@needs_image_model
def test_verdict_uses_trained_thresholds_not_placeholders():
    probe = load_probe(ARTIFACT_NAME)
    assert probe is not None
    bounds = probe.thresholds
    assert bounds.authentic_below <= bounds.possible_above <= bounds.manipulated_above
    # If these matched the eyeballed placeholders the calibration step silently
    # did nothing (D4).
    assert (
        bounds.authentic_below,
        bounds.possible_above,
        bounds.manipulated_above,
    ) != (0.25, 0.50, 0.75)


@needs_image_model
def test_same_file_scores_deterministically():
    first = _result(_upload("image", PNG_BYTES, "image/png"))
    second = _result(_upload("image", PNG_BYTES, "image/png"))
    assert first["verdict"] == second["verdict"]
    assert first["analysis_id"] != second["analysis_id"]


@needs_image_model
def test_undecodable_image_is_400_not_500():
    response = _upload("image", CORRUPT_PNG, "image/png")
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_missing_artifact_returns_503_and_never_a_fake_verdict(monkeypatch):
    """A model that cannot load must refuse, not guess (D9)."""
    monkeypatch.setattr("app.services.models.image_model.load_probe", lambda _: None)
    response = _upload("image", PNG_BYTES, "image/png")
    assert response.status_code == 503
    assert "not available" in response.json()["detail"]


# --- Upload validation ------------------------------------------------------


def test_oversize_upload_is_rejected():
    oversize = b"\x89PNG\r\n\x1a\n" + b"\x00" * (settings.max_image_bytes + 1)
    assert _upload("image", oversize, "image/png").status_code == 413


def test_wrong_content_type_is_rejected():
    assert _upload("image", MP4_BYTES, "video/mp4").status_code == 415


def test_empty_upload_is_rejected():
    assert _upload("image", b"", "image/png").status_code == 400


def test_unknown_analysis_id_is_404():
    assert client.get("/analyze/does-not-exist").status_code == 404


# --- Report -----------------------------------------------------------------


def test_report_is_501_until_step_7():
    analysis_id = _upload("audio", WAV_BYTES, "audio/wav").json()["analysis_id"]
    assert client.post("/report", json={"analysis_id": analysis_id}).status_code == 501


def test_report_rejects_unknown_analysis():
    assert client.post("/report", json={"analysis_id": "nope"}).status_code == 404
