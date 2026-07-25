"""Smoke tests for the step-1 skeleton.

These assert the API *contract* the frontend will be built against in step 2 —
not detection quality, which does not exist yet. They should keep passing
unchanged as the stubs are replaced by real models in steps 3-6; if one starts
failing then, the contract broke and the frontend broke with it.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048
WAV_BYTES = b"RIFF" + b"\x00" * 2048
MP4_BYTES = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 2048


def _upload(kind: str, data: bytes, content_type: str, name: str = "sample"):
    return client.post(
        f"/analyze/{kind}",
        files={"file": (name, io.BytesIO(data), content_type)},
    )


def test_health_reports_stub_state():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models_are_stubs"] is True


def test_limits_match_config():
    body = client.get("/limits").json()
    assert body["image"]["max_bytes"] == settings.max_image_bytes
    assert body["video"]["max_seconds"] == settings.max_video_seconds


@pytest.mark.parametrize(
    ("kind", "data", "ctype"),
    [
        ("image", PNG_BYTES, "image/png"),
        ("audio", WAV_BYTES, "audio/wav"),
        ("video", MP4_BYTES, "video/mp4"),
    ],
)
def test_analyze_accepts_and_completes(kind, data, ctype):
    response = _upload(kind, data, ctype)
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "complete"

    polled = client.get(f"/analyze/{accepted['analysis_id']}")
    assert polled.status_code == 200
    result = polled.json()["result"]

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
    assert result["is_mock"] is True
    assert result["media_deleted"] is True


def test_routing_matches_architecture():
    """Image/audio use one model; video fuses all three via the authenticator."""
    image_id = _upload("image", PNG_BYTES, "image/png").json()["analysis_id"]
    assert client.get(f"/analyze/{image_id}").json()["result"]["signals_used"] == ["image"]

    audio_id = _upload("audio", WAV_BYTES, "audio/wav").json()["analysis_id"]
    assert client.get(f"/analyze/{audio_id}").json()["result"]["signals_used"] == ["audio"]

    video_id = _upload("video", MP4_BYTES, "video/mp4").json()["analysis_id"]
    assert client.get(f"/analyze/{video_id}").json()["result"]["signals_used"] == [
        "image",
        "audio",
        "raw_frames",
        "video_authenticator",
    ]


def test_internal_scores_are_never_returned():
    """The two-tier disclosure rule, asserted at the wire."""
    analysis_id = _upload("video", MP4_BYTES, "video/mp4").json()["analysis_id"]
    raw = client.get(f"/analyze/{analysis_id}").text
    for leaked in ("image_score", "raw_frames_score", "fused_score", "internal"):
        assert leaked not in raw


def test_same_file_scores_deterministically():
    first = _upload("image", PNG_BYTES, "image/png").json()["analysis_id"]
    second = _upload("image", PNG_BYTES, "image/png").json()["analysis_id"]
    a = client.get(f"/analyze/{first}").json()["result"]
    b = client.get(f"/analyze/{second}").json()["result"]
    assert a["verdict"] == b["verdict"]
    assert a["analysis_id"] != b["analysis_id"]


def test_oversize_upload_is_rejected():
    oversize = b"\x89PNG\r\n\x1a\n" + b"\x00" * (settings.max_image_bytes + 1)
    assert _upload("image", oversize, "image/png").status_code == 413


def test_wrong_content_type_is_rejected():
    assert _upload("image", MP4_BYTES, "video/mp4").status_code == 415


def test_empty_upload_is_rejected():
    assert _upload("image", b"", "image/png").status_code == 400


def test_unknown_analysis_id_is_404():
    assert client.get("/analyze/does-not-exist").status_code == 404


def test_report_is_501_until_step_7():
    analysis_id = _upload("image", PNG_BYTES, "image/png").json()["analysis_id"]
    response = client.post("/report", json={"analysis_id": analysis_id})
    assert response.status_code == 501


def test_report_rejects_unknown_analysis():
    assert client.post("/report", json={"analysis_id": "nope"}).status_code == 404
