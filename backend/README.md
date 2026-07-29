# Backend — AI Deepfake Protection API

FastAPI service. **Build step 3 of 7.**

| Detector | State |
|---|---|
| Image | **Real** — frozen CLIP ViT-B/32 + linear probe. Fully-synthetic detection only |
| Audio | Stub (step 4) |
| Raw Frames | Stub (step 5) |
| Video Authenticator | Stub (step 6) |

`GET /health` reports this per model. `is_mock` on a result is true if *any*
detector that contributed to it was a stub — so image results are now real,
while audio and video results are still placeholders.

The image model needs `models/image_synthetic_probe.json` (committed) and
downloads its CLIP backbone (~350MB) from Hugging Face on first inference. If the
artifact is missing, `/analyze/image` returns **503** — it does not fall back to
a stub score (D13).

## Run

```bash
cd backend && python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

```bash
cd backend && ./.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

Interactive docs at http://localhost:8000/docs

```bash
cd backend && ./.venv/Scripts/python.exe -m pytest tests -q
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/analyze/image` | Image upload, max 5MB |
| `POST` | `/analyze/audio` | Audio upload, max 10MB |
| `POST` | `/analyze/video` | Video upload, max 25MB |
| `GET`  | `/analyze/{analysis_id}` | Poll for the result |
| `POST` | `/report` | PDF evidence report — **501 until step 7** |
| `GET`  | `/health` | Liveness + whether models are still stubs |
| `GET`  | `/limits` | Upload limits, so the client can validate before sending |

All three analyse endpoints return `202` with `{analysis_id, status, poll_url}`.
See [docs/DECISIONS.md](../docs/DECISIONS.md#d1--async-job--polling-not-synchronous-responses) for why they don't return the result inline.

## Layout

```
app/
  main.py              FastAPI app, CORS, health/limits
  config.py            Upload limits, retention, disclosure flags
  schemas.py           Public vs internal result split
  storage.py           Job store (in-memory; swap for Redis before scaling)
  uploads.py           Streaming size cap + guaranteed temp-file cleanup
  routers/
    analyze.py         The three upload endpoints + polling
    report.py          Evidence report endpoint
  services/
    pipeline.py        Routing, verdict banding, public/internal split
    disclaimer.py      Disclaimer copy (PROVISIONAL — needs sign-off)
    report_service.py  Report generation (step 7)
    models/
      base.py                 Shared Detector contract; score = P(manipulated)
      artifacts.py            Artifact loading + frozen CLIP backbone (lazy)
      image_model.py          Step 3 — REAL
      audio_model.py          Step 4
      raw_frames_model.py     Step 5
      video_authenticator.py  Step 6 — weighted-average fallback for now
```

Training code lives in [`ml/`](../ml/README.md); artifacts in
[`models/`](../models/README.md). Nothing in the request path imports from `ml/`.

## Routing

Follows the four-model architecture exactly:

- **video** → Image + Audio + Raw Frames → Video Authenticator → final verdict
- **image** → Image model's output is the final verdict
- **audio** → Audio model's output is the final verdict

Asserted by `test_routing_matches_architecture`.

## Two things to know before editing

**Numeric scores never leave the server.** Per-model scores live in
`InternalScores` and are omitted from every response. `test_internal_scores_are_never_returned`
asserts this at the wire — if you add a field, keep it out of `AnalysisResult`.

**Uploads are deleted the moment analysis finishes.** `uploads.staged_upload` is
a context manager that unlinks the temp file on every exit path. Don't copy the
path anywhere that outlives it.

## Known gaps at step 3

- **The image model detects fully AI-generated images only.** It does not detect
  face swaps or localised edits to real photographs — that head needs gated
  datasets (D6). Every image result states this to the user.
- **On video, the image model is still a stub.** Scoring frames needs frame
  extraction, which arrives with ffmpeg/PyAV in step 5
  (`ImageModel.analyze_video_frames`).
- Duration limits (audio 2min / video 60s) are **not enforced** — needs ffprobe.
  Byte caps are the effective limit today.
- **Inference runs in FastAPI's threadpool, not a worker queue.** Sync path
  operations already run off the event loop, so CPU inference does not block it,
  and a single image is fast enough that a queue would be premature. This stops
  being true in step 5, when one video upload runs three models over many frames
  — arq/Celery/RQ is required before then (D1).
- Job store is an in-process dict — single worker only.
- No rate limiting. Required before any public deploy.
