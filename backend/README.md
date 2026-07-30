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
  main.py              FastAPI app, middleware order, health/limits, expiry sweep
  config.py            Upload limits, retention, pixel ceiling, rate limits
  schemas.py           Public vs internal result split
  errors.py            Domain exceptions carrying user-safe messages
  bodylimit.py         Request-body ceiling, enforced before the body is parsed
  ratelimit.py         Per-client sliding-window limiter (middleware)
  media_probe.py       Container duration + stream-type validation (PyAV)
  storage.py           Job store (in-memory; swap for Redis before scaling)
  uploads.py           Per-file size cap + guaranteed temp-file cleanup
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

**Resource limits sit above the parser, not below it.** `bodylimit.py` runs as
ASGI middleware because that is the only layer that sees `receive` before FastAPI
parses the multipart body — and parsing is what writes the body to disk. A cap
checked inside the endpoint is checked after the damage. If you add an endpoint
that accepts a body, add it to `_UPLOAD_PATHS` or it gets the 64KB default.
Likewise, decoded size is a separate limit from encoded size: a 140KB PNG can
decode to 144 megapixels, so `settings.max_image_pixels` is checked before
`load()`. See `docs/DECISIONS.md` D17.

**Middleware order is load-bearing and reads backwards.** `add_middleware`
inserts at the front, so the last registered is the outermost. CORS must stay
last, or a browser client that trips the rate limiter or the body ceiling sees an
opaque CORS failure instead of the message explaining what happened. Two tests
guard this.

## Known gaps at step 3

- **The image model detects fully AI-generated images only.** It does not detect
  face swaps or localised edits to real photographs — that head needs gated
  datasets (D6). Every image result states this to the user.
- **On video, the image model is still a stub.** Scoring frames needs frame
  extraction, which arrives with ffmpeg/PyAV in step 5
  (`ImageModel.analyze_video_frames`).
- **Inference runs in FastAPI's threadpool, not a worker queue.** Sync path
  operations already run off the event loop, so CPU inference does not block it,
  and a single image is fast enough that a queue would be premature. This stops
  being true in step 5, when one video upload runs three models over many frames
  — arq/Celery/RQ is required before then (D1).
- Job store **and rate-limiter state** are in-process — single worker only. Two
  workers means two independent rate-limit allowances. Both need Redis together
  before scaling out.
- **First image request after start takes ~19s** while the CLIP backbone loads;
  warm requests are ~500ms. Pre-fetch the backbone before a public deploy.

## Abuse controls

**Rate limiting** (`app/ratelimit.py`) — middleware, not a per-route dependency,
so a route added later cannot forget it. Two sliding windows per client, because
a burst limit alone misses a slow drip and a sustained limit alone lets through a
damaging spike.

| Scope | Default |
|---|---|
| `POST /analyze/*`, `/report` | 5/min **and** 30/hour |
| Everything else (polling, `/health`, `/limits`) | 300/min |

Reads are deliberately generous: the frontend polls ~1/s while a job runs (D1).
Returns `429` with `Retry-After`. Configure via `ADP_ANALYZE_BURST_LIMIT`,
`ADP_ANALYZE_SUSTAINED_LIMIT`, `ADP_READ_LIMIT`, `ADP_RATE_LIMIT_ENABLED`.

`X-Forwarded-For` is **ignored** unless `ADP_TRUST_FORWARDED_FOR=true`. Any
client can send that header, so honouring it on a directly-exposed server makes
the limiter bypassable with one curl flag. Enable it only behind a proxy that
overwrites the header.

**Duration limits** (`app/media_probe.py`) — audio 2min, video 60s, enforced with
PyAV. Byte caps are a poor proxy for cost: a 25MB H.264 file can be ten minutes
long. Over-long media returns `413`; unparseable containers return `400`.

The probe also checks the **stream type**, which is stronger than the declared
Content-Type: an upload to `/analyze/audio` must actually contain an audio
stream, so a renamed file is caught even though the client controls the header.
Media whose duration cannot be determined is rejected rather than waved through —
it could be arbitrarily long, which is the thing being bounded.
