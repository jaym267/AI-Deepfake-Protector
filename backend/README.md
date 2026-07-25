# Backend — AI Deepfake Protection API

FastAPI service. **Build step 1 of 7: skeleton with stub detection models.**
Every score is currently mock. `is_mock: true` on every result and
`models_are_stubs: true` on `/health` mark this unambiguously.

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
    disclaimer.py      Disclaimer copy (DRAFT — needs sign-off)
    report_service.py  Report generation (step 7)
    models/
      base.py                 Shared Detector contract; score = P(manipulated)
      image_model.py          Step 3
      audio_model.py          Step 4
      raw_frames_model.py     Step 5
      video_authenticator.py  Step 6 — weighted-average fallback for now
```

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

## Known gaps at step 1

- Duration limits (audio 2min / video 60s) are **not enforced** — needs ffprobe.
  Byte caps are the effective limit today.
- Analysis runs inline in the request handler. Must move to a worker queue
  before real inference lands in step 3.
- Job store is an in-process dict — single worker only.
- No rate limiting. Required before any public deploy.
