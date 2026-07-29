# Frontend — AI Deepfake Protection

React + Vite + TypeScript. Build step 2: upload UI and results dashboard, wired
to the stubbed backend from step 1.

## Running it

The backend must be running first — the frontend has no mock layer of its own,
deliberately (see "No local mocks" below).

```bash
# terminal 1 — backend
cd backend
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173.

Port 5173 is fixed (`strictPort` in `vite.config.ts`) because it is what the
backend's CORS allowlist permits (`backend/app/config.py`). If Vite silently
fell back to 5174, every request would fail preflight with an error that looks
like a network fault rather than a config mismatch.

To point at a backend somewhere else, set `VITE_API_BASE`:

```bash
VITE_API_BASE=http://127.0.0.1:9000 npm run dev
```

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server on :5173 |
| `npm run build` | Typecheck (`tsc -b`) then production build to `dist/` |
| `npm run typecheck` | Types only, no emit |
| `npm run preview` | Serve the built `dist/` |

## Layout

```
src/
  api/
    types.ts      Mirrors backend/app/schemas.py — public half only
    client.ts     fetch wrappers + the 202/poll loop
    media.ts      File classification and pre-flight size/type checks
  components/
    UploadPanel.tsx        Drag-drop + picker, one file at a time
    AnalysisProgress.tsx   Polling state
    ResultsDashboard.tsx   Verdict, evidence, signals, disclaimer
    EvidenceList.tsx       The "why", in plain language
  copy.ts         All verdict/confidence/signal wording
  App.tsx         Shell + upload→poll→result state machine
  styles.css
```

## Constraints this code is holding to

These are product decisions from `docs/DECISIONS.md`, not stylistic preferences.
Changing any of them is a decision to re-open, not a refactor.

**No numeric scores, anywhere.** The API doesn't return per-model scores, the
fused score, or thresholds (D3), and nothing here should try to reconstruct or
approximate one. A precise number reads as more authoritative than the models
warrant, and gives anyone tuning a fake a gradient to optimise against. The UI
shows a verdict band, a confidence band, and prose.

**No binary real/fake.** Four bands, and `uncertain` is a real outcome that
stays visible rather than being rounded toward a confident answer (D4).

**The disclaimer comes from the server.** `result.disclaimer` is rendered
verbatim. It is authored in `backend/app/services/disclaimer.py` so one approved
wording reaches the dashboard and the PDF report together and cannot drift.
Do not hardcode disclaimer copy in a component.

**No history, no accounts.** One file, one result on screen. Uploads are deleted
immediately after analysis (D2); a UI that accumulated someone's past results
would quietly undercut that.

**The mock banner is not decorative.** While `is_mock` is true the result is a
placeholder derived from a hash of the file's bytes. It has to be impossible to
mistake for a real finding about a real person's file. It disappears on its own
when `MODELS_ARE_STUBS` is set false in step 6.

### No local mocks

There is intentionally no MSW/fixture layer. The whole point of step 2 in the
build order is a working full-stack demo — a frontend that renders beautifully
against invented data proves nothing about the contract. It talks to the real
backend or it shows an error.

## Known gaps (deliberate, for this step)

- **Cancellation is client-side only.** Navigating away or hitting the poll
  timeout stops the UI; it doesn't stop server work. Worth a `DELETE /analyze/{id}`
  once inference actually costs GPU time (step 3).
- **Duration limits aren't checked here.** Byte caps are, from `GET /limits`.
  Duration (audio 2min / video 60s) isn't enforced on either side yet —
  D5 item 4, deferred to steps 3–5.
- **No evidence overlay.** Findings carry timestamps and coarse regions
  ("around the jaw and hairline") but there's no player scrubbing to them or
  heatmap over the image. That needs real localisation data from real models.
- **Not internationalised.** All copy is English, in `src/copy.ts`.
- **No frontend tests.** Backend has pytest coverage; this step was verified
  manually against the running backend (upload → poll → result, unsupported
  type, oversize, backend-down, report 501). Component tests are worth adding
  before the copy stops moving.
