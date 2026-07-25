# Architecture decisions

Decisions made during the build, with the reasoning. Recorded so later steps
don't silently reverse them.

---

## D1 — Async job + polling, not synchronous responses

**Decided at:** step 1.

`POST /analyze/{kind}` returns `202 Accepted` with `{analysis_id, status,
poll_url}`. The client polls `GET /analyze/{analysis_id}` until status is
`complete` or `failed`.

**Why:** once real models land, a 60-second video runs three networks plus a
fusion step. That does not fit in a request the browser should hold open. The
stub is fast enough to finish inline, but it returns the async contract anyway,
so the frontend built in step 2 doesn't need rewriting in step 5.

**Consequence for step 3:** analysis currently runs inline inside the request
handler. When real inference lands it must move to a worker queue (arq / Celery /
RQ). The response contract does not change when it does.

---

## D2 — Uploaded files are deleted immediately after analysis

**Decided at:** step 1.

The file is streamed to a temp path under a hard size cap, analysed, and deleted
on every exit path including errors (`app/uploads.py`). Only the derived JSON
result is retained, for 24 hours.

**Why:** people upload intimate or defamatory material about themselves to a
tool like this. Not retaining it is both the right default and a genuine
differentiator worth stating on the site.

**Consequences:**
- A PDF report must be generated from the stored result, not by re-reading the
  file. Step 7 has no access to the original media.
- Re-running an analysis requires a re-upload. This is intended.
- Uploads cannot become training data. Anything else needs an explicit,
  separate opt-in consent flow — see D5.

---

## D3 — Two-tier disclosure: evidence yes, numeric scores no

**Decided at:** step 1.

Public API returns a verdict band, a coarse confidence band (`low` / `moderate` /
`high`), and plain-language evidence with timestamps. Per-model numeric scores,
the fused score, and thresholds stay server-side in `InternalScores` and are
never serialised to a client. Enforced by `settings.expose_internal_scores`
(default `False`) and asserted by `test_internal_scores_are_never_returned`.

**Why:** exposing per-model scores turns a free public endpoint into an
evaluation harness — upload, read the score, adjust the fake, repeat until it
passes. Banded confidence rather than a percentage is the same reasoning: a
precise number is a more useful gradient to optimise against, and reads as more
authoritative than the model warrants.

The internal scores are still computed and stored, because the Video
Authenticator (step 6) needs them as inputs and the report generator (step 7)
needs them to write from.

---

## D4 — Verdict is banded, never binary

Four bands: `likely_authentic` / `uncertain` / `possibly_manipulated` /
`likely_manipulated`. There is deliberately no `authentic` or `fake` value.
`uncertain` is a legitimate and common outcome, and must stay visible rather than
being rounded into a confident answer.

Current thresholds (`pipeline._band_verdict`) are placeholders. They must be
re-derived from validation-set ROC curves once real models exist — picking
thresholds by eye is how a detector ends up confidently wrong.

---

## D5 — Open items requiring sign-off

| # | Item | Status |
|---|------|--------|
| 1 | Result disclaimer wording (`services/disclaimer.py`) | **DRAFT — needs approval before step 2 UI copy is finalised** |
| 2 | GenImage CC BY-NC-SA 4.0 non-commercial clause vs. any future monetisation | **Open — see below** |
| 3 | Rate limiting / abuse controls on a free public endpoint | Not started; needed before public deploy |
| 4 | Duration limits (audio 2min, video 60s) | Not enforced yet — needs ffprobe, deferred to step 3-5. Byte caps are the effective limit today |

### On item 2 — the GenImage licence

GenImage is **CC BY-NC-SA 4.0**. Two distinct problems, worth separating:

- **Non-commercial (NC):** a model fine-tuned on GenImage is a derivative work.
  Keeping the site free to users may not be sufficient — NC is generally read as
  barring commercial *purpose*, so ads, a paid tier, an API business, or an
  enterprise offering built on the same weights would all be doubtful. This is
  fine while the project is free, and becomes a real blocker the moment
  monetisation is considered.
- **ShareAlike (SA):** arguably requires derivative works to be released under
  the same licence. Under one reading that reaches the fine-tuned weights
  themselves, which would mean they cannot be kept proprietary.

Neither is a reason to avoid GenImage now — it is the right dataset to start
with, and it's open access. But the image model's training provenance should be
tracked per-dataset from step 3, so a commercially-clean model can be retrained
later without archaeology. ASVspoof and WaveFake licence terms need the same
check before step 4.

This is a flag, not legal advice; a lawyer should confirm before any money is
involved.

---

## D6 — Deferred by design

- **Second stacked meta-layer** above the Video Authenticator — explicitly
  deferred. The authenticator is the top of the ensemble for this phase.
- **Gated datasets** (FaceForensics++, Celeb-DF, DeeperForensics-1.0) — need an
  institutional advisor. The face-manipulation head of the image model is the
  weaker one until these are available; open-access GenImage covers the
  fully-synthetic head in the meantime.
- **C2PA provenance checks, rPPG/blood-flow, SyncNet-style lip-sync** — these
  appear in `README.md` but are not part of the four-model architecture in the
  brief. Not in scope for the current build order.
