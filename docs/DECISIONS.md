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
never serialised to a client. Asserted by
`test_internal_scores_are_never_returned`.

**Amended post-step-3:** this originally claimed enforcement came from
`settings.expose_internal_scores` (default `False`). Nothing ever read that
setting — the only function that consulted it, `pipeline.peek_internal`, was never
called from any route. A documented control that does not run is worse than no
control, because it stops anyone looking for a real one, so the flag has been
removed rather than wired up.

The actual enforcement is structural, and stronger than the flag would have been:
`AnalysisStatusResponse` has no field capable of carrying these values. There is
no code path that could serialise them, and nothing to misconfigure in
production. Adding a numeric score to a response is a schema change, which is a
reversal of this decision rather than a config change — which is exactly the
friction it should have.

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

Thresholds were placeholders through steps 1-2. **Resolved for the image model at
step 3** — see D11. Audio, raw frames and the authenticator still use the
eyeballed placeholders in `pipeline.PLACEHOLDER_THRESHOLDS` and must be
recalibrated the same way as each becomes real.

---

## D5 — Open items requiring sign-off

| # | Item | Status |
|---|------|--------|
| 1 | Result disclaimer wording (`services/disclaimer.py`) | **PROVISIONAL — approved for development at step 2; still needs a real review before public deploy** |
| 2 | GenImage CC BY-NC-SA 4.0 non-commercial clause vs. any future monetisation | **Live constraint as of step 3 — trained weights now exist and are published. See below and D12** |
| 3 | Rate limiting / abuse controls on a free public endpoint | **Done — see D15.** In-process only; needs Redis before running more than one worker |
| 4 | Duration limits (audio 2min, video 60s) | **Done — see D16.** Enforced with PyAV |

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

---

## D10 — Frozen CLIP backbone with a linear probe, not a fine-tuned CNN

**Decided at:** step 3.

The image model is a logistic regression over the 512-d output of a frozen CLIP
ViT-B/32 vision tower. The backbone is not fine-tuned. Training code in `ml/`.

**Why:** fine-tuning end-to-end on a fixed generator set produces a detector that
learns each generator's fingerprint. It scores better on those generators and
degrades sharply on new ones. Every image a user uploads comes from a generator
newer than the training data, so in-distribution accuracy is close to worthless
as a predictor of field behaviour, and cross-generator generalisation is the
metric that matters. A linear probe on frozen CLIP features holds up
substantially better on unseen generators (Ojha et al., CVPR 2023), and
`ml/evaluate.py` measures that with a leave-one-generator-out study rather than
assuming it.

Two side effects, both convenient rather than motivating: the pipeline trains on
a CPU in under an hour because features are extracted once and cached, and the
head is 513 floats — small enough to commit as JSON and serve with a dot product,
so scikit-learn stays out of the runtime.

**Consequence:** the backbone is a downloaded dependency (~350MB from Hugging
Face on first inference). Production deployments must pre-fetch it rather than
fetching on a user's first request.

---

## D11 — Verdict thresholds are derived from the validation ROC, per detector

**Decided at:** step 3. Discharges the obligation D4 left open.

`ml/train.py:derive_thresholds` sets the three band boundaries from the
validation ROC, each pinned to an explicit error rate rather than chosen by eye:

- `authentic_below` — the 5th percentile of *fake* scores, so calling something
  "likely authentic" misses at most ~5% of fakes.
- `manipulated_above` — the 95th percentile of *real* scores, so a genuine photo
  is branded "likely manipulated" at most ~5% of the time.
- `possible_above` — the equal-error point, splitting the middle band.

Thresholds ship inside the model artifact and travel with it, because they are
only meaningful relative to the score distribution of the detector that produced
them. `DetectorOutput.thresholds` carries them up to the pipeline;
`PLACEHOLDER_THRESHOLDS` applies only to detectors that are still stubs.

**Why per-detector rather than global:** a single 0.25/0.5/0.75 ladder assumes
every model's scores are calibrated identically. They are not, and a shared
ladder silently mis-bands whichever model deviates.

### Correction (post-step-3): `authentic_below` was measured on the wrong population

The three thresholds were all derived from the validation split — which contains
only the generators the probe trained on. For `manipulated_above` and
`possible_above` that is fine: the real-image distribution does not change when a
new generator ships, and the EER point is a property of the score scale.

For `authentic_below` it was wrong, and wrong in the direction that hurts. Its
whole purpose is to bound how often the tool says "no signs of manipulation"
about something fake, and it was computed from fakes the model had effectively
already seen. Measured against a generator excluded from training, the miss rate
at 0.5 reached **63%** (VQDM). Since every image a user uploads comes from a
generator newer than the training data — the premise D10 is built on — the
in-distribution quantile was describing a model that does not exist.

`ml/train.py:held_out_fake_scores` now pools fake scores from leave-one-generator-out
probes and takes the 5th percentile of *those*. The effect is large:

| | `authentic_below` |
|---|---|
| In-distribution (shipped through step 3) | 0.2571 |
| Held-out generators (correct) | **0.0297** |

Roughly 9x more permissive than it claimed to be. The artifact records both
figures plus the realised miss rate, so the size of the correction stays visible
rather than living only in this file.

Fewer files now reach `likely_authentic`, which is the point. The band that tells
someone their video is probably fine is the one that has to be hardest to enter.

The same test also drove `C_GRID`: selection had landed on `C=100.0`, the top of
the old sweep, meaning the grid rather than the data chose it — and chose the
least-regularised end, the one most able to memorise a generator's fingerprint.
Extended to 3000; the curve now peaks in the interior at **C=300** and `train.py`
warns if selection ever hits a boundary again.

**Still not covered:** the held-out study measures generalisation to generators of
roughly the same era. It cannot measure generalisation to a 2026 generator, because
none is in the dataset. The user-facing evidence now says so directly rather than
implying the calibration covers it.

---

## D12 — The trained head is committed, and carries the dataset's licence

**Decided at:** step 3.

`models/image_synthetic_probe.json` is committed to the repo, along with its
metrics and generalisation reports.

**Why commit it:** at 513 floats it is small, diffable and inspectable, a clone
produces real results without a 35-minute training run, and publishing the
metrics next to the weights keeps the model's claims checkable against the
artifact actually in use.

**The licence consequence, which is real:** the head is trained on Tiny-GenImage
(CC BY-NC-SA 4.0) and is therefore a derivative work. Committing it to a public
repository is redistribution. On the cautious reading it inherits both clauses,
so the artifact is **released under CC BY-NC-SA 4.0** and attributed, separately
from whatever licence the rest of the repo carries. NonCommercial then binds any
future ads, paid tier, or API business built on these coefficients — keeping the
site free to users is probably not sufficient, since NC restricts commercial
purpose rather than just direct charging.

Every artifact embeds a `provenance` block (dataset, licence, generators, git
SHA, date) so a commercially-clean retrain is a known, bounded job rather than
archaeology. Good-faith reading, not legal advice; confirm with a lawyer before
any money is involved.

---

## D13 — A model that cannot load refuses; it never falls back to a stub

**Decided at:** step 3.

Missing or malformed artifact → `ModelUnavailable` → HTTP 503. Undecodable
upload → 400. Neither path produces a verdict.

**Why:** a silent fallback to a placeholder score is indistinguishable from a
real answer at the API boundary, and the person reading "likely manipulated"
acts on it either way. The failure mode being prevented is someone being
disbelieved because a weights file was missing from a deploy.

The same reasoning drives `is_stub` moving from one global `MODELS_ARE_STUBS`
flag onto each `DetectorOutput`. Between steps 3 and 6 a single boolean is
necessarily wrong about something; `AnalysisResult.is_mock` is now true if *any*
contributing detector was a placeholder, so a video — whose audio, frames and
fusion are all still stubs — stays correctly marked even though the image model
is real.

---

## D14 — Evidence may not claim more than the model can see

**Decided at:** step 3.

The step-1 stub emitted findings like "the edges of the face don't blend
naturally into the hair and neck… around the jaw and hairline". A linear probe
over a global image embedding produces one scalar. It has no spatial
localisation, no per-region attribution, and no notion of a jaw. Those strings
were fabricated and were removed when the real model landed; the replacements
describe the strength and direction of a whole-image signal and nothing else.
A test asserts every image finding leaves `region` and `start_seconds` null.

Every image result also carries a permanent `scope_synthetic_only` finding
stating that the check cannot detect face swaps or localised edits. That gap is
not an edge case, it is half the intended model (head (a), blocked on gated
datasets — D6). A confident "no signs of manipulation" on a face-swapped photo
is precisely the failure that gets someone disbelieved, so the limit is stated
to the user rather than left in a docstring.

**Consequence for step 7:** when an LLM rewrites these summaries into report
prose, it must be constrained to the evidence it is given. A model asked to
"explain why this looks fake" will invent plausible detail, which is this same
failure with better grammar.

---

## D15 — Rate limiting is middleware, with two windows and no trust in XFF

**Decided at:** post-step-3. Closes D5 item 3.

Sliding-window limiter in `app/ratelimit.py`. `POST /analyze/*` and `/report` get
5/min and 30/hour; everything else gets 300/min.

**Middleware rather than a per-route dependency.** A dependency has to be
remembered on every new route, and the routes added in steps 4 and 7 would be
exactly the expensive ones. Middleware cannot be forgotten.

**Two windows, because one is always wrong.** A burst limit alone misses a slow
drip that still exhausts the inference path; a sustained limit alone permits a
spike that takes the service down before it trips.

**Reads are limited far more generously than writes.** The frontend polls once a
second while a job runs (D1). A limit tuned for uploads would break normal use,
and the usual response to that is to disable the control rather than tune it.

**`X-Forwarded-For` is ignored by default.** Any client can send it, so
honouring it on a directly-exposed server makes the limiter bypassable in one
line. Behind a trusted proxy, enable `trust_forwarded_for` — and note the code
takes the *rightmost* entry, since the leftmost is client-supplied.

**A sliding rather than fixed window**, because a fixed window lets a client send
its whole allowance in the last second of one window and again in the first
second of the next — double the intended rate, precisely when it hurts.

**Rejected requests do not extend the window.** Otherwise a client that keeps
retrying locks itself out indefinitely, which punishes an impatient user far
more than an attacker.

**Middleware order is load-bearing and reads backwards.** `add_middleware`
inserts at the front of the stack, so the *last* registered is the *outermost*.
CORS must therefore be added after the limiter, so it wraps it and still attaches
headers to a 429 the limiter returns without calling through. Get this wrong and
a rate-limited browser client sees an opaque CORS failure instead of the "please
wait" message. Asserted by `test_rate_limited_response_still_has_cors_headers`.

**Consequences:**
- State is per process, like the job store. Two workers means two independent
  allowances; both need Redis together before scaling out.
- The client dictionary is bounded (`MAX_TRACKED_CLIENTS`) and evicted, so a
  spoofed-IP flood cannot turn the limiter itself into the memory exhaustion it
  exists to prevent.
- Not a defence against a distributed flood. That needs a CDN or WAF in front.

---

## D16 — Duration limits via PyAV, and the container is trusted over the header

**Decided at:** post-step-3. Closes D5 item 4.

Audio capped at 2 minutes, video at 60 seconds, enforced in
`app/media_probe.py` after the file is staged.

**Why byte caps were not enough:** they are a poor proxy for compute. A 25MB
H.264 file can be ten minutes long, and step 5 will run three models across its
frames. Duration is what actually bounds the work.

**PyAV rather than shelling out to ffprobe:** it ships its own ffmpeg libraries,
so there is no system binary to install, no subprocess to sanitise, and no
parsing of another program's stdout. Step 5 needs PyAV for frame extraction
regardless.

**The check runs after staging, not mid-stream**, because the metadata carrying
duration can sit at either end of the file depending on how it was written. The
byte cap is what bounds the damage until that point.

**Stream type is validated too**, which is a stronger check than Content-Type:
the client controls the header, but not the container. An upload to
`/analyze/audio` must actually contain an audio stream.

**Unknown duration is rejected, not accepted.** A file whose length cannot be
determined could be arbitrarily long, which is the exact thing being bounded.

**Security consequence:** this hands attacker-controlled bytes to ffmpeg's
demuxers, which have a real CVE history. The exposure is unavoidable — analysing
uploaded media is the product — but it argues for keeping PyAV patched and, before
a public deploy, running analysis in a sandbox rather than in the API process.

---

## D17 — Resource limits belong above the parser, not below it

**Decided at:** post-step-3, after a security review of steps 1-3.

`uploads.py` had claimed since step 1 that its size cap was applied "while
streaming, not after", and warned in the same docstring that reading the whole
body first "would let a single request exhaust memory or disk regardless of the
configured limit". That warning was an accurate description of what the code was
doing.

With `file: UploadFile = File(...)`, Starlette parses the entire multipart body
before the endpoint function is entered, spooling it to a temp file once it
outgrows `spool_max_size`. The loop in `staged_upload` was copying from that temp
file — it never touched the network. Measured: a 40MB body to `/analyze/image`
(cap 5MB) returned a correct 413 with all 41,943,048 bytes already on disk. The
413 was never the problem. The write behind it was, on an unauthenticated
endpoint, with no authentication and 5 requests per minute per IP as the only
brake.

**The ceiling moved into ASGI middleware** (`app/bodylimit.py`), which is the
only layer that sees `receive` before the parser does. `BaseHTTPMiddleware`
cannot do this — it is handed a `Request` whose body is read downstream.

**Per-route, not global.** The per-kind caps differ by 5x. A single ceiling at
the video cap would leave `/analyze/image` five times cheaper to attack than its
own configured limit permits, and everything that is not an upload route gets
64KB, since `/report` sends about seventy bytes of JSON.

**`Content-Length` is rejected outright; chunked bodies are counted and cut off.**
The second path cannot reject up front because the size is not knowable, and
truncation alone would be a worse bug than the one being fixed: a short file can
still decode, and a *verdict on partial media* is a wrong answer rather than an
error. So the request is flagged on `scope["state"]` and `staged_upload` refuses
it. In a tool people act on, failing loudly beats answering quietly.

**The old cap stays** as a second line of defence. It is the only check that knows
the size of the file rather than the size of the request carrying it.

### Decoded size is a separate limit from encoded size

A byte cap does not bound a decoder. A 140KB PNG decodes to 144 million pixels,
and `.convert("RGB")` on the way into CLIP triples that. PIL's own guard warns —
but still allocates — below 2x its default, and above it raises
`DecompressionBombError`, which is not an `OSError` and so escaped the handler
that was catching decode failures, surfacing as a 500.

Dimensions are now checked after `Image.open` (header only) and before `load()`,
against `settings.max_image_pixels = 40M` — above any real camera, well below a
bomb. Bombs are 413, not 400: the file is valid, it is just more than this service
will spend memory on.

### A declared Content-Type does not get to choose a decoder

The `ALLOWED_CONTENT_TYPES` allowlist claimed anything unlisted was "rejected
rather than best-effort decoded". It checked a header the client writes. PIL then
opened whatever the bytes actually were, so GIF, TIFF and BMP uploads declared as
`image/png` were all accepted — the real decoder surface was every format PIL
supports, rather than the three that were reviewed.

`image.format` is now checked against the same three. This is the principle
`media_probe.py` already applied to audio and video containers at D16 — *the
client controls the header, but not the container* — finally applied to images
too.

### Retention needs a sweep, not a lazy check

`result_ttl_seconds` was only ever evaluated inside `store.get`. The normal client
polls until its result is ready, reads it, and never returns — so nothing
triggered the check and every analysis retained its record, including its
`InternalScores`, for the process lifetime. `purge_expired` now runs on a timer
from the app lifespan, and `create` enforces a hard ceiling for bursts that
outrun the timer. Expired records are evicted before live ones, since dropping a
live one loses a result someone may still be polling for.

### Clearing the rate-limit table is worse than the growth it prevents

When every tracked client was live, `_evict` called `self._hits.clear()`. That
resets every client's allowance — including the allowance of whoever caused the
flood — which converts a memory bound into an attacker-triggerable reset of the
control itself. It now drops the least-recently-active tenth: bounded, and a
client mid-flood is by definition among the most recently active, so its own
counters are the last to go.

### Everything else in that review

- **The backbone is pinned to a commit.** Every Python dependency was pinned
  exactly, then 350MB of weights were fetched from a moving tag. A substituted
  backbone does not fail loudly — the probe still returns a number and the API
  still returns a verdict; the coefficients simply no longer mean anything.
  `CLIP_REVISION` is asserted identical in `ml/config.py` and
  `services/models/artifacts.py`, because a train/serve mismatch here is silent.
- **`except ValueError` narrowed to `UndecodableUpload`.** The broad version
  reflected `str(exc)` from any `ValueError` — including ones raised inside PIL,
  numpy or torch, whose text can carry filesystem paths — into a 400 body, and
  let an internal invariant failure present as a user error.
- **The container is probed once.** `_has_audio_track` returned a hardcoded `True`
  with a TODO to "probe the container with ffprobe", while PyAV was already
  probing every video upload two frames up the stack and reporting exactly that.
  A silent video therefore had a hash-derived audio stub folded into fusion at
  weight 0.20 instead of being renormalised away, and the `False` branch the
  authenticator implements was unreachable. The probe result now travels on
  `StagedUpload`, which also means attacker bytes reach ffmpeg's demuxers once per
  upload instead of twice.
- **Stub evidence codes are namespaced per model.** Both the audio and raw-frames
  stubs emitted `code="model_not_implemented"`, and `code` is a finding's
  identity — the dashboard keys its list on it — so every video result rendered
  two items with the same React key.
- **Dead code deleted:** `pipeline.execute_job` (a background entry point for a
  queue that was never wired up) and `pipeline.peek_internal` (see D3).

**What this review did not fix, and remains outstanding:** analysis still runs
inline in the request handler rather than on a worker queue, the job store and
rate limiter are both still per-process, and media parsing still happens inside
the API process rather than a sandbox. All three are known, documented, and
unchanged by this pass.

---

## D7 — Frontend renders bands and prose only, and owns no disclaimer copy

**Decided at:** step 2.

The results dashboard displays a verdict band, a confidence band, evidence
sentences, and which checks ran. It has no code path that renders a number, and
`frontend/src/api/types.ts` deliberately has no counterpart to `InternalScores` —
so there is nothing to accidentally bind to if the server ever over-serialises.
Which checks ran is disclosed; what they scored is not.

`result.disclaimer` is rendered verbatim from the API rather than being stored in
the frontend, so the wording approved once in `services/disclaimer.py` reaches the
dashboard and the step 7 PDF together. Verdict wording itself lives in one module
(`frontend/src/copy.ts`), for the same reason: this copy is where the product's
honesty about uncertainty either holds or doesn't, and it should be reviewable in
one place rather than scattered across components.

**Consequence:** a design change that wants to show "87% likely fake" is a
reversal of D3, not a UI tweak.

---

## D8 — No mock data layer in the frontend

**Decided at:** step 2.

No MSW, no fixtures. The frontend talks to the running backend or it shows an
error state.

**Why:** step 2 exists to prove the full stack works end to end before any ML
lands. A frontend rendering against invented data would prove the CSS works and
nothing about the contract — and the contract (202 + poll, banded fields,
server-supplied disclaimer) is exactly what steps 3-7 build on. Verified
manually against the live backend: upload → poll → result, unsupported type,
oversize, backend unreachable, and `POST /report` returning 501.

**Consequence:** frontend component tests, when added, will need either a test
backend or a fixture layer introduced only for tests — not for development.

---

## D9 — The stub banner is load-bearing

**Decided at:** step 2.

While `is_mock` is true, the dashboard shows an unmissable banner stating the
result is a placeholder and says nothing about the uploaded file.

**Why:** the stub score is a hash of the file's bytes, so it is stable per file
and looks entirely plausible — a person could upload a video of themselves and
read "Strong signs of manipulation" off a hash. That is precisely the harm this
project exists to reduce. The banner clears itself when the last stub is replaced;
it is not a manual cleanup step someone has to remember.

**Amended at step 3:** the single global `MODELS_ARE_STUBS` flag this originally
described no longer exists. `is_stub` lives on each `DetectorOutput` and
`AnalysisResult.is_mock` is true if *any* contributing detector was a placeholder —
see D13. The banner therefore clears itself per media kind as each model becomes
real, rather than all at once in step 6: image results already show no banner,
while audio and video still do.

