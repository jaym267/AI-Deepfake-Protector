# AI Deepfake Protection

A public, free-to-use website that lets anyone upload a video and get a clear,
explainable assessment of whether it's likely AI-generated (deepfaked) or authentic.

## Problem

AI-generated fake videos are increasingly convincing and increasingly believed.
A single viral deepfake can damage someone's career or reputation before anyone
has a chance to verify it. Existing detection tools are almost all built for
enterprises — fraud teams, banks, law enforcement — priced and designed for
institutions, not for an individual who just got sent a suspicious video and
needs an answer.

## Who this is for

Anyone — not limited to hiring/interviews. A person trying to verify a video
sent to them, a journalist doing quick verification, someone defending their
own reputation against a fake video circulating about them.

## What makes this different from existing tools

- **Free and built for individuals**, not enterprise/KYC pricing tiers
- **Reputation-protection framing**, not just financial-fraud framing
- **Plain-language explainability** — shows *why* something looks fake
  (e.g. "lip movement doesn't match audio in these 3 seconds"), not a
  jargon-heavy forensic PDF
- **Transparent about uncertainty** — a probability + evidence, not a false
  binary "100% REAL" badge

## Detection approach: ensemble of signals

No single model is reliable enough on its own. The site combines multiple
independent signals into one report:

1. **Spatial/frame-level artifacts** — CNN classifier trained on datasets like
   FaceForensics++ / Celeb-DF / DFDC, looking for blending inconsistencies,
   unnatural texture, warped features
2. **Temporal consistency** — sequence model checking for flicker, unnatural
   blink rate, inconsistent lighting frame-to-frame
3. **Audio-visual sync** — lip movement vs. phoneme timing mismatch detection
   (SyncNet-style), catches face-swap + voice-clone combos
4. **Metadata/provenance (C2PA)** — checks for Content Credentials, an
   industry-standard cryptographic signature some cameras/AI tools now embed;
   fast-path check before running any ML
5. **(Stretch goal) rPPG/blood-flow signal** — inspired by Intel FakeCatcher's
   approach of analyzing subtle blood-flow patterns in the face via
   photoplethysmography, instead of only visual artifacts

Individual model outputs get combined (weighted average, or a small meta-
classifier trained on top of them) into one final report with per-signal
breakdown.

## MVP phasing

- **Phase 1** — video upload → frame-artifact CNN → confidence score +
  flagged frames shown to the user
- **Phase 2** — add audio-visual sync scoring
- **Phase 3** — add C2PA metadata check + explainability overlay (heatmap of
  which regions triggered suspicion)
- **Stretch** — rPPG signal, real-time/live analysis instead of upload-only

## Tech stack

- **Backend**: Python, FastAPI (model inference, ensemble scoring)
- **Frontend**: React + Vite (upload UI, results dashboard)
- **Models**: pretrained backbones fine-tuned on public deepfake datasets
  (not trained from scratch — compute/data constraints)
- **Preprocessing**: face detection/alignment (e.g. MTCNN) before
  classification

## Reference datasets

Public, properly licensed/consented research datasets identified for
fine-tuning each model. Do not substitute scraped social media or YouTube
content — that violates platform ToS and raises consent issues around using
real people's faces/voices without permission; the datasets below already
handled that properly.

**Video model (Raw Frames)**
- **FaceForensics++** — real facial video sequences manipulated with four
  methods (DeepFakes, Face2Face, FaceSwap, Neural Textures). Standard
  benchmark. ~1.8TB full size; a subset is enough for fine-tuning.
- **DFDC (Deepfake Detection Challenge)** — largest public set (Meta/
  Microsoft/Amazon + academic partners), 128,064 video clips generated with
  8 different deepfake/GAN-based/non-learned methods. Hosted on Kaggle.
- **Celeb-DF** — 590 real videos + 5,639 high-quality deepfake videos of
  celebrities. Harder to detect (fewer obvious artifacts) — good "hard mode"
  test set.
- **DeeperForensics-1.0** — 60,000 videos, 17.6M frames, includes real-world
  distortions (compression, noise) — useful for robustness to lower-quality
  uploads.
- **Deepfake-Eval-2024** — built from deepfakes that actually circulated in
  the real world in 2024, not lab-generated fakes. Recommended as a final
  validation/test set regardless of which datasets are used for training.

**Image model — manipulated/face-swap detection**
- Frame-extracted stills from the video datasets above (FaceForensics++ is
  explicitly frame-based, easiest to extract from).
- **DiffusionForensics** — 42,000 synthetic face images from Stable
  Diffusion V2 paired with 42,000 real faces from CelebA-HQ. Bridges well
  with the face-manipulation datasets above.

**Image model — fully-synthetic image detection**
- **GenImage** — primary training set. Over one million pairs of AI-generated
  and real images, generated from seven diffusion/text-to-image models
  including Stable Diffusion, Midjourney, and GLIDE. Large and diverse.
- **Artifact dataset** — ~2.5 million images (real + synthetic) generated
  using 25 different GAN and diffusion models. Broader generator variety
  than GenImage — helps generalization instead of overfitting to one
  generator's fingerprint.
- **SynthBuster** — smaller (9,000 images) but covers newer/higher-quality
  generators: DALL·E 2, DALL·E 3, Adobe Firefly, Midjourney v5. Good
  targeted test set for the newest, most photorealistic generators.
- **NTIRE 2026 robustness dataset** — 108,750 real + 185,750 AI-generated
  images from 42 generators, augmented with real-world transformations
  (cropping, resizing, compression, blurring). Use as a robustness/real-world
  test set, since live user uploads won't be clean lab images.

**Audio model**
- **ASVspoof** — standard benchmark for speech spoofing/deepfake audio
  detection.
- **WaveFake** — dataset specifically built to facilitate audio deepfake
  detection research.

**Access notes**
- Most face-related datasets require agreeing to a research-use license
  before download (standard practice to prevent misuse) — review and sign
  before building around a specific one.
- Datasets are large; a well-chosen subset (a few thousand clips/images) is
  enough to fine-tune a pretrained backbone without needing the full dataset.

## Open questions / things to decide early

- **Liability & disclaimers**: results must be framed as probabilistic, not
  definitive proof — especially generalizing beyond hiring to "anyone's video."
- **Abuse potential**: decide how much of the internal scoring to expose
  publicly, to avoid helping someone tune a deepfake to evade the detector.

## Status

Build steps 1–2 of 7 complete (see `docs/CLAUDE_CODE_BRIEF.md` for the full
sequence, `docs/DECISIONS.md` for decisions made along the way).

- ✅ **1. Backend skeleton** — FastAPI, `/analyze/{image,audio,video}`,
  `/report`, async 202 + polling contract, upload caps, delete-after-analysis.
  Detection models are stubs returning deterministic placeholder scores.
- ✅ **2. Frontend skeleton** — React + Vite + TypeScript upload UI and results
  dashboard, running against the real backend. The site is demoable end to end.
- ⬜ 3. Image model · ⬜ 4. Audio model · ⬜ 5. Raw Frames model ·
  ⬜ 6. Video Authenticator · ⬜ 7. PDF evidence report

**No real detection exists yet.** Every verdict the site currently produces is a
placeholder derived from a hash of the uploaded file's bytes, and is labelled as
such in the UI. Next step: dataset access/licensing review, then the Image model.

### Running it locally

```bash
# backend — http://localhost:8000 (docs at /docs)
cd backend
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000

# frontend — http://localhost:5173
cd frontend
npm install && npm run dev
```
