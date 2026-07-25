# Project Brief for Claude Code: AI Deepfake Protection

Read this entire document before writing any code. This is a planning brief,
not a finished spec — ask clarifying questions where something is ambiguous
rather than guessing silently on anything that affects architecture.

## What this project is

A free, public website where anyone can upload a video, image, or audio file
and get a clear, evidence-backed assessment of whether it's likely
AI-generated/deepfaked or authentic. Not limited to any one use case (hiring,
interviews, etc.) — built for the general public, including people trying to
verify a suspicious video sent to them or defend their own reputation against
a fake circulating about them.

## Why this project exists / what makes it different

Existing deepfake detection tools (Reality Defender, Sensity AI, CloudSEK,
Microblink, Intel FakeCatcher) are almost all enterprise/B2B — priced and
built for fraud teams, banks, and law enforcement, not individuals. This
project differentiates on:
- Free and built for individuals, not enterprise pricing tiers
- Reputation-protection framing, not just financial-fraud framing
- Plain-language explainability (e.g. "lip movement doesn't match audio in
  these 3 seconds") instead of jargon-heavy forensic reports
- Honesty about uncertainty — a probability + visible evidence, not a false
  binary "100% REAL" badge
- A downloadable evidence report (PDF) a person can actually use — send to a
  platform, employer, or as a starting point with law enforcement

## Model architecture (this is the core design — follow it precisely)

Four models, not one monolithic classifier:

1. **Image model** — detects both (a) manipulated/face-swapped real photos
   and (b) fully AI-generated images (e.g. Midjourney/Stable Diffusion
   output). These are different artifact types and may need different
   training data/heads even within one model.
2. **Audio model** — standalone voice-clone / synthetic-speech detector.
   Must work independently of video (e.g. for voicemails, phone scam
   recordings), not only as a sub-component of video analysis.
3. **Raw Frames model** — video-specific temporal analysis: flicker,
   unnatural blink rate, inconsistent lighting/motion frame-to-frame. This
   is distinct from the Image model because it needs sequence/temporal
   information a single still frame can't provide.
4. **Video Authenticator** — takes the outputs of the Image model, Audio
   model, and Raw Frames model (all three, run against the same video) and
   combines them into one final video-level verdict. This should be a
   trained meta-classifier (stacking ensemble) that learns how to weigh the
   three inputs — not a fixed formula/simple average — once there's enough
   labeled data to train it; a simple weighted-average fallback is
   acceptable for an early MVP if training data is initially too limited.

For a **video** upload: run Image + Audio + Raw Frames → Video Authenticator
→ final verdict.
For an **image-only** upload: Image model's output is the final verdict.
For an **audio-only** upload: Audio model's output is the final verdict.

A second stacked meta-layer above the Video Authenticator was discussed and
explicitly deferred — do not build it now. Note it as a future extension in
code comments/docs, but do not implement it in this phase.

## Technical approach for building the models

- **Do not train from scratch.** Fine-tune existing open pretrained
  backbones (e.g. a pretrained CNN/Vision Transformer for image/video,
  something like Wav2Vec2 for audio) on the datasets listed below.
- **Do not just wrap an existing hosted deepfake-detection API** (e.g.
  calling a third-party detection service and passing through its result).
  The detection itself must be genuinely built/fine-tuned as part of this
  project — that's the whole point of it being a real ML project rather
  than a UI on top of someone else's product.
- An LLM (via API) has a legitimate, different role: taking the ensemble's
  raw technical outputs (scores, flagged timestamps/regions) and generating
  the plain-language explanation and the evidence report text. This is a
  supporting/presentation role, not the core detection logic.

## Datasets

**Use these first — open access, no institutional forms required:**
- **GenImage** (image model, fully-synthetic detection) —
  https://github.com/GenImage-Dataset/GenImage — CC BY-NC-SA 4.0 license,
  non-commercial use. Smaller starter subset also available at
  https://huggingface.co/datasets/TheKernel01/Tiny-GenImage
- **DFDC** (video/Raw Frames model) — hosted as a Kaggle competition:
  https://www.kaggle.com/c/deepfake-detection-challenge — requires only a
  free Kaggle account and accepting competition rules, no institutional
  affiliation needed
- **ASVspoof** (audio model) — https://www.asvspoof.org — check current
  year's registration page; open to individuals
- **WaveFake** (audio model) — hosted on Zenodo, typically open download

**Gated/deferred — require an institutional advisor field the user does not
currently have; do not build around these yet, treat as a future data
upgrade once an advisor is available:**
- FaceForensics++ — https://github.com/ondyari/FaceForensics
- Celeb-DF — https://github.com/yuezunli/celeb-deepfakeforensics
- DeeperForensics-1.0 — https://github.com/EndlessSora/DeeperForensics-1.0

**Do not scrape YouTube or any other platform directly for training data.**
This violates platform ToS and raises consent issues around using real
people's faces/voices without permission. Only use the datasets listed
above (or similarly properly-licensed/consented public research datasets).

## Upload limits (MVP)

Strict, to keep compute/storage cheap during development:
- Video: max ~30-60 seconds, max ~25MB
- Audio: max ~2 minutes, max ~10MB
- Images: max ~5MB

## Tech stack

- **Backend**: Python, FastAPI
- **Frontend**: React + Vite
- **Models**: pretrained backbones fine-tuned on the datasets above
- **Preprocessing**: face detection/alignment (e.g. MTCNN) before
  image/video classification steps

## Build order (follow this sequence — do not skip ahead to real models
before the skeleton is working end-to-end)

1. **Backend skeleton** — FastAPI app with stub endpoints that return
   mock/hardcoded scores:
   - `POST /analyze/image`
   - `POST /analyze/audio`
   - `POST /analyze/video`
   - `POST /report` (takes an analysis result, will eventually generate a
     PDF evidence report)
2. **Frontend skeleton** — upload UI (accepts image/audio/video) + a results
   dashboard, wired up against the mock backend endpoints above. Goal: a
   fully working, demoable full-stack site before any real ML exists.
3. **Image model** — real fine-tuned model, swapped into `/analyze/image`
4. **Audio model** — real fine-tuned model, swapped into `/analyze/audio`
5. **Raw Frames model** — video temporal analysis
6. **Video Authenticator** — combine Image + Audio + Raw Frames outputs for
   video uploads, swapped into `/analyze/video`
7. **Evidence report generation** — real PDF output once real model outputs
   exist to report on (use an LLM to translate technical outputs into
   plain-language explanations, per above)

## Things to flag to the user rather than deciding silently

- **Liability/disclaimers**: results must be presented as probabilistic, not
  definitive proof, especially since this covers "anyone's" video/image/
  audio, not a narrow use case. Confirm disclaimer wording with the user
  before finalizing UI copy.
- **Abuse potential**: how much internal scoring detail to expose publicly
  (full breakdown vs. summary only) — exposing too much detail could help
  someone tune a deepfake to evade the detector. Confirm with the user
  before deciding what the results UI actually displays.
- Any point where a dataset's license terms (e.g. GenImage's non-commercial
  clause) might conflict with a future monetization idea — surface this
  rather than assuming it's fine.

## Current status

Planning complete. No code written yet. Start at Build Order Step 1.
