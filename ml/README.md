# Training pipeline

Produces the model artifacts in `models/`. The backend consumes those artifacts;
nothing in the request path imports from this package.

Currently covers **build step 3 — the image model's synthetic-detection head**.
Steps 4–6 (audio, raw frames, video authenticator) will add scripts here.

## Approach: frozen CLIP + linear probe

The image head is a logistic regression over the 512-d output of a **frozen**
CLIP ViT-B/32 vision tower. The backbone is not fine-tuned.

This is not a compute compromise — it is the choice that matches the threat
model. Fine-tuning a CNN end-to-end on a fixed set of generators reliably scores
better on those generators and reliably falls apart on new ones, because what it
learns is each generator's particular fingerprint. But every image a user
actually uploads comes from a generator newer than the training set. Ojha et al.
(CVPR 2023, *Towards Universal Fake Image Detectors*) showed a linear probe on
frozen CLIP features generalises substantially better to unseen generators, and
`evaluate.py` measures that property directly rather than taking it on faith.

Two useful side effects: the features are extracted once and cached, so the
whole pipeline runs on a CPU in well under an hour; and the trained head is 513
floats, small enough to commit as JSON and serve with a dot product.

## Running it

From the repo root, with the backend venv active:

```bash
pip install -r ml/requirements.txt

python -m ml.features     # stage 1 — encode 35k images (~35 min on 8 CPU cores)
python -m ml.train        # stage 2 — fit the probe, derive thresholds (seconds)
python -m ml.evaluate     # stage 3 — leave-one-generator-out study (seconds)
```

Stage 1 downloads ~8.4GB of dataset on first run and caches CLIP features to
`ml/.cache/` (gitignored). Stages 2 and 3 read only the cached features, so
re-running them is nearly free — which is what makes the sweep in `train.py` and
the nine-fold study in `evaluate.py` practical.

Set `ADP_ML_CACHE` to relocate the cache, `ADP_ML_BATCH` to change batch size.
On a CUDA machine stage 1 uses the GPU automatically and takes a minute or two.

## Files

| File | Role |
|---|---|
| `config.py` | Paths, backbone id, dataset id and licence, seed |
| `features.py` | Stage 1 — CLIP encoding, cached to `.npz` |
| `train.py` | Stage 2 — probe fit, threshold derivation, artifact export |
| `evaluate.py` | Stage 3 — leave-one-generator-out generalisation |

## Dataset

**Tiny-GenImage** (`TheKernel01/Tiny-GenImage`) — 28,000 train / 7,000
validation, balanced real vs. generated, labelled by generator: ADM, BigGAN,
GLIDE, Midjourney, SD1.4, SD1.5, VQDM, Wukong. A subset of GenImage, open access
with no gating.

> **Licence: CC BY-NC-SA 4.0 — non-commercial, share-alike.**
> The probe head is a derivative work. Fine while the project is free; a genuine
> blocker for ads, a paid tier, an API business, or an enterprise offering built
> on these weights. Every artifact carries a `provenance` block recording the
> dataset and licence so a commercially-clean model can be retrained later
> without archaeology. See `docs/DECISIONS.md` D5 item 2.

## Reading the numbers

`train.py` reports validation AUC on the eight generators it trained on. That
number is the optimistic one and should never be quoted as the model's accuracy.

`evaluate.py` reports AUC per generator with that generator held out of training
entirely. **Those are the numbers that describe field performance**, because
real uploads come from generators this probe has never seen. The worst-case
figure is the honest headline; the mean is already flattering.

Both land in `models/*.metrics.json` and `models/*.generalisation.json`, which
are committed so the claims stay checkable.

## What this head does not do

It detects **fully AI-generated images**. It does **not** detect a real
photograph with a swapped face or an edited region — that is head (a) in the
brief, and it needs FaceForensics++ / Celeb-DF, both gated behind institutional
access this project does not have (`docs/DECISIONS.md` D6). The backend states
this limitation to the user in every image result rather than leaving it implied.

There is no MTCNN face detection step. The brief lists it as step 3
preprocessing, but it is preprocessing for head (a); a whole-image synthetic
classifier gains nothing from knowing where the faces are.
