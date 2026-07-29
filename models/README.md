# Model artifacts

What the backend loads at inference time. Produced by `ml/` — see `ml/README.md`
for how to regenerate any of it.

## What's here

| Artifact | Contents |
|---|---|
| `image_synthetic_probe.json` | The trained head: probe coefficients, ROC-derived verdict thresholds, and a provenance block. **Committed.** |
| `image_synthetic_probe.metrics.json` | Validation AUC, accuracy, per-generator breakdown. **Committed.** |
| `image_synthetic_probe.generalisation.json` | Leave-one-generator-out results. **Committed.** |

Committed on purpose. The head is a logistic regression over 512 features — 513
floats — so it is small, diffable, and inspectable, and shipping it means the
repo produces real results on clone rather than requiring a 35-minute training
run first. Its metrics are committed alongside it so the claims made about the
model stay checkable against the artifact actually in use.

The `.gitignore` still excludes binary checkpoints (`*.pt`, `*.safetensors`, …).
When a step produces real weights rather than a linear head — likely step 4 or 5
— those belong in release assets or a model registry, not here.

## The backbone is not here

`image_synthetic_probe.json` names a backbone (`openai/clip-vit-base-patch32`)
rather than embedding it. It is downloaded from Hugging Face on first inference
and cached (~350MB). That means:

- **First request after a cold start is slow** and needs network access.
- **A production deployment should pre-fetch it** into the image or a mounted
  cache rather than fetching on first user request.
- The backbone is CLIP, under its own licence (MIT for the model code; OpenAI's
  weights are released for research use). It is used unmodified and frozen.

## Licence — read before reusing or monetising

`image_synthetic_probe.json` is trained on **Tiny-GenImage**, a subset of
GenImage, licensed **CC BY-NC-SA 4.0**.

The probe head is a derivative work of that dataset, so on the most cautious
reading it inherits both clauses:

- **NonCommercial** — no ads, paid tier, API business, or enterprise offering
  built on these coefficients. Keeping the site free to users is probably not
  sufficient on its own; NC is generally read as restricting commercial
  *purpose*, not just direct charging.
- **ShareAlike** — redistributing the head (which this repo does, by committing
  it) plausibly requires it to carry the same licence.

**Accordingly, `image_synthetic_probe.json` is released under CC BY-NC-SA 4.0**,
attributed to the GenImage authors, independent of whatever licence the rest of
this repository carries.

This is a good-faith reading, not legal advice. Anyone getting close to
monetisation should have a lawyer confirm it, and the cheap insurance is to
retrain the head on permissively licensed data — the provenance block in each
artifact records exactly what would need replacing. See `docs/DECISIONS.md` D5
item 2.

## Scope of the image model

`image_synthetic_probe` detects **fully AI-generated images**. It does **not**
detect face swaps or localised edits to real photographs; that head needs gated
face-manipulation datasets (D6). Every image result states this limitation to
the user directly, and it should keep doing so until head (a) actually exists.
