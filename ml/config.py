"""Shared configuration for the image model's training pipeline.

Every path is derived from the repo root so the scripts run identically from a
laptop, a Colab runtime, or CI, with no absolute paths baked into artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Large, regenerable, and gitignored. Overridable so a Colab run can point at
# /content and a workstation can point at a scratch disk.
CACHE_DIR = Path(os.environ.get("ADP_ML_CACHE", REPO_ROOT / "ml" / ".cache"))
FEATURES_DIR = CACHE_DIR / "features"

# Small, versioned artifacts. The probe head is a few hundred floats, so unlike
# the backbone it is committed to the repo as JSON — see models/README.md.
ARTIFACT_DIR = REPO_ROOT / "models"

# --- Backbone ---------------------------------------------------------------
# Frozen CLIP vision tower. Not fine-tuned: the whole point of the linear-probe
# approach (Ojha et al., CVPR 2023, "Towards Universal Fake Image Detectors")
# is that CLIP's *unmodified* feature space separates real from generated across
# generators it was never shown, whereas a fine-tuned CNN latches onto the
# fingerprint of whichever generator produced its training data and degrades
# sharply on new ones. Since real uploads come from generators that did not
# exist when this was trained, cross-generator behaviour is the metric that
# matters, not in-distribution accuracy.
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
FEATURE_DIM = 512

# --- Dataset ----------------------------------------------------------------
# Tiny-GenImage: a 35k-image subset of GenImage. Open access, no gating.
#
# LICENCE: CC BY-NC-SA 4.0 — non-commercial, share-alike. The probe head trained
# on it is a derivative work. This is fine while the project is free and becomes
# a blocker for ads, a paid tier, or an API business. See docs/DECISIONS.md D5
# item 2; provenance is recorded in every artifact so a commercially-clean model
# can be retrained later without archaeology.
DATASET_ID = "TheKernel01/Tiny-GenImage"
DATASET_LICENCE = "CC BY-NC-SA 4.0 (non-commercial, share-alike)"

LABEL_NAMES = ["real", "fake"]
GENERATOR_NAMES = [
    "Real",
    "ADM",
    "BigGAN",
    "GLIDE",
    "Midjourney",
    "SD14",
    "SD15",
    "VQDM",
    "Wukong",
]

SEED = 20260729

BATCH_SIZE = int(os.environ.get("ADP_ML_BATCH", 64))


def ensure_dirs() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
