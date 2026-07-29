"""Loading of trained model artifacts and the frozen CLIP backbone.

Two things live here so no detector has to think about either:

* **Lazy loading.** The CLIP tower is ~350MB and takes a second or two to
  initialise. Loading it at import time would make ``pytest`` and ``--reload``
  painful and would slow container start for a process that might only ever
  serve ``/health``. It loads on first inference and is cached after.

* **Absence as a first-class state.** If the probe artifact is missing the model
  reports itself unavailable and the API returns 503. It does *not* fall back to
  a stub score. A tool that tells someone their video is fake must never do so
  because a file failed to load — see docs/DECISIONS.md D9.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# backend/app/services/models/artifacts.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_DIR = REPO_ROOT / "models"

_clip_lock = threading.Lock()


@dataclass(frozen=True)
class Thresholds:
    """Verdict-band boundaries, derived from a validation ROC (D4)."""

    authentic_below: float
    possible_above: float
    manipulated_above: float


@dataclass(frozen=True)
class LinearProbe:
    """A logistic regression over frozen backbone features.

    Small enough (513 floats) to ship as JSON in the repo, which means inference
    needs no sklearn — just this dot product.
    """

    coef: np.ndarray
    intercept: float
    thresholds: Thresholds
    backbone_id: str
    version: int
    metrics: dict
    provenance: dict

    def score(self, features: np.ndarray) -> float:
        """P(manipulated) for one L2-normalised feature vector."""
        logit = float(np.dot(self.coef, features) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))


@lru_cache(maxsize=None)
def load_probe(name: str) -> LinearProbe | None:
    """Load a probe artifact by name, or None if it hasn't been trained yet."""
    path = ARTIFACT_DIR / f"{name}.json"
    if not path.exists():
        logger.warning(
            "Model artifact %s not found. Run `python -m ml.train` to build it; "
            "the corresponding endpoint will return 503 until then.",
            path,
        )
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        thresholds = raw["thresholds"]
        return LinearProbe(
            coef=np.asarray(raw["head"]["coef"], dtype=np.float32),
            intercept=float(raw["head"]["intercept"]),
            thresholds=Thresholds(
                authentic_below=float(thresholds["authentic_below"]),
                possible_above=float(thresholds["possible_above"]),
                manipulated_above=float(thresholds["manipulated_above"]),
            ),
            backbone_id=raw["backbone"]["model_id"],
            version=int(raw["version"]),
            metrics=raw.get("metrics", {}),
            provenance=raw.get("provenance", {}),
        )
    except (KeyError, ValueError, TypeError):
        # A malformed artifact is a deployment error, not a user error. Refusing
        # to load is correct; scoring with a half-parsed model is not.
        logger.exception("Model artifact %s is malformed and was not loaded", path)
        return None


@lru_cache(maxsize=None)
def _load_clip(model_id: str):
    # Imported here rather than at module scope so that importing the app (for
    # tests, or to serve /health) doesn't pull in torch and transformers.
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    logger.info("Loading CLIP backbone %s", model_id)
    processor = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPVisionModelWithProjection.from_pretrained(model_id)
    model.eval()
    torch.set_grad_enabled(False)
    return processor, model


def embed_image(image, model_id: str) -> np.ndarray:
    """Encode a PIL image into an L2-normalised CLIP feature vector.

    The normalisation matters: ml/features.py normalises before training the
    probe, so skipping it here would feed the probe vectors on a different scale
    and produce confident nonsense rather than an obvious failure.
    """
    import torch

    # from_pretrained is not guaranteed thread-safe on first call, and FastAPI
    # runs sync endpoints in a threadpool, so two uploads can race here.
    with _clip_lock:
        processor, model = _load_clip(model_id)

    inputs = processor(images=[image.convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        embeds = model(**inputs).image_embeds
        embeds = torch.nn.functional.normalize(embeds, dim=-1)
    return embeds[0].cpu().numpy().astype(np.float32)
