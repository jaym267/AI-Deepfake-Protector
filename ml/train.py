"""Stage 2 — fit the linear probe and export the artifact the backend loads.

    python -m ml.train

Produces ``models/image_synthetic_probe.json``: the probe weights, the verdict
thresholds derived from the validation ROC, and a provenance block.

The head is a single logistic regression over 512 L2-normalised CLIP features —
513 numbers. That is why the artifact is committed JSON rather than a binary
checkpoint: it is small, diffable, and inspectable, and the backend needs no
sklearn at runtime, just a dot product and a sigmoid.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from .config import (
    ARTIFACT_DIR,
    CLIP_MODEL_ID,
    DATASET_ID,
    DATASET_LICENCE,
    FEATURE_DIM,
    FEATURES_DIR,
    GENERATOR_NAMES,
    SEED,
    ensure_dirs,
)

ARTIFACT_PATH = ARTIFACT_DIR / "image_synthetic_probe.json"
METRICS_PATH = ARTIFACT_DIR / "image_synthetic_probe.metrics.json"


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = FEATURES_DIR / f"{split}.npz"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `python -m ml.features --split {split}` first."
        )
    data = np.load(path)
    return data["embeddings"], data["labels"], data["generators"]


def derive_thresholds(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Turn a validation ROC into the three verdict-band boundaries.

    docs/DECISIONS.md D4 requires these come from the ROC rather than being
    picked by eye, and the two outer bands are the ones that can actually harm
    someone, so each is pinned to an explicit error rate:

      t_authentic     at most ~5% of *fakes* score below it, so calling
                      something "likely authentic" rarely misses a real fake.
      t_manipulated   at most ~5% of *real* images score above it, so a genuine
                      photo of a real person is rarely branded manipulated.
      t_possible      the equal-error point, splitting the middle into
                      "uncertain" and "possibly manipulated".

    Clamping to the EER point keeps the three ordered even when the model
    separates the classes so cleanly that the two quantiles cross over.
    """
    fake_scores = scores[labels == 1]
    real_scores = scores[labels == 0]

    fpr, tpr, cuts = roc_curve(labels, scores)
    eer_index = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    eer_threshold = float(cuts[eer_index])
    eer = float((fpr[eer_index] + (1 - tpr[eer_index])) / 2)

    t_authentic = min(float(np.quantile(fake_scores, 0.05)), eer_threshold)
    t_manipulated = max(float(np.quantile(real_scores, 0.95)), eer_threshold)

    return {
        "authentic_below": round(t_authentic, 6),
        "possible_above": round(eer_threshold, 6),
        "manipulated_above": round(t_manipulated, 6),
        "equal_error_rate": round(eer, 6),
    }


def per_generator(scores, labels, generators, threshold: float) -> dict[str, dict]:
    """Detection rate broken down by which generator produced the image.

    Reported because a single headline accuracy hides the failure that matters:
    a probe can look excellent overall while being near-blind to one generator.
    """
    out: dict[str, dict] = {}
    for index, name in enumerate(GENERATOR_NAMES):
        mask = generators == index
        if not mask.any():
            continue
        subset_labels = labels[mask]
        subset_scores = scores[mask]
        predicted = (subset_scores >= threshold).astype(int)
        out[name] = {
            "n": int(mask.sum()),
            "accuracy": round(float(accuracy_score(subset_labels, predicted)), 4),
            "mean_score": round(float(subset_scores.mean()), 4),
        }
    return out


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main() -> None:
    ensure_dirs()

    x_train, y_train, g_train = load_split("train")
    x_val, y_val, g_val = load_split("validation")
    print(f"[train] train={x_train.shape} validation={x_val.shape}")

    # Derived from the data, never from the dataset card. Tiny-GenImage declares
    # nine class labels but ships only eight: 'SD14' has zero examples. Recording
    # the card's list as "generators seen" would put a generator the model has
    # never been shown into the artifact's provenance, and provenance that is
    # wrong is worse than provenance that is absent.
    present = sorted(set(g_train.tolist()) - {0})
    generators_seen = [GENERATOR_NAMES[i] for i in present]
    declared = set(GENERATOR_NAMES[1:])
    missing = declared - set(generators_seen)
    if missing:
        print(f"[train] NOTE: declared but absent from the data: {sorted(missing)}")
    print(f"[train] generators actually trained on ({len(generators_seen)}): {generators_seen}")

    if x_train.shape[1] != FEATURE_DIM:
        raise SystemExit(f"Expected {FEATURE_DIM}-d features, got {x_train.shape[1]}.")

    # A sweep rather than a fixed C: the useful capacity of a linear probe over
    # frozen features depends on how separable this particular feature space is,
    # and that is not knowable in advance. Selected on validation AUC.
    best = None
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        probe = LogisticRegression(C=c, max_iter=3000, random_state=SEED)
        probe.fit(x_train, y_train)
        auc = roc_auc_score(y_val, probe.predict_proba(x_val)[:, 1])
        print(f"[train] C={c:<6} val AUC={auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, c, probe)

    assert best is not None
    val_auc, best_c, probe = best
    print(f"[train] selected C={best_c} (val AUC={val_auc:.4f})")

    val_scores = probe.predict_proba(x_val)[:, 1]
    thresholds = derive_thresholds(val_scores, y_val)
    accuracy = accuracy_score(y_val, (val_scores >= 0.5).astype(int))

    metrics = {
        "validation_auc": round(float(val_auc), 4),
        "validation_accuracy_at_0.5": round(float(accuracy), 4),
        "thresholds": thresholds,
        "per_generator": per_generator(val_scores, y_val, g_val, 0.5),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
    }

    artifact = {
        "name": "image_synthetic_probe",
        "version": 1,
        "task": "P(image is fully AI-generated)",
        "scope": (
            "Fully synthetic image detection only. This head does NOT detect "
            "face swaps or localised edits to real photographs — that requires "
            "face-manipulation training data which is gated behind institutional "
            "access (docs/DECISIONS.md D6)."
        ),
        "backbone": {
            "model_id": CLIP_MODEL_ID,
            "frozen": True,
            "feature_dim": FEATURE_DIM,
            "preprocessing": "CLIPImageProcessor defaults, then L2 normalisation",
        },
        "head": {
            "kind": "logistic_regression",
            "C": best_c,
            "coef": probe.coef_[0].astype(float).round(6).tolist(),
            "intercept": float(round(probe.intercept_[0], 6)),
        },
        "thresholds": thresholds,
        "metrics": metrics,
        "provenance": {
            "dataset": DATASET_ID,
            "dataset_licence": DATASET_LICENCE,
            "licence_note": (
                "Non-commercial. This head is a derivative work of a CC BY-NC-SA "
                "4.0 dataset. Retrain on permissively licensed data before any "
                "commercial use. See docs/DECISIONS.md D5 item 2."
            ),
            "generators_seen": generators_seen,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "seed": SEED,
        },
    }

    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n[train] validation AUC      {val_auc:.4f}")
    print(f"[train] accuracy @0.5      {accuracy:.4f}")
    print(f"[train] thresholds         {thresholds}")
    print("[train] per-generator accuracy:")
    for name, stats in metrics["per_generator"].items():
        print(f"          {name:<12} n={stats['n']:<6} acc={stats['accuracy']:.4f}")
    print(f"\n[train] wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
