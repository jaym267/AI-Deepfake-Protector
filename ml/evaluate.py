"""Stage 3 — leave-one-generator-out generalisation study.

    python -m ml.evaluate

Headline validation accuracy is close to meaningless for this task. The probe is
trained on eight generators that all existed before 2024; the images people
actually upload come from models that did not exist when it was trained. The
question that matters is therefore not "how accurate is it?" but "how much
accuracy survives contact with a generator it has never seen?"

So for each generator in turn: train with that generator's images entirely
removed, then test only on them. The gap between the in-distribution number and
these held-out numbers is the honest estimate of field performance, and it is
what the confidence bands and the results copy have to be truthful about.

Cheap to run because the CLIP features are already cached — nine logistic
regressions over pre-computed vectors, not nine training runs.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import ARTIFACT_DIR, GENERATOR_NAMES, SEED
from .train import load_split

REPORT_PATH = ARTIFACT_DIR / "image_synthetic_probe.generalisation.json"


def main() -> None:
    x_train, y_train, g_train = load_split("train")
    x_val, y_val, g_val = load_split("validation")

    with open(ARTIFACT_DIR / "image_synthetic_probe.json", encoding="utf-8") as handle:
        best_c = json.load(handle)["head"]["C"]
    print(f"[eval] using C={best_c} (selected in training)\n")

    results: dict[str, dict] = {}
    held_out_scores = []

    for index, name in enumerate(GENERATOR_NAMES):
        if index == 0:  # "Real" is not a generator to hold out.
            continue

        # Tiny-GenImage declares a generator ('SD14') that has no examples.
        # Holding out a generator that isn't there would silently produce an
        # ordinary in-distribution score and report it as a generalisation
        # result — the most flattering possible bug.
        if not (g_train == index).any():
            # ASCII only: this runs in a Windows console under cp1252, where a
            # non-ASCII dash comes out as a replacement character.
            print(f"[eval] {name:<12} skipped - no examples in the dataset")
            continue

        train_mask = g_train != index

        probe = LogisticRegression(C=best_c, max_iter=3000, random_state=SEED)
        probe.fit(x_train[train_mask], y_train[train_mask])

        # Test on this generator's fakes plus every real image, so the AUC is a
        # genuine real-vs-this-generator discrimination, not a recall figure
        # that could be gamed by simply calling everything fake.
        test_mask = (g_val == index) | (y_val == 0)
        x_test, y_test = x_val[test_mask], y_val[test_mask]
        if len(np.unique(y_test)) < 2:
            continue

        scores = probe.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, scores)
        recall = float(((scores >= 0.5) & (y_test == 1)).sum() / max((y_test == 1).sum(), 1))

        results[name] = {
            "held_out_auc": round(float(auc), 4),
            "held_out_recall_at_0.5": round(recall, 4),
            "n_test_fakes": int((y_test == 1).sum()),
        }
        held_out_scores.append(auc)
        print(f"[eval] {name:<12} unseen AUC={auc:.4f}  recall={recall:.4f}")

    summary = {
        "mean_held_out_auc": round(float(np.mean(held_out_scores)), 4),
        "worst_held_out_auc": round(float(np.min(held_out_scores)), 4),
        "worst_generator": min(results, key=lambda k: results[k]["held_out_auc"]),
        "per_generator": results,
        "interpretation": (
            "Each figure is measured against a generator excluded from that "
            "probe's training data. The worst case is the number to quote when "
            "describing what this model does on generators released after it was "
            "trained; the mean is optimistic by comparison."
        ),
    }

    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n[eval] mean unseen AUC {summary['mean_held_out_auc']:.4f} · "
        f"worst {summary['worst_held_out_auc']:.4f} ({summary['worst_generator']})"
    )
    print(f"[eval] wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
