"""Stage 1 — encode every image once with the frozen CLIP vision tower.

Because the backbone never changes, its outputs never change either. Extracting
features once and caching them turns probe training from an hours-long GPU job
into seconds of linear algebra, and makes hyperparameter sweeps and
leave-one-generator-out evaluation cheap enough to actually run. This is what
makes the whole step feasible on a CPU.

    python -m ml.features            # both splits
    python -m ml.features --split validation --limit 500
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .config import (
    BATCH_SIZE,
    CACHE_DIR,
    CLIP_MODEL_ID,
    CLIP_REVISION,
    DATASET_ID,
    FEATURES_DIR,
    ensure_dirs,
)


def _device() -> torch.device:
    # CUDA if present (Colab), else CPU. Deliberately no DirectML/MPS branch:
    # the extraction is a one-off and correctness matters more than speed.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract(split: str, limit: int | None = None) -> None:
    ensure_dirs()
    device = _device()
    print(f"[features] split={split} device={device} model={CLIP_MODEL_ID}")

    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID, revision=CLIP_REVISION)
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID, revision=CLIP_REVISION)
    model.to(device).eval()

    dataset = load_dataset(DATASET_ID, split=split, cache_dir=str(CACHE_DIR))
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    total = len(dataset)
    print(f"[features] {total} images")

    embeddings = np.zeros((total, model.config.projection_dim), dtype=np.float32)
    labels = np.zeros(total, dtype=np.int64)
    generators = np.zeros(total, dtype=np.int64)

    started = time.time()
    for start in range(0, total, BATCH_SIZE):
        batch = dataset[start : start + BATCH_SIZE]

        # Some GenImage source images are palettised or greyscale; CLIP's
        # processor expects 3-channel input.
        images = [img.convert("RGB") for img in batch["image"]]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model(**inputs).image_embeds

        # L2-normalise here rather than at train time so the probe, the
        # evaluation, and the backend inference path all consume features on
        # exactly the same scale. A mismatch here is silent and ruinous.
        out = torch.nn.functional.normalize(out, dim=-1)

        end = start + len(images)
        embeddings[start:end] = out.cpu().numpy()
        labels[start:end] = np.asarray(batch["label"], dtype=np.int64)
        generators[start:end] = np.asarray(batch["generator"], dtype=np.int64)

        if start % (BATCH_SIZE * 20) == 0 or end == total:
            elapsed = time.time() - started
            rate = end / max(elapsed, 1e-6)
            eta = (total - end) / max(rate, 1e-6)
            print(
                f"[features] {end}/{total}  {rate:.1f} img/s  eta {eta / 60:.1f} min",
                flush=True,
            )

    out_path = FEATURES_DIR / f"{split}.npz"
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        labels=labels,
        generators=generators,
        clip_model=CLIP_MODEL_ID,
    )
    print(f"[features] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache CLIP features for Tiny-GenImage.")
    parser.add_argument("--split", choices=["train", "validation", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Cap images, for smoke tests.")
    args = parser.parse_args()

    splits = ["train", "validation"] if args.split == "both" else [args.split]
    for split in splits:
        extract(split, args.limit)


if __name__ == "__main__":
    main()
