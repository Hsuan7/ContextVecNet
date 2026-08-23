#!/home/angle/miniconda3/envs/contextvecnet/bin/python
"""Preflight checks for the final experiment pipeline."""

from argparse import Namespace

import numpy as np
import torch

from datasets.twitter_learn import CombinedTwitterDataset
from utils import load_args


WINDOW_SIZES = (16, 32, 64, 128)
SPLITS = ("train", "valid", "test")


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        raise SystemExit("CUDA is required by the current training loop.")

    for window_size in WINDOW_SIZES:
        args, _ = load_args(
            Namespace(
                config_file="configs/combos/multi_only.yaml",
                window_size=window_size,
                fold=0,
            )
        )
        split_users = {}
        for split in SPLITS:
            dataset = CombinedTwitterDataset(args=args, kind=split)
            if not dataset:
                raise RuntimeError(
                    f"Empty dataset: window_size={window_size}, split={split}"
                )
            split_users[split] = set(dataset.users)

            sample = dataset[0]
            if len(sample["texts"]) != window_size:
                raise RuntimeError("Text window length mismatch.")
            if len(sample["images"]) != window_size:
                raise RuntimeError("Image window length mismatch.")
            if sample["images"][0].shape != (3, 224, 224):
                raise RuntimeError("Image tensor shape mismatch.")
            if sample["text_mask"].shape != (1, 1, window_size):
                raise RuntimeError("Text mask shape mismatch.")
            if sample["image_mask"].shape != (1, 1, window_size):
                raise RuntimeError("Image mask shape mismatch.")
            if not np.isfinite(np.asarray(sample["time"])).all():
                raise RuntimeError("Timestamp features contain non-finite values.")

            print(
                f"w{window_size} {split}: users={len(dataset)}, "
                f"text_valid={int(sample['text_mask'].sum())}, "
                f"image_valid={int(sample['image_mask'].sum())}"
            )

        if split_users["train"] & split_users["valid"]:
            raise RuntimeError(f"Train/validation leakage at w{window_size}.")
        if split_users["train"] & split_users["test"]:
            raise RuntimeError(f"Train/test leakage at w{window_size}.")
        if split_users["valid"] & split_users["test"]:
            raise RuntimeError(f"Validation/test leakage at w{window_size}.")

    print("FINAL EXPERIMENT PREFLIGHT: OK")


if __name__ == "__main__":
    main()
