"""Evaluate checkpoint robustness under test-time input perturbations.

The baseline uses the normal validation split for threshold selection, then
applies perturbations only to the test batches. This supports the reviewer
checks for temporal order, user-history matching, and image matching.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import evaluate_calibration_comparison as base_eval
from datasets.twitter_learn import CombinedTwitterDataset
from utils import load_args, load_checkpoint


DEVICE = base_eval.DEVICE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--perturbations",
        nargs="+",
        choices=[
            "baseline",
            "time_shuffle",
            "history_mismatch",
            "no_history",
            "image_mismatch",
            "image_zero",
        ],
        default=[
            "baseline",
            "time_shuffle",
            "history_mismatch",
            "no_history",
            "image_mismatch",
            "image_zero",
        ],
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--position_embeddings", default=None)
    parser.add_argument("--image_embeddings_type", default=None)
    parser.add_argument("--text_embeddings_type", default=None)
    return parser.parse_args()


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_collate_fn(args):
    return base_eval.build_collate_fn(args)


def clone_batch(batch):
    cloned = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned



def deranged_indices(n, seed):
    rng = np.random.default_rng(seed)
    if n <= 1:
        return np.arange(n)
    indices = np.arange(n)
    for _ in range(100):
        perm = rng.permutation(n)
        if np.all(perm != indices):
            return perm
    return np.roll(indices, 1)


class HistoryMismatchDataset(Dataset):
    """Pair each target user's label with another user's full history."""

    def __init__(self, base_dataset, seed):
        self.base_dataset = base_dataset
        self.source_indices = deranged_indices(len(base_dataset), seed)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        target = self.base_dataset[idx]
        source = dict(self.base_dataset[int(self.source_indices[idx])])
        source["author"] = target["author"]
        source["label"] = target["label"]
        return source

def apply_time_shuffle(batch, rng):
    out = clone_batch(batch)
    for row_idx, texts in enumerate(out["texts"]):
        text_valid = out["text_mask"][row_idx].reshape(-1).bool()
        image_valid = out["image_mask"][row_idx].reshape(-1).bool()
        valid_mask = (text_valid | image_valid).cpu().numpy()
        valid_positions = np.flatnonzero(valid_mask)
        if len(valid_positions) <= 1:
            continue

        shuffled_positions = rng.permutation(valid_positions)
        new_texts = list(out["texts"][row_idx])
        new_images = list(out["images"][row_idx])
        new_image_paths = list(out["images_paths"][row_idx])
        new_text_mask = out["text_mask"][row_idx].clone()
        new_image_mask = out["image_mask"][row_idx].clone()

        for target_pos, source_pos in zip(valid_positions, shuffled_positions):
            new_texts[target_pos] = out["texts"][row_idx][source_pos]
            new_images[target_pos] = out["images"][row_idx][source_pos]
            new_image_paths[target_pos] = out["images_paths"][row_idx][source_pos]
            new_text_mask[:, :, target_pos] = out["text_mask"][row_idx][
                :, :, source_pos
            ]
            new_image_mask[:, :, target_pos] = out["image_mask"][row_idx][
                :, :, source_pos
            ]

        # Keep the original chronological time slots fixed. This breaks the
        # content-time alignment instead of simply permuting an order-invariant set.
        out["texts"][row_idx] = new_texts
        out["images"][row_idx] = new_images
        out["images_paths"][row_idx] = new_image_paths
        out["text_mask"][row_idx] = new_text_mask
        out["image_mask"][row_idx] = new_image_mask
    return out


def rotate_indices(labels, rng):
    n = len(labels)
    if n <= 1:
        return np.arange(n)
    candidates = np.arange(n)
    for _ in range(20):
        perm = rng.permutation(n)
        if np.all(perm != candidates):
            return perm
    return np.roll(candidates, 1)


def apply_history_mismatch(batch, rng):
    out = clone_batch(batch)
    perm = rotate_indices(batch["label"].reshape(-1).numpy(), rng)
    for key in ["texts", "images", "images_paths", "padding_amount"]:
        out[key] = [out[key][i] for i in perm]
    for key in ["time", "text_mask", "image_mask"]:
        out[key] = out[key][perm].clone()
    return out


def apply_no_history(batch):
    out = clone_batch(batch)
    for row_idx in range(len(out["texts"])):
        text_valid = out["text_mask"][row_idx].reshape(-1).bool()
        image_valid = out["image_mask"][row_idx].reshape(-1).bool()
        valid_mask = (text_valid | image_valid).cpu().numpy()
        valid_positions = np.flatnonzero(valid_mask)
        if len(valid_positions) <= 1:
            continue

        keep_pos = int(valid_positions[-1])
        new_texts = ["<PAD>"] * len(out["texts"][row_idx])
        new_images = list(out["images"][row_idx])
        new_image_paths = ["<PAD_PATH>"] * len(out["images_paths"][row_idx])
        new_texts[keep_pos] = out["texts"][row_idx][keep_pos]
        new_image_paths[keep_pos] = out["images_paths"][row_idx][keep_pos]

        new_text_mask = torch.zeros_like(out["text_mask"][row_idx])
        new_image_mask = torch.zeros_like(out["image_mask"][row_idx])
        new_text_mask[:, :, keep_pos] = out["text_mask"][row_idx][:, :, keep_pos]
        new_image_mask[:, :, keep_pos] = out["image_mask"][row_idx][:, :, keep_pos]

        out["texts"][row_idx] = new_texts
        out["images"][row_idx] = new_images
        out["images_paths"][row_idx] = new_image_paths
        out["text_mask"][row_idx] = new_text_mask
        out["image_mask"][row_idx] = new_image_mask
    return out

def apply_image_mismatch(batch, rng):
    out = clone_batch(batch)
    perm = rotate_indices(batch["label"].reshape(-1).numpy(), rng)
    out["images"] = [out["images"][i] for i in perm]
    out["images_paths"] = [out["images_paths"][i] for i in perm]
    out["image_mask"] = out["image_mask"][perm].clone()
    return out


def apply_image_zero(batch):
    out = clone_batch(batch)
    out["image_mask"] = torch.zeros_like(out["image_mask"])
    return out


def perturb_batch(batch, perturbation, rng):
    if perturbation == "baseline":
        return batch
    if perturbation == "time_shuffle":
        return apply_time_shuffle(batch, rng)
    if perturbation == "history_mismatch":
        return apply_history_mismatch(batch, rng)
    if perturbation == "no_history":
        return apply_no_history(batch)
    if perturbation == "image_mismatch":
        return apply_image_mismatch(batch, rng)
    if perturbation == "image_zero":
        return apply_image_zero(batch)
    raise ValueError(perturbation)


def collect_logits(model, args, kind, perturbation, seed):
    dataset = CombinedTwitterDataset(args=args, kind=kind)
    if kind == "test" and perturbation == "history_mismatch":
        dataset = HistoryMismatchDataset(dataset, seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        collate_fn=build_collate_fn(args),
    )
    rng = np.random.default_rng(seed)
    logits = []
    labels = []
    authors = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{kind}:{perturbation}"):
            if kind == "test" and perturbation != "history_mismatch":
                batch = perturb_batch(batch, perturbation, rng)
            output = model(batch)
            logits.extend(output["logits"].detach().cpu().numpy().reshape(-1))
            labels.extend(batch["label"].numpy().reshape(-1))
            authors.extend(batch["author"])
    return np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=int), authors


def metric_row(args, perturbation, labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, _fn, _tp = metrics.confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "experiment": args.name,
        "fold": args.fold,
        "window_size": args.window_size,
        "perturbation": perturbation,
        "threshold": threshold,
        "accuracy": metrics.accuracy_score(labels, predictions),
        "precision": metrics.precision_score(labels, predictions, zero_division=0),
        "recall": metrics.recall_score(labels, predictions, zero_division=0),
        "macro_f1": metrics.f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "f1": metrics.f1_score(labels, predictions, zero_division=0),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "auc": metrics.roc_auc_score(labels, probabilities),
        "nll": metrics.log_loss(labels, probabilities, labels=[0, 1]),
        "brier": metrics.brier_score_loss(labels, probabilities),
        "ece_10": base_eval.expected_calibration_error(labels, probabilities),
        "n_samples": len(labels),
    }


def main():
    fix_seed(42)
    args, _ = load_args(parse_args())
    model = base_eval.build_model(args)
    checkpoint = load_checkpoint(args)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE).float().eval()

    val_logits, val_labels, _ = collect_logits(
        model, args, "valid", "baseline", args.seed
    )
    val_probabilities = base_eval.sigmoid(val_logits)
    threshold = base_eval.select_f1_threshold(val_labels, val_probabilities)

    rows = []
    prediction_rows = []
    for perturbation in args.perturbations:
        test_logits, test_labels, authors = collect_logits(
            model,
            args,
            "test",
            perturbation,
            args.seed + args.fold * 100 + len(rows),
        )
        probabilities = base_eval.sigmoid(test_logits)
        rows.append(metric_row(args, perturbation, test_labels, probabilities, threshold))
        for author, label, logit, probability in zip(
            authors, test_labels, test_logits, probabilities
        ):
            prediction_rows.append(
                {
                    "experiment": args.name,
                    "fold": args.fold,
                    "author": author,
                    "label": label,
                    "perturbation": perturbation,
                    "logit": logit,
                    "probability": probability,
                    "threshold": threshold,
                    "prediction": int(probability >= threshold),
                }
            )

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(
        args.output_dir, f"{args.name}_input_perturbation_metrics.csv"
    )
    predictions_path = os.path.join(
        args.output_dir, f"{args.name}_input_perturbation_predictions.csv"
    )
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("WROTE", metrics_path)
    print("WROTE", predictions_path)


if __name__ == "__main__":
    main()
