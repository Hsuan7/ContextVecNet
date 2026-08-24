"""Offline robustness checks from saved user-level prediction CSVs.

This script evaluates how reported conclusions change when test labels are
randomly flipped or fully permuted. It does not retrain a model; it treats the
saved probabilities as fixed model outputs and perturbs the weak user labels.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn import metrics


DEFAULT_METHODS = ["text_bert", "bert_clip"]


def expected_calibration_error(labels, probabilities, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, edges[1:-1])
    total = len(labels)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        confidence = probabilities[mask].mean()
        accuracy = labels[mask].mean()
        ece += (mask.sum() / total) * abs(confidence - accuracy)
    return float(ece)


def specificity_score(labels, predictions):
    tn, fp, _fn, _tp = metrics.confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def metric_row(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "auc": metrics.roc_auc_score(labels, probabilities)
        if len(np.unique(labels)) == 2
        else np.nan,
        "macro_f1": metrics.f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "sensitivity": metrics.recall_score(
            labels, predictions, zero_division=0
        ),
        "specificity": specificity_score(labels, predictions),
        "ece_10": expected_calibration_error(labels, probabilities),
        "brier": metrics.brier_score_loss(labels, probabilities),
        "nll": metrics.log_loss(labels, probabilities, labels=[0, 1]),
    }


def load_predictions(results_root, method, window_size):
    pattern = os.path.join(
        results_root,
        "methods",
        method,
        f"w{window_size}",
        f"{method}_w{window_size}_fold*_predictions.csv",
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No prediction files matched: {pattern}")
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    data = data[
        (data["calibration"] == "none")
        & (data["threshold_strategy"] == "validation_f1")
    ].copy()
    if data.empty:
        raise ValueError(f"No uncalibrated validation_f1 rows for {method}")
    return data


def perturb_labels(labels, mode, rate, seed):
    rng = np.random.default_rng(seed)
    labels = labels.copy()
    if mode == "label_flip":
        n_flip = int(round(len(labels) * rate))
        flip_idx = rng.choice(len(labels), size=n_flip, replace=False)
        labels[flip_idx] = 1 - labels[flip_idx]
        return labels, n_flip
    if mode == "label_permutation":
        return rng.permutation(labels), len(labels)
    raise ValueError(mode)


def summarize(rows, group_cols):
    frame = pd.DataFrame(rows)
    metric_cols = [
        "auc",
        "macro_f1",
        "sensitivity",
        "specificity",
        "ece_10",
        "brier",
        "nll",
    ]
    summary = (
        frame.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if part != ""]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    return frame, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root", default="results/final_preprocessing_v2"
    )
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--noise_rates", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--output_dir",
        default="results/robustness_supplement/label_noise",
    )
    args = parser.parse_args()

    rows = []
    for method in args.methods:
        data = load_predictions(args.results_root, method, args.window_size)
        labels = data["label"].to_numpy(dtype=int)
        probabilities = data["probability"].to_numpy(dtype=float)
        threshold = float(data["threshold"].median())

        base = metric_row(labels, probabilities, threshold)
        rows.append(
            {
                "experiment": "original",
                "method": method,
                "noise_rate": 0.0,
                "repeat": 0,
                "seed": args.seed,
                "n_changed_labels": 0,
                "n_samples": len(labels),
                **base,
            }
        )

        for rate in args.noise_rates:
            for repeat in range(args.repeats):
                seed = args.seed + repeat + int(rate * 1000)
                noisy_labels, n_changed = perturb_labels(
                    labels, "label_flip", rate, seed
                )
                row = metric_row(noisy_labels, probabilities, threshold)
                rows.append(
                    {
                        "experiment": "label_flip",
                        "method": method,
                        "noise_rate": rate,
                        "repeat": repeat,
                        "seed": seed,
                        "n_changed_labels": n_changed,
                        "n_samples": len(labels),
                        **row,
                    }
                )

        for repeat in range(args.repeats):
            seed = args.seed + 10000 + repeat
            permuted_labels, n_changed = perturb_labels(
                labels, "label_permutation", 1.0, seed
            )
            row = metric_row(permuted_labels, probabilities, threshold)
            rows.append(
                {
                    "experiment": "label_permutation",
                    "method": method,
                    "noise_rate": 1.0,
                    "repeat": repeat,
                    "seed": seed,
                    "n_changed_labels": int((permuted_labels != labels).sum()),
                    "n_samples": len(labels),
                    **row,
                }
            )

    detail, summary = summarize(
        rows, ["experiment", "method", "noise_rate"]
    )
    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "label_noise_detail.csv")
    summary_path = os.path.join(args.output_dir, "label_noise_summary.csv")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print("WROTE", detail_path)
    print("WROTE", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
