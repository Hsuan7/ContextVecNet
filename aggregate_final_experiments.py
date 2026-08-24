"""Aggregate unified final-experiment metric files across folds."""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_calibration_comparison import reliability_bins


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "nll",
    "log_loss",
    "brier",
    "ece_10",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    args = parser.parse_args()

    paths = sorted(
        glob.glob(
            os.path.join(args.results_root, "**", "*_metrics.csv"),
            recursive=True,
        )
    )
    if not paths:
        raise SystemExit(f"No metric files found below {args.results_root}")

    by_fold = pd.concat(
        [pd.read_csv(path).assign(source_file=path) for path in paths],
        ignore_index=True,
    )
    if "nll" not in by_fold.columns and "log_loss" in by_fold.columns:
        by_fold["nll"] = by_fold["log_loss"]
    if "log_loss" not in by_fold.columns and "nll" in by_fold.columns:
        by_fold["log_loss"] = by_fold["nll"]
    by_fold = by_fold.drop_duplicates(
        subset=["experiment", "fold", "calibration", "threshold_strategy"],
        keep="first",
    )
    group_columns = [
        "experiment",
        "window_size",
        "model",
        "modality",
        "text_embedding",
        "image_embedding",
        "calibration",
        "threshold_strategy",
    ]
    by_fold["method"] = by_fold["experiment"].str.replace(
        r"_w\d+_fold\d+$", "", regex=True
    )
    group_columns[0] = "method"

    summary = (
        by_fold.groupby(group_columns, dropna=False)[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    fold_counts = (
        by_fold.groupby(group_columns, dropna=False)["fold"]
        .nunique()
        .reset_index(name="fold_count")
    )
    summary = summary.merge(fold_counts, on=group_columns, how="left")
    summary["complete_5fold"] = summary["fold_count"] == 5

    os.makedirs(args.results_root, exist_ok=True)
    by_fold_path = os.path.join(args.results_root, "all_results_by_fold.csv")
    summary_path = os.path.join(args.results_root, "all_results_summary.csv")
    by_fold.to_csv(by_fold_path, index=False)
    summary.to_csv(summary_path, index=False)

    method_summary = summary[
        (summary["window_size"] == 64)
        & (summary["calibration"] == "none")
        & (summary["threshold_strategy"] == "validation_f1")
    ].copy()
    comparison_names = {
        ("contextvecnet", "none"): "Full ContextVecNet",
        ("text_bert", "none"): "Text-only BERT-base Chinese",
        ("bert_clip", "none"): "BERT-base Chinese + Image CLIP ContextVecNet",
        ("text_clip", "none"): "Text-only CLIP",
        ("image_clip", "none"): "Image-only CLIP",
        ("concat", "none"): "Text+Image concat",
        ("lstm", "none"): "LSTM-only",
    }
    method_summary.insert(
        0,
        "comparison_name",
        [
            comparison_names.get((method, calibration), method)
            for method, calibration in zip(
                method_summary["method"], method_summary["calibration"]
            )
        ],
    )
    calibration_summary = summary[
        (summary["method"] == "contextvecnet")
        & (summary["window_size"] == 64)
        & (summary["threshold_strategy"] == "fixed_0.5")
    ]
    window_summary = summary[
        (summary["method"] == "contextvecnet")
        & (summary["calibration"] == "none")
        & (summary["threshold_strategy"] == "validation_f1")
    ]

    method_path = os.path.join(
        args.results_root, "method_and_modality_comparison.csv"
    )
    calibration_path = os.path.join(
        args.results_root, "calibration_comparison.csv"
    )
    window_path = os.path.join(
        args.results_root, "window_size_comparison.csv"
    )
    method_summary.to_csv(method_path, index=False)
    calibration_summary.to_csv(calibration_path, index=False)
    window_summary.to_csv(window_path, index=False)

    prediction_paths = sorted(
        glob.glob(
            os.path.join(
                args.results_root,
                "methods",
                "contextvecnet",
                "w64",
                "*_predictions.csv",
            )
        )
    )
    if prediction_paths:
        predictions = pd.concat(
            [pd.read_csv(path) for path in prediction_paths],
            ignore_index=True,
        )
        predictions = predictions[
            predictions["threshold_strategy"] == "fixed_0.5"
        ]
        reliability_dir = os.path.join(
            args.results_root, "calibration_reliability"
        )
        os.makedirs(reliability_dir, exist_ok=True)
        calibration_curves = {}
        for calibration in ("none", "temperature", "platt"):
            subset = predictions[predictions["calibration"] == calibration]
            if subset.empty:
                continue
            bins = reliability_bins(
                subset["label"].to_numpy(),
                subset["probability"].to_numpy(),
                n_bins=10,
            )
            calibration_curves[calibration] = (subset, bins)
            bins.to_csv(
                os.path.join(
                    reliability_dir,
                    f"contextvecnet_w64_{calibration}_5fold_reliability_bins.csv",
                ),
                index=False,
            )

        plt.figure(figsize=(7, 7))
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            label="Perfect",
        )
        for calibration, (_, bins) in calibration_curves.items():
            nonempty = bins["count"] > 0
            plt.plot(
                bins.loc[nonempty, "confidence"],
                bins.loc[nonempty, "accuracy"],
                marker="o",
                label=calibration,
            )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed positive rate")
        plt.title("ContextVecNet w64 reliability diagram (5 folds combined)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        calibration_curve_path = os.path.join(
            reliability_dir,
            "contextvecnet_w64_calibration_5fold_calibration_curve.png",
        )
        plt.savefig(calibration_curve_path, dpi=300, bbox_inches="tight")
        plt.close()

        figure, (curve_axis, histogram_axis) = plt.subplots(
            2,
            1,
            figsize=(8, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        curve_axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            label="Perfect",
        )
        for calibration, (subset, bins) in calibration_curves.items():
            nonempty = bins["count"] > 0
            curve_axis.plot(
                bins.loc[nonempty, "confidence"],
                bins.loc[nonempty, "accuracy"],
                marker="o",
                label=calibration,
            )
            histogram_axis.hist(
                subset["probability"],
                bins=np.linspace(0, 1, 11),
                alpha=0.4,
                label=calibration,
            )
        curve_axis.set_xlim(0, 1)
        curve_axis.set_ylim(0, 1)
        curve_axis.set_ylabel("Observed positive rate")
        curve_axis.set_title(
            "ContextVecNet w64 reliability diagram (5 folds combined)"
        )
        curve_axis.grid(True, alpha=0.3)
        curve_axis.legend()
        histogram_axis.set_xlabel("Predicted probability")
        histogram_axis.set_ylabel("Count")
        histogram_axis.legend()
        reliability_diagram_path = os.path.join(
            reliability_dir,
            "contextvecnet_w64_calibration_5fold_reliability_diagram.png",
        )
        figure.savefig(
            reliability_diagram_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)
    print("WROTE", by_fold_path)
    print("WROTE", summary_path)
    print("WROTE", method_path)
    print("WROTE", calibration_path)
    print("WROTE", window_path)
    if prediction_paths:
        print("WROTE", calibration_curve_path)
        print("WROTE", reliability_diagram_path)


if __name__ == "__main__":
    main()
