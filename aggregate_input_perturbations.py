"""Aggregate fold-level input perturbation metrics."""

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    args = parser.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.input_dir, "*_input_perturbation_metrics.csv"))
    )
    if not paths:
        raise FileNotFoundError(
            f"No input perturbation metric files found under {args.input_dir}"
        )

    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "macro_f1",
        "f1",
        "auc",
        "ece_10",
        "brier",
        "nll",
    ]
    summary = (
        data.groupby(["perturbation"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if part != ""]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]

    detail_path = os.path.join(args.input_dir, "input_perturbation_detail.csv")
    summary_path = os.path.join(args.input_dir, "input_perturbation_summary.csv")
    data.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print("WROTE", detail_path)
    print("WROTE", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
