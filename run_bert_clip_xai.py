"""BERT+Image CLIP interpretability analysis for final preprocessing v2."""

import argparse
import csv
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from datasets.preprocessing import clean_text
from datasets.twitter_learn import CombinedTwitterDataset
from utils import load_args


METRIC_COLS = ["accuracy", "precision", "recall", "f1"]
USER_FEATURES = [
    "probability",
    "threshold",
    "signed_margin",
    "decision_margin",
    "confidence",
    "text_valid_count",
    "image_valid_count",
    "text_valid_ratio",
    "image_valid_ratio",
    "padding_amount",
]
SURROGATE_FEATURES = [
    "text_valid_count",
    "image_valid_count",
    "text_valid_ratio",
    "image_valid_ratio",
    "padding_amount",
    "threshold",
]


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, key):
        return self.get(key)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/final_preprocessing_v2")
    parser.add_argument("--pred_dir", default="results/final_preprocessing_v2/methods/bert_clip/w64")
    parser.add_argument("--config_file", default="configs/combos/bert_clip_contextvecnet.yaml")
    parser.add_argument("--group", default="final_preprocessing_v2")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--output_dir", default="interpretability_outputs/bert_clip_w64_main_xai")
    parser.add_argument("--calibration", default="none")
    parser.add_argument("--threshold_strategy", default="validation_f1")
    parser.add_argument("--top_n_cases", type=int, default=10)
    parser.add_argument("--mode", default="dryrun")
    return parser.parse_args()


def load_predictions(args):
    paths = sorted(glob.glob(os.path.join(args.pred_dir, "*_predictions.csv")))
    if not paths:
        raise SystemExit(f"No prediction files found in {args.pred_dir}")
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["calibration"] == args.calibration) & (df["threshold_strategy"] == args.threshold_strategy)].copy()
    if df.empty:
        raise SystemExit("No predictions matched calibration/threshold filters")
    return df


def fast_user_features(user_path, loaded_args):
    timeline_path = os.path.join(user_path, "timeline.txt")
    user = os.path.basename(user_path)
    if not os.path.isfile(timeline_path):
        return {
            "author": user,
            "text_valid_count": 0,
            "image_valid_count": 0,
            "text_valid_ratio": 0.0,
            "image_valid_ratio": 0.0,
            "padding_amount": loaded_args.window_size,
        }

    timeline = pd.read_json(timeline_path, lines=True)
    if timeline.empty:
        n = 0
        text_mask = np.zeros(0, dtype=float)
        image_mask = np.zeros(0, dtype=float)
    else:
        dates = [int(round(date.timestamp())) for date in timeline["created_at"].tolist()]
        order_idx = np.argsort(dates).ravel()
        high = len(dates) - loaded_args.window_size
        if high <= 0:
            start_idx = 0
            end_idx = len(dates)
            padding_amount = abs(high)
        else:
            start_idx = len(dates) - loaded_args.window_size
            end_idx = len(dates)
            padding_amount = 0
        idxs = order_idx[start_idx:end_idx]
        selected = timeline.iloc[idxs]
        texts = [clean_text(text) for text in selected["text"].tolist()]
        text_mask = np.array([1 if text else 0 for text in texts], dtype=float)
        image_valid = []
        for _, row in selected.iterrows():
            image_rel_path = row.get("image_path", "")
            if pd.notna(image_rel_path) and str(image_rel_path).strip() != "":
                img_path = os.path.join(user_path, str(image_rel_path))
            else:
                img_path = os.path.join(user_path, f"{row['id']}.jpg")
            image_valid.append(1 if os.path.isfile(img_path) else 0)
        image_mask = np.array(image_valid, dtype=float)
        n = len(selected)
    if 'padding_amount' not in locals():
        padding_amount = max(loaded_args.window_size - n, 0)
    if padding_amount > 0:
        text_mask = np.pad(text_mask, (0, padding_amount), constant_values=0)
        image_mask = np.pad(image_mask, (0, padding_amount), constant_values=0)
    return {
        "author": user,
        "text_valid_count": int(np.nansum(text_mask)),
        "image_valid_count": int(np.nansum(image_mask)),
        "text_valid_ratio": float(np.nanmean(text_mask)) if text_mask.size else np.nan,
        "image_valid_ratio": float(np.nanmean(image_mask)) if image_mask.size else np.nan,
        "padding_amount": float(padding_amount),
    }


def collect_dataset_features(base_args, folds):
    rows = []
    for fold in folds:
        cli_args = argparse.Namespace(
            name=f"bert_clip_w{base_args.window_size}_fold{fold}",
            group=base_args.group,
            notes="",
            mode=base_args.mode,
            epochs=1,
            batch_size=1,
            accumulation_steps=1,
            output_dir="unused",
            log_every=5,
            dataset=None,
            fold=int(fold),
            window_size=base_args.window_size,
            position_embeddings=None,
            image_embeddings_type=None,
            text_embeddings_type=None,
            config_file=base_args.config_file,
        )
        loaded_args, _ = load_args(cli_args)
        dataset = CombinedTwitterDataset(args=loaded_args, kind="test")
        for user_path in dataset.users:
            row = fast_user_features(user_path, loaded_args)
            row["fold"] = int(fold)
            rows.append(row)
    return pd.DataFrame(rows)


def add_error_columns(df):
    df["label"] = df["label"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["correct"] = df["label"] == df["prediction"]
    conditions = [
        (df["label"] == 1) & (df["prediction"] == 1),
        (df["label"] == 0) & (df["prediction"] == 0),
        (df["label"] == 0) & (df["prediction"] == 1),
        (df["label"] == 1) & (df["prediction"] == 0),
    ]
    df["error_type"] = np.select(conditions, ["TP", "TN", "FP", "FN"], default="unknown")
    df["signed_margin"] = df["probability"] - df["threshold"]
    df["decision_margin"] = df["signed_margin"].abs()
    df["confidence"] = np.where(df["prediction"] == 1, df["probability"], 1 - df["probability"])
    return df


def binary_metrics(frame):
    y_true = frame["label"].astype(int).to_numpy()
    y_pred = frame["prediction"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(frame) if len(frame) else np.nan
    return pd.Series({"n": len(frame), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})


def save_confusion(df, out_dir):
    label_order = [1, 0]
    label_names = ["depression", "non-depression"]
    cm = confusion_matrix(df["label"], df["prediction"], labels=label_order)
    plot_cm = cm.T
    pd.DataFrame(
        plot_cm,
        index=[f"Predicted {label}" for label in label_names],
        columns=[f"True {label}" for label in label_names],
    ).to_csv(out_dir / "confusion_matrix_counts.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    disp = ConfusionMatrixDisplay(plot_cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("BERT+Image CLIP Confusion Matrix")
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def save_error_and_boundary_plots(df, out_dir):
    counts = df["error_type"].value_counts().reindex(["TP", "TN", "FP", "FN"], fill_value=0)
    counts.to_csv(out_dir / "error_type_counts.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#2563eb", "#16a34a", "#dc2626", "#f97316"])
    ax.set_title("Prediction Outcome Counts")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "error_type_counts.png", dpi=300)
    plt.close(fig)

    summary = df.groupby("error_type")[["probability", "threshold", "signed_margin", "decision_margin", "confidence"]].agg(["count", "mean", "std", "min", "median", "max"])
    summary.to_csv(out_dir / "decision_boundary_by_error_type.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["TN", "FP", "FN", "TP"]
    data = [df.loc[df["error_type"] == key, "signed_margin"].dropna().to_numpy() for key in order]
    ax.boxplot(data, labels=order, showfliers=True)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_title("Signed Decision Margin by Outcome")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Probability - threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "decision_margin_by_error_type_boxplot.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    jitter = {"TN": -0.18, "FP": -0.06, "FN": 0.06, "TP": 0.18}
    colors = {"TN": "#16a34a", "TP": "#2563eb", "FP": "#dc2626", "FN": "#f97316"}
    for idx, row in df.iterrows():
        x = int(row["fold"]) + jitter.get(row["error_type"], 0)
        ax.scatter(x, row["probability"], color=colors.get(row["error_type"], "gray"), alpha=0.75, s=24)
    for fold, sub in df.groupby("fold"):
        ax.hlines(sub["threshold"].iloc[0], fold - 0.35, fold + 0.35, color="black", linewidth=2)
    ax.set_title("Decision Boundary by Fold and Outcome")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0, 1)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=7) for k, v in colors.items()]
    handles.append(plt.Line2D([0], [0], color='black', label='Threshold'))
    ax.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_dir / "decision_boundary_by_fold.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_case_tables(df, out_dir, top_n):
    cols = ["author", "fold", "label", "prediction", "error_type", "probability", "threshold", "signed_margin", "decision_margin", "confidence", "text_valid_count", "image_valid_count", "text_valid_ratio", "image_valid_ratio", "padding_amount"]
    for etype in ["TP", "TN", "FP", "FN"]:
        table = df[df["error_type"] == etype].sort_values("decision_margin", ascending=False)[cols].head(top_n)
        table.to_csv(out_dir / f"top_{etype}_cases.csv", index=False, encoding="utf-8-sig")
    df[~df["correct"]].sort_values("confidence", ascending=False)[cols].to_csv(out_dir / "errors_sorted_by_confidence.csv", index=False, encoding="utf-8-sig")
    df.sort_values("decision_margin", ascending=True)[cols].head(top_n * 2).to_csv(out_dir / "near_boundary_cases.csv", index=False, encoding="utf-8-sig")

    recommended = []
    labels = {
        "high_confidence_TP": df[df["error_type"] == "TP"].sort_values("decision_margin", ascending=False),
        "high_confidence_TN": df[df["error_type"] == "TN"].sort_values("decision_margin", ascending=False),
        "largest_margin_FP": df[df["error_type"] == "FP"].sort_values("decision_margin", ascending=False),
        "largest_margin_FN": df[df["error_type"] == "FN"].sort_values("decision_margin", ascending=False),
        "nearest_boundary_error": df[~df["correct"]].sort_values("decision_margin", ascending=True),
    }
    for reason, table in labels.items():
        if not table.empty:
            row = table.iloc[0][cols].copy()
            row["case_reason"] = reason
            recommended.append(row)
    if recommended:
        rec = pd.DataFrame(recommended)
        rec = rec[["case_reason"] + cols]
        rec.to_csv(out_dir / "recommended_attention_cases.csv", index=False, encoding="utf-8-sig")
        with open(out_dir / "extract_attention_commands.txt", "w", encoding="utf-8") as f:
            for fold, group in rec.groupby("fold"):
                users = " ".join(group["author"].astype(str).tolist())
                f.write(
                    "python extract_attention.py "
                    "--config_file configs/combos/bert_clip_contextvecnet.yaml "
                    f"--name bert_clip_w64_fold{int(fold)} "
                    "--group final_preprocessing_v2 "
                    f"--fold {int(fold)} --window_size 64 --kind test "
                    "--output_dir interpretability_outputs/bert_clip_w64_main_xai/attention_cases "
                    f"--users {users}\n"
                )


def save_surrogate_importance(df, out_dir):
    features = [col for col in SURROGATE_FEATURES if col in df.columns]
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median(numeric_only=True))
    y_error = (~df["correct"]).astype(int)
    y_prob = df["probability"].astype(float)

    clf = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=42, class_weight="balanced")
    clf.fit(X, y_error)
    pred = clf.predict(X)
    with open(out_dir / "surrogate_error_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_error, pred, zero_division=0))

    perm = permutation_importance(clf, X, y_error, n_repeats=30, random_state=42, scoring="f1")
    importance = pd.DataFrame({
        "feature": features,
        "rf_feature_importance": clf.feature_importances_,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    importance.to_csv(out_dir / "surrogate_error_feature_importance.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df = importance.sort_values("permutation_importance_mean")
    ax.barh(plot_df["feature"], plot_df["permutation_importance_mean"], xerr=plot_df["permutation_importance_std"], color="#2563eb", alpha=0.8)
    ax.set_title("Surrogate Feature Importance for Prediction Error")
    ax.set_xlabel("Permutation importance (F1 decrease)")
    fig.tight_layout()
    fig.savefig(out_dir / "surrogate_error_feature_importance.png", dpi=300)
    plt.close(fig)

    reg = RandomForestRegressor(n_estimators=500, max_depth=4, random_state=42)
    reg.fit(X, y_prob)
    perm_reg = permutation_importance(reg, X, y_prob, n_repeats=30, random_state=42, scoring="r2")
    reg_importance = pd.DataFrame({
        "feature": features,
        "rf_feature_importance": reg.feature_importances_,
        "permutation_importance_mean": perm_reg.importances_mean,
        "permutation_importance_std": perm_reg.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    reg_importance.to_csv(out_dir / "surrogate_probability_feature_importance.csv", index=False, encoding="utf-8-sig")

    try:
        import shap
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            values = shap_values[1]
        else:
            values = np.asarray(shap_values)
            if values.ndim == 3:
                values = values[:, :, 1]
        if values.ndim != 2:
            raise ValueError(f"Unsupported SHAP value shape: {values.shape}")
        shap_importance = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(out_dir / "shap_error_importance.csv", index=False, encoding="utf-8-sig")
        plt.figure(figsize=(7, 4))
        shap.summary_plot(values, X, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_error_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        with open(out_dir / "SHAP_NOT_RUN.txt", "w", encoding="utf-8") as f:
            f.write("SHAP was not run. Install shap in the environment to enable SHAP outputs.\n")
            f.write(f"Error: {exc}\n")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(args)
    folds = sorted(pred["fold"].astype(int).unique().tolist())
    features = collect_dataset_features(args, folds)
    df = pred.merge(features, on=["fold", "author"], how="left")
    df = add_error_columns(df)

    df.to_csv(out_dir / "bert_clip_w64_user_level_xai_table.csv", index=False, encoding="utf-8-sig")
    binary_metrics(df).to_csv(out_dir / "overall_metrics.csv", encoding="utf-8-sig")
    df.groupby("fold", dropna=False).apply(binary_metrics).reset_index().to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    df.groupby("error_type")[USER_FEATURES].agg(["count", "mean", "std", "min", "median", "max"]).to_csv(out_dir / "user_feature_profiles_by_error_type.csv", encoding="utf-8-sig")

    save_confusion(df, out_dir)
    save_error_and_boundary_plots(df, out_dir)
    save_case_tables(df, out_dir, args.top_n_cases)
    save_surrogate_importance(df, out_dir)

    print("WROTE", out_dir)
    print("rows", len(df), "folds", folds)


if __name__ == "__main__":
    main()