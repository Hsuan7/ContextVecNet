"""Evaluate one checkpoint with no, temperature, and Platt calibration."""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip import clip
from datasets.twitter_learn import CombinedTwitterDataset
from maple_model import BertImageCLIP, CustomCLIP, TextOnlyBERT
from utils import load_args, load_checkpoint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--calibrations",
        nargs="+",
        choices=["none", "temperature", "platt"],
        default=["none", "temperature", "platt"],
    )
    parser.add_argument(
        "--threshold_strategy",
        choices=["fixed_0.5", "validation_f1"],
        default="validation_f1",
    )
    parser.add_argument(
        "--include_platt_validation_f1",
        action="store_true",
    )
    parser.add_argument(
        "--include_none_validation_f1",
        action="store_true",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--position_embeddings", default=None)
    parser.add_argument("--image_embeddings_type", default=None)
    parser.add_argument("--text_embeddings_type", default=None)
    return parser.parse_args()


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_maple_config(path="configs/maple.yaml"):
    with open(path, encoding="utf-8") as file:
        return Config(yaml.safe_load(file))


def load_clip_to_cpu(cfg):
    model_path = clip._download(clip._MODELS[cfg.MODEL.BACKBONE.NAME])
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {
        "trainer": "MaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.MAPLE.N_CTX,
    }
    return clip.build_model(state_dict or model.state_dict(), design_details)


def build_model(args):
    if args.text_embeddings_type == "bert" and args.modality == "text":
        return TextOnlyBERT(args)

    cfg = load_maple_config()
    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.MAPLE.PREC in ("fp32", "amp"):
        clip_model.float()
    if args.text_embeddings_type == "bert" and args.modality == "both":
        return BertImageCLIP(cfg, clip_model, args)
    return CustomCLIP(cfg, clip_model, args)


def build_collate_fn(args):
    def collate(batch):
        sample = {
            "author": [item["author"] for item in batch],
            "images": [item["images"] for item in batch],
            "images_paths": [item["images_paths"] for item in batch],
            "texts": [item["texts"] for item in batch],
            "time": torch.tensor(
                [item["time"] for item in batch], dtype=torch.float32
            ),
            "label": torch.tensor(
                [item["label"] for item in batch], dtype=torch.float32
            ).view(-1, 1),
            "padding_amount": [item["padding_amount"] for item in batch],
            "text_mask": torch.tensor(
                [item["text_mask"] for item in batch], dtype=torch.float32
            ),
            "image_mask": torch.tensor(
                [item["image_mask"] for item in batch], dtype=torch.float32
            ),
        }
        if args.modality == "text":
            sample["image_mask"] = torch.zeros_like(sample["image_mask"])
        elif args.modality == "image":
            sample["text_mask"] = torch.zeros_like(sample["text_mask"])
        return sample

    return collate


def collect_logits(model, args, kind):
    dataset = CombinedTwitterDataset(args=args, kind=kind)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        collate_fn=build_collate_fn(args),
    )
    logits = []
    labels = []
    authors = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=kind):
            output = model(batch)
            logits.extend(output["logits"].detach().cpu().numpy().reshape(-1))
            labels.extend(batch["label"].numpy().reshape(-1))
            authors.extend(batch["author"])
    return np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=int), authors


def sigmoid(values):
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def fit_temperature(logits, labels):
    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        return metrics.log_loss(
            labels,
            sigmoid(logits / temperature),
            labels=[0, 1],
        )

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    return float(np.exp(result.x))


def fit_platt(logits, labels):
    model = LogisticRegression(solver="lbfgs", random_state=42)
    model.fit(logits.reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def select_f1_threshold(labels, probabilities):
    precision, recall, thresholds = metrics.precision_recall_curve(
        labels, probabilities
    )
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1], 1e-12, None
    )
    return float(thresholds[int(np.nanargmax(f1))])


def select_threshold(strategy, validation_labels, validation_probabilities):
    if strategy == "fixed_0.5":
        return 0.5, "fixed"
    if strategy == "validation_f1":
        return (
            select_f1_threshold(validation_labels, validation_probabilities),
            "validation",
        )
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def reliability_bins(labels, probabilities, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, edges[1:-1])
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        confidence = float(probabilities[mask].mean()) if count else np.nan
        accuracy = float(labels[mask].mean()) if count else np.nan
        rows.append(
            {
                "bin": bin_id,
                "bin_lower": edges[bin_id],
                "bin_upper": edges[bin_id + 1],
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
                "gap": abs(confidence - accuracy) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(labels, probabilities, n_bins=10):
    bins = reliability_bins(labels, probabilities, n_bins=n_bins)
    total = bins["count"].sum()
    if total == 0:
        return 0.0
    return float(
        ((bins["count"] / total) * bins["gap"].fillna(0.0)).sum()
    )


def save_reliability_diagram(
    labels,
    probabilities,
    output_dir,
    filename_prefix,
    title,
    n_bins=10,
):
    bins = reliability_bins(labels, probabilities, n_bins=n_bins)
    bins_path = os.path.join(output_dir, f"{filename_prefix}_reliability_bins.csv")
    plot_path = os.path.join(output_dir, f"{filename_prefix}_reliability_diagram.png")
    bins.to_csv(bins_path, index=False)

    nonempty = bins["count"] > 0
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect")
    plt.plot(
        bins.loc[nonempty, "confidence"],
        bins.loc[nonempty, "accuracy"],
        marker="o",
        label="Model",
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return bins_path, plot_path


def metric_row(
    args,
    calibration,
    threshold_strategy,
    parameters,
    labels,
    probabilities,
    threshold,
    threshold_source,
):
    predictions = (probabilities >= threshold).astype(int)
    nll = metrics.log_loss(labels, probabilities, labels=[0, 1])
    return {
        "experiment": args.name,
        "group": args.group,
        "fold": args.fold,
        "window_size": args.window_size,
        "model": args.model,
        "modality": args.modality,
        "text_embedding": args.text_embeddings_type,
        "image_embedding": args.image_embeddings_type,
        "calibration": calibration,
        "calibration_fit_split": (
            "none" if calibration == "none" else "validation"
        ),
        "threshold_strategy": threshold_strategy,
        "threshold_selection_split": threshold_source,
        "evaluation_split": "test",
        "threshold": threshold,
        "temperature": parameters.get("temperature"),
        "platt_a": parameters.get("platt_a"),
        "platt_b": parameters.get("platt_b"),
        "accuracy": metrics.accuracy_score(labels, predictions),
        "precision": metrics.precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": metrics.recall_score(labels, predictions, zero_division=0),
        "f1": metrics.f1_score(labels, predictions, zero_division=0),
        "auc": metrics.roc_auc_score(labels, probabilities),
        "nll": nll,
        "log_loss": nll,
        "brier": metrics.brier_score_loss(labels, probabilities),
        "ece_10": expected_calibration_error(labels, probabilities),
        "n_samples": len(labels),
    }


def main():
    fix_seed()
    args, _ = load_args(parse_args())
    model = build_model(args)
    checkpoint = load_checkpoint(args)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE).float().eval()

    val_logits, val_labels, _ = collect_logits(model, args, "valid")
    test_logits, test_labels, test_authors = collect_logits(model, args, "test")
    if np.unique(val_labels).size < 2:
        raise RuntimeError("Validation fold must contain both classes for calibration.")

    available_calibrations = {}
    if "none" in args.calibrations:
        available_calibrations["none"] = (
            sigmoid(val_logits),
            sigmoid(test_logits),
            {},
        )
    if "temperature" in args.calibrations:
        temperature = fit_temperature(val_logits, val_labels)
        available_calibrations["temperature"] = (
            sigmoid(val_logits / temperature),
            sigmoid(test_logits / temperature),
            {"temperature": temperature},
        )
    if "platt" in args.calibrations:
        platt_a, platt_b = fit_platt(val_logits, val_labels)
        available_calibrations["platt"] = (
            sigmoid(platt_a * val_logits + platt_b),
            sigmoid(platt_a * test_logits + platt_b),
            {"platt_a": platt_a, "platt_b": platt_b},
        )

    rows = []
    prediction_rows = []
    evaluation_specs = [
        (calibration, args.threshold_strategy)
        for calibration in args.calibrations
    ]
    if args.include_platt_validation_f1:
        if "platt" not in available_calibrations:
            raise ValueError(
                "--include_platt_validation_f1 requires Platt calibration."
            )
        evaluation_specs.append(("platt", "validation_f1"))
    if args.include_none_validation_f1:
        if "none" not in available_calibrations:
            raise ValueError(
                "--include_none_validation_f1 requires uncalibrated probabilities."
            )
        evaluation_specs.append(("none", "validation_f1"))

    for calibration, threshold_strategy in evaluation_specs:
        val_probabilities, test_probabilities, parameters = available_calibrations[
            calibration
        ]
        threshold, threshold_source = select_threshold(
            threshold_strategy,
            val_labels,
            val_probabilities,
        )
        rows.append(
            metric_row(
                args,
                calibration,
                threshold_strategy,
                parameters,
                test_labels,
                test_probabilities,
                threshold,
                threshold_source,
            )
        )
        for author, label, logit, probability in zip(
            test_authors, test_labels, test_logits, test_probabilities
        ):
            prediction_rows.append(
                {
                    "experiment": args.name,
                    "fold": args.fold,
                    "author": author,
                    "label": label,
                    "logit": logit,
                    "calibration": calibration,
                    "threshold_strategy": threshold_strategy,
                    "threshold_selection_split": threshold_source,
                    "probability": probability,
                    "threshold": threshold,
                    "prediction": int(probability >= threshold),
                }
            )

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"{args.name}_metrics.csv")
    predictions_path = os.path.join(
        args.output_dir, f"{args.name}_predictions.csv"
    )
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    # Reliability is threshold-independent, so save one diagram per calibration.
    for calibration in args.calibrations:
        _, test_probabilities, _ = available_calibrations[calibration]
        prefix = f"{args.name}_{calibration}"
        save_reliability_diagram(
            test_labels,
            test_probabilities,
            args.output_dir,
            prefix,
            (
                f"{args.name} | {calibration} | test fold {args.fold}\n"
                f"fit=validation, evaluation=test"
            ),
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("WROTE", metrics_path)
    print("WROTE", predictions_path)


if __name__ == "__main__":
    main()
