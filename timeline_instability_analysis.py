import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.twitter_learn import CombinedTwitterDataset
from extract_attention import build_model, custom_collate_fn
from make_timeline_visualization import calibrated_prob, get_calibration
from utils import load_args


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare model-inferred timeline risk instability between positive "
            "and negative users."
        )
    )
    parser.add_argument("--config_file", default="configs/combos/multi_only.yaml")
    parser.add_argument("--name_template", default="multi_w64_fold{fold}_maskpool_posw_b2_fixpool")
    parser.add_argument("--group", default="modality_ablation")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--kind", default="test", choices=["valid", "val", "test"])
    parser.add_argument("--output_dir", default="timeline_instability_outputs/multi_w64_maskpool_posw_b2_fixpool")
    parser.add_argument("--mode", choices=["prefix", "rolling"], default="prefix")
    parser.add_argument("--rolling_window", type=int, default=8)
    parser.add_argument("--max_users_per_class", type=int, default=None)
    parser.add_argument("--max_users_total", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--position_embeddings", type=str, default=None)
    parser.add_argument("--image_embeddings_type", type=str, default=None)
    parser.add_argument("--text_embeddings_type", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clone_batch_with_position_mask(batch, keep_positions):
    masked = {}
    for key, value in batch.items():
        masked[key] = value.clone() if torch.is_tensor(value) else value

    text_mask = masked["text_mask"].reshape(masked["text_mask"].shape[0], -1)
    image_mask = masked["image_mask"].reshape(masked["image_mask"].shape[0], -1)

    keep = torch.zeros_like(text_mask)
    keep[:, keep_positions] = 1.0

    text_mask *= keep
    image_mask *= keep
    return masked


def forward_prob(model, batch, calibration):
    output = model(batch)
    logit = float(output["logits"].detach().cpu().numpy().reshape(-1)[0])
    raw_prob = float(output["probas"].detach().cpu().numpy().reshape(-1)[0])
    prob = calibrated_prob(logit, raw_prob, calibration)
    return logit, raw_prob, prob


def valid_positions_from_batch(batch):
    text_mask = batch["text_mask"].detach().cpu().numpy().reshape(1, -1)[0]
    image_mask = batch["image_mask"].detach().cpu().numpy().reshape(1, -1)[0]
    return np.where((text_mask > 0) | (image_mask > 0))[0]


def get_keep_positions(valid_positions, position, mode, rolling_window):
    if mode == "prefix":
        return valid_positions[valid_positions <= position]

    start = max(int(position) - rolling_window + 1, int(valid_positions.min()))
    keep = valid_positions[(valid_positions >= start) & (valid_positions <= position)]
    return keep


def trajectory_metrics(risk_values):
    risk = np.asarray(risk_values, dtype=float)
    risk = risk[np.isfinite(risk)]
    if len(risk) == 0:
        return {
            "trajectory_len": 0,
            "risk_mean": np.nan,
            "risk_std": np.nan,
            "risk_min": np.nan,
            "risk_max": np.nan,
            "risk_range": np.nan,
            "mean_abs_delta": np.nan,
            "max_abs_delta": np.nan,
            "spike_count_q75": np.nan,
        }

    delta = np.diff(risk)
    abs_delta = np.abs(delta)
    if len(abs_delta):
        q75 = np.quantile(abs_delta, 0.75)
        spike_count = int((abs_delta > q75).sum())
        mean_abs_delta = float(abs_delta.mean())
        max_abs_delta = float(abs_delta.max())
    else:
        spike_count = 0
        mean_abs_delta = 0.0
        max_abs_delta = 0.0

    return {
        "trajectory_len": int(len(risk)),
        "risk_mean": float(risk.mean()),
        "risk_std": float(risk.std(ddof=0)),
        "risk_min": float(risk.min()),
        "risk_max": float(risk.max()),
        "risk_range": float(risk.max() - risk.min()),
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "spike_count_q75": spike_count,
    }


def select_user_indices(dataset, max_users_per_class, max_users_total, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.labels)
    all_indices = np.arange(len(labels))

    if max_users_per_class is not None:
        selected = []
        for label in [0, 1]:
            candidates = all_indices[labels == label]
            if len(candidates) > max_users_per_class:
                candidates = rng.choice(candidates, size=max_users_per_class, replace=False)
            selected.extend(candidates.tolist())
        selected = np.asarray(selected)
        rng.shuffle(selected)
    else:
        selected = all_indices

    if max_users_total is not None and len(selected) > max_users_total:
        selected = rng.choice(selected, size=max_users_total, replace=False)

    return [int(i) for i in selected]


def save_trajectory_plot(traj_df, path, title, threshold):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(traj_df["position"], traj_df["risk_prob"], marker="o", linewidth=1.5)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.2, label="threshold")
    ax.set_xlabel("Timeline position")
    ax.set_ylabel("Model-inferred positive probability")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_group_plots(user_df, out_dir):
    metrics = ["risk_mean", "risk_std", "mean_abs_delta", "max_abs_delta", "risk_range"]
    label_names = {0: "negative", 1: "positive"}

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [
            user_df.loc[user_df["label"] == label, metric].dropna().to_numpy()
            for label in [0, 1]
        ]
        ax.boxplot(data, labels=[label_names[0], label_names[1]], showmeans=True)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by class")
        plt.tight_layout()
        fig.savefig(out_dir / f"{metric}_by_class.png", dpi=200)
        plt.close(fig)


def mann_whitney_or_fallback(pos, neg):
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    result = {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "positive_mean": float(pos.mean()) if len(pos) else np.nan,
        "negative_mean": float(neg.mean()) if len(neg) else np.nan,
        "positive_median": float(np.median(pos)) if len(pos) else np.nan,
        "negative_median": float(np.median(neg)) if len(neg) else np.nan,
        "mean_difference_pos_minus_neg": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else np.nan,
        "test": "none",
        "p_value": np.nan,
    }
    try:
        from scipy.stats import mannwhitneyu

        stat, p_value = mannwhitneyu(pos, neg, alternative="two-sided")
        result["test"] = "mannwhitneyu"
        result["statistic"] = float(stat)
        result["p_value"] = float(p_value)
    except Exception:
        result["statistic"] = np.nan
    return result


def main():
    cli_args = parse_args()
    out_dir = Path(cli_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    all_user_rows = []

    for fold in cli_args.folds:
        args = argparse.Namespace(**vars(cli_args))
        args.name = cli_args.name_template.format(fold=fold)
        args.fold = fold
        args, _ = load_args(args)
        args.batch_size = 1

        print(f"===== Fold {fold} | {args.name} =====")
        calibration = get_calibration(args)
        model = build_model(args)
        model.eval()

        dataset = CombinedTwitterDataset(args=args, kind=args.kind)
        indices = select_user_indices(
            dataset,
            max_users_per_class=cli_args.max_users_per_class,
            max_users_total=cli_args.max_users_total,
            seed=cli_args.seed + fold,
        )

        for idx in indices:
            sample = dataset[idx]
            user_id = sample["author"]
            label = int(sample["label"])

            loader = DataLoader(
                [sample],
                batch_size=1,
                shuffle=False,
                collate_fn=lambda batch: custom_collate_fn(batch, args),
            )
            batch = next(iter(loader))
            valid_positions = valid_positions_from_batch(batch)
            if len(valid_positions) == 0:
                continue

            with torch.no_grad():
                baseline_logit, baseline_raw_prob, baseline_prob = forward_prob(
                    model, batch, calibration
                )

                rows = []
                for position in valid_positions:
                    keep_positions = get_keep_positions(
                        valid_positions,
                        int(position),
                        mode=cli_args.mode,
                        rolling_window=cli_args.rolling_window,
                    )
                    masked_batch = clone_batch_with_position_mask(batch, keep_positions)
                    logit, raw_prob, prob = forward_prob(model, masked_batch, calibration)
                    rows.append(
                        {
                            "fold": fold,
                            "user_id": user_id,
                            "label": label,
                            "position": int(position),
                            "risk_prob": prob,
                            "raw_prob": raw_prob,
                            "logit": logit,
                            "n_posts_used": int(len(keep_positions)),
                            "mode": cli_args.mode,
                        }
                    )

            traj_df = pd.DataFrame(rows)
            metrics = trajectory_metrics(traj_df["risk_prob"].to_numpy())
            user_row = {
                "fold": fold,
                "user_id": user_id,
                "label": label,
                "baseline_prob": baseline_prob,
                "baseline_raw_prob": baseline_raw_prob,
                "baseline_logit": baseline_logit,
                "threshold": calibration["threshold"],
                "baseline_pred": int(baseline_prob >= calibration["threshold"]),
                "valid_post_count": int(len(valid_positions)),
                "mode": cli_args.mode,
            }
            user_row.update(metrics)
            all_user_rows.append(user_row)

            user_slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in user_id)
            traj_path = traj_dir / f"fold{fold}_{user_slug}_trajectory.csv"
            traj_df.to_csv(traj_path, index=False, encoding="utf-8-sig")
            save_trajectory_plot(
                traj_df,
                traj_dir / f"fold{fold}_{user_slug}_trajectory.png",
                title=f"{user_id} | {cli_args.mode} risk trajectory",
                threshold=calibration["threshold"],
            )
            print("saved trajectory:", user_id, "label=", label)

    user_df = pd.DataFrame(all_user_rows)
    user_df.to_csv(out_dir / "user_instability_metrics.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for metric in ["risk_mean", "risk_std", "mean_abs_delta", "max_abs_delta", "risk_range"]:
        pos = user_df.loc[user_df["label"] == 1, metric]
        neg = user_df.loc[user_df["label"] == 0, metric]
        row = {"metric": metric}
        row.update(mann_whitney_or_fallback(pos, neg))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "positive_vs_negative_instability_summary.csv", index=False, encoding="utf-8-sig")
    save_group_plots(user_df, out_dir)

    print("saved:", out_dir / "user_instability_metrics.csv")
    print("saved:", out_dir / "positive_vs_negative_instability_summary.csv")
    print(summary_df)


if __name__ == "__main__":
    main()
