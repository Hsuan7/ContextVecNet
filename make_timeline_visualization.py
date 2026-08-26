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
from extract_attention import (
    AttentionRecorder,
    build_model,
    custom_collate_fn,
    record_timeline_rows,
    safe_name,
    save_heatmap,
)
from utils import load_args, load_checkpoint


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create user-level timeline visualization with attention and occlusion risk."
    )
    parser.add_argument("--config_file", default="configs/combos/multi_only.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", default="modality_ablation")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--kind", default="test", choices=["train", "valid", "val", "test"])
    parser.add_argument("--user", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--position_embeddings", type=str, default=None)
    parser.add_argument("--image_embeddings_type", type=str, default=None)
    parser.add_argument("--text_embeddings_type", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--mode", type=str, default="dryrun")
    return parser.parse_args()


def get_calibration(args):
    checkpoint = load_checkpoint(args)
    metrics = checkpoint.get("metrics", {})
    platt_a = metrics.get("val/platt_a")
    platt_b = metrics.get("val/platt_b")
    threshold = metrics.get("val/best_threshold")

    if threshold is None:
        threshold = 0.5
        print("[WARN] Validation best threshold not found; using 0.5.")
    if platt_a is None or platt_b is None:
        platt_a = None
        platt_b = None
        print("[WARN] Validation Platt parameters not found; using raw model probabilities.")

    return {
        "platt_a": None if platt_a is None else float(platt_a),
        "platt_b": None if platt_b is None else float(platt_b),
        "threshold": float(threshold),
    }


def calibrated_prob(logit, raw_prob, calibration):
    platt_a = calibration["platt_a"]
    platt_b = calibration["platt_b"]
    if platt_a is None or platt_b is None:
        return float(raw_prob)
    return float(sigmoid(platt_a * logit + platt_b))


def find_user_sample(dataset, user_id):
    for idx, user_path in enumerate(dataset.users):
        if Path(user_path).name == user_id:
            return dataset[idx]
    raise ValueError(f"User {user_id!r} not found in {dataset.kind} fold.")


def clone_batch_with_occlusion(batch, position):
    occluded = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            occluded[key] = value.clone()
        else:
            occluded[key] = value

    text_mask = occluded["text_mask"].reshape(occluded["text_mask"].shape[0], -1)
    image_mask = occluded["image_mask"].reshape(occluded["image_mask"].shape[0], -1)
    text_mask[:, position] = 0
    image_mask[:, position] = 0
    return occluded


def forward_prob(model, batch, calibration):
    output = model(batch)
    logit = float(output["logits"].detach().cpu().numpy().reshape(-1)[0])
    raw_prob = float(output["probas"].detach().cpu().numpy().reshape(-1)[0])
    prob = calibrated_prob(logit, raw_prob, calibration)
    return logit, raw_prob, prob


def minmax(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    low = np.nanmin(values)
    high = np.nanmax(values)
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def add_attention_summary_columns(df):
    attention_cols = [
        "text_received_from_image_attention",
        "image_received_from_text_attention",
        "text_self_attention_received",
        "image_self_attention_received",
    ]
    df["attention_score"] = df[attention_cols].astype(float).mean(axis=1)
    valid = (df["text_valid"].astype(int) == 1) | (df["image_valid"].astype(int) == 1)
    df["attention_score_norm"] = 0.0
    df.loc[valid, "attention_score_norm"] = minmax(df.loc[valid, "attention_score"])
    return df


def save_timeline_plot(df, output_path, title, baseline_prob, threshold):
    valid_df = df[(df["text_valid"].astype(int) == 1) | (df["image_valid"].astype(int) == 1)].copy()
    if valid_df.empty:
        return

    x = valid_df["position"].astype(int).to_numpy()
    risk = valid_df["risk_contribution"].astype(float).to_numpy()
    attention = valid_df["attention_score_norm"].astype(float).to_numpy()

    colors = ["#d62728" if value >= 0 else "#1f77b4" for value in risk]

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.bar(x, risk, color=colors, alpha=0.72, label="Risk contribution")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Timeline position")
    ax1.set_ylabel("Occlusion risk contribution")

    ax2 = ax1.twinx()
    ax2.plot(x, attention, color="#2ca02c", marker="o", linewidth=1.6, label="Attention score (normalized)")
    ax2.set_ylabel("Normalized attention score")
    ax2.set_ylim(-0.05, 1.05)

    highest_risk = valid_df.sort_values("risk_contribution", ascending=False).iloc[0]
    highest_attention = valid_df.sort_values("attention_score_norm", ascending=False).iloc[0]

    ax1.scatter(
        [int(highest_risk["position"])],
        [float(highest_risk["risk_contribution"])],
        s=120,
        color="#8b0000",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        label="Highest risk post",
    )
    ax2.scatter(
        [int(highest_attention["position"])],
        [float(highest_attention["attention_score_norm"])],
        s=120,
        color="#006400",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        label="Highest attention post",
    )

    ax1.set_title(f"{title}\nprob={baseline_prob:.4f}, threshold={threshold:.4f}")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower right",
        framealpha=0.88,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_post_cards(df, output_path):
    valid_df = df[(df["text_valid"].astype(int) == 1) | (df["image_valid"].astype(int) == 1)].copy()
    highest_risk = valid_df.sort_values("risk_contribution", ascending=False).iloc[0]
    highest_attention = valid_df.sort_values("attention_score_norm", ascending=False).iloc[0]
    most_protective = valid_df.sort_values("risk_contribution", ascending=True).iloc[0]

    rows = [
        ("highest_risk_post", highest_risk),
        ("highest_attention_post", highest_attention),
        ("most_protective_post", most_protective),
    ]

    fieldnames = ["case_type"] + list(df.columns)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for label, row in rows:
            item = {"case_type": label}
            item.update(row.to_dict())
            writer.writerow(item)


def main():
    args = parse_args()
    args, _ = load_args(args)
    args.batch_size = 1

    output_dir = Path(
        args.output_dir
        or f"timeline_outputs/{args.group}_{args.name}_{args.kind}/{safe_name(args.user)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = get_calibration(args)
    model = build_model(args)

    dataset = CombinedTwitterDataset(args=args, kind=args.kind)
    sample = find_user_sample(dataset, args.user)

    loader = DataLoader(
        [sample],
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
    )
    batch = next(iter(loader))

    recorder = AttentionRecorder()
    recorder.register(model)
    recorder.reset()

    model.eval()
    with torch.no_grad():
        logit, raw_prob, prob = forward_prob(model, batch, calibration)

    text_mask = batch["text_mask"].detach().cpu().numpy().reshape(1, -1)[0]
    image_mask = batch["image_mask"].detach().cpu().numpy().reshape(1, -1)[0]
    valid_positions = np.where((text_mask > 0) | (image_mask > 0))[0]

    # Save attention heatmaps from the baseline forward pass.
    for record in recorder.records:
        attention = record["attention"][0].mean(axis=0)
        path = output_dir / f"{record['kind']}.png"
        save_heatmap(
            attention,
            path,
            title=f"{args.user} | {record['kind']}",
            xlabel="Key timeline position",
            ylabel="Query timeline position",
        )

    timeline_csv = output_dir / "timeline_attention.csv"
    timeline_df = record_timeline_rows(
        user_dir=output_dir,
        user_id=args.user,
        texts=batch["texts"][0],
        image_paths=batch["images_paths"][0],
        text_mask=text_mask,
        image_mask=image_mask,
        records=recorder.records,
        output_csv=timeline_csv,
    )
    timeline_df = add_attention_summary_columns(timeline_df)

    occluded_probs = {}
    with torch.no_grad():
        for position in valid_positions:
            occluded_batch = clone_batch_with_occlusion(batch, int(position))
            _, _, occluded_prob = forward_prob(model, occluded_batch, calibration)
            occluded_probs[int(position)] = occluded_prob

    timeline_df["baseline_prob"] = prob
    timeline_df["threshold"] = calibration["threshold"]
    timeline_df["occluded_prob"] = timeline_df["position"].map(occluded_probs)
    timeline_df["risk_contribution"] = timeline_df["baseline_prob"] - timeline_df["occluded_prob"]
    timeline_df["risk_contribution_norm"] = 0.0
    valid = timeline_df["occluded_prob"].notna()
    timeline_df.loc[valid, "risk_contribution_norm"] = minmax(
        timeline_df.loc[valid, "risk_contribution"]
    )

    timeline_df.to_csv(output_dir / "timeline_visualization.csv", index=False, encoding="utf-8-sig")
    save_timeline_plot(
        timeline_df,
        output_dir / "timeline_risk_attention.png",
        title=f"{args.user} | timeline risk and attention",
        baseline_prob=prob,
        threshold=calibration["threshold"],
    )
    save_post_cards(timeline_df, output_dir / "selected_posts.csv")

    summary = {
        "user_id": args.user,
        "label": int(batch["label"].detach().cpu().numpy().reshape(-1)[0]),
        "raw_prob": raw_prob,
        "calibrated_prob": prob,
        "logit": logit,
        "threshold": calibration["threshold"],
        "pred": int(prob >= calibration["threshold"]),
        "text_valid_count": int(text_mask.sum()),
        "image_valid_count": int(image_mask.sum()),
        "highest_risk_position": int(
            timeline_df.loc[timeline_df["risk_contribution"].idxmax(), "position"]
        ),
        "highest_attention_position": int(
            timeline_df.loc[timeline_df["attention_score_norm"].idxmax(), "position"]
        ),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "timeline_summary.csv", index=False, encoding="utf-8-sig")

    recorder.close()
    print("saved timeline visualization to:", output_dir)
    print(summary)


if __name__ == "__main__":
    main()
