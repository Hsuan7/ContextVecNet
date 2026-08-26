"""Per-post occlusion contribution plus attention timeline plots for BERT+Image CLIP cases."""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from clip import clip
from datasets.twitter_learn import CombinedTwitterDataset
from maple_model import BertImageCLIP, CustomCLIP
from utils import load_args, load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTN_COLS = [
    "text_received_from_image_attention",
    "image_received_from_text_attention",
    "text_self_attention_received",
    "image_self_attention_received",
]


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self[key] = value


def load_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as file:
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
    cfg = load_config("configs/maple.yaml")
    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.MAPLE.PREC in ["fp32", "amp"]:
        clip_model.float()
    if args.text_embeddings_type == "bert" and args.modality == "both":
        model = BertImageCLIP(cfg, clip_model, args)
    else:
        model = CustomCLIP(cfg, clip_model, args)
    checkpoint = load_checkpoint(args)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE).float().eval()
    return model


def collate_one(sample, args):
    batch = {
        "author": [sample["author"]],
        "images": [sample["images"]],
        "images_paths": [sample["images_paths"]],
        "texts": [sample["texts"]],
        "time": torch.tensor([sample["time"]], dtype=torch.float32),
        "label": torch.tensor([sample["label"]], dtype=torch.float32).view(-1, 1),
        "padding_amount": [sample["padding_amount"]],
        "text_mask": torch.tensor([sample["text_mask"]], dtype=torch.float32),
        "image_mask": torch.tensor([sample["image_mask"]], dtype=torch.float32),
    }
    if args.modality == "text":
        batch["image_mask"] = torch.zeros_like(batch["image_mask"])
    elif args.modality == "image":
        batch["text_mask"] = torch.zeros_like(batch["text_mask"])
    return batch


def clone_batch(batch):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def prob_from_model(model, batch):
    with torch.no_grad():
        output = model(batch)
    if "probas" in output:
        return float(output["probas"].detach().cpu().numpy().reshape(-1)[0])
    logits = output["logits"].detach().cpu().numpy().reshape(-1)[0]
    return float(1.0 / (1.0 + np.exp(-logits)))


def normalized_attention(user_attention_csv):
    attn = pd.read_csv(user_attention_csv)
    for col in ATTN_COLS:
        attn[col] = pd.to_numeric(attn[col], errors="coerce").fillna(0.0)
    attn["combined_attention"] = attn[ATTN_COLS].mean(axis=1)
    valid = attn["text_valid"].astype(int).eq(1)
    values = attn["combined_attention"].to_numpy(float)
    values[~valid.to_numpy()] = 0.0
    vmin = np.nanmin(values) if np.isfinite(values).any() else 0.0
    vmax = np.nanmax(values) if np.isfinite(values).any() else 0.0
    if vmax > vmin:
        attn["attention_score_norm"] = (values - vmin) / (vmax - vmin)
    else:
        attn["attention_score_norm"] = 0.0
    return attn[["position", "text", "image_path", "text_valid", "image_valid", "attention_score_norm"]]


def find_sample(dataset, user_id):
    for idx, user_path in enumerate(dataset.users):
        if os.path.basename(user_path) == user_id:
            return dataset[idx]
    raise ValueError(f"User {user_id} not found in test split")


def make_args(base_args, fold):
    parser_ns = argparse.Namespace(
        name=f"bert_clip_w{base_args.window_size}_fold{fold}",
        group=base_args.group,
        notes="",
        mode="dryrun",
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
    loaded_args, _ = load_args(parser_ns)
    return loaded_args


def plot_case(case_row, model, dataset, loaded_args, attention_root, out_root):
    user_id = case_row["author"]
    sample = find_sample(dataset, user_id)
    batch = collate_one(sample, loaded_args)
    base_prob = prob_from_model(model, batch)
    text_mask = batch["text_mask"].detach().cpu().numpy().reshape(-1)
    image_mask = batch["image_mask"].detach().cpu().numpy().reshape(-1)
    valid_positions = np.where((text_mask > 0) | (image_mask > 0))[0]

    rows = []
    for pos in valid_positions:
        masked = clone_batch(batch)
        masked["text_mask"][:, :, :, pos] = 0
        masked["image_mask"][:, :, :, pos] = 0
        masked_prob = prob_from_model(model, masked)
        rows.append(
            {
                "user_id": user_id,
                "position": int(pos),
                "base_probability": base_prob,
                "masked_probability": masked_prob,
                "risk_contribution": base_prob - masked_prob,
                "text_valid": int(text_mask[pos] > 0),
                "image_valid": int(image_mask[pos] > 0),
            }
        )
    contrib = pd.DataFrame(rows)
    attn_csv = attention_root / user_id / "timeline_attention.csv"
    if not attn_csv.exists():
        raise FileNotFoundError(attn_csv)
    attn = normalized_attention(attn_csv)
    merged = contrib.merge(attn, on=["position", "text_valid", "image_valid"], how="left")

    user_dir = out_root / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(user_dir / "occlusion_attention_contribution.csv", index=False, encoding="utf-8-sig")

    x = merged["position"].to_numpy()
    y = merged["risk_contribution"].to_numpy()
    attn_y = merged["attention_score_norm"].fillna(0).to_numpy()
    colors = np.where(y >= 0, "#ef4444", "#3b82f6")

    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    ax1.bar(x, y, color=colors, alpha=0.75, label="Risk contribution")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Timeline position")
    ax1.set_ylabel("Occlusion risk contribution")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, attn_y, color="#16a34a", marker="o", linewidth=1.5, markersize=3.5, label="Attention score (normalized)")
    ax2.set_ylabel("Normalized attention score")
    ax2.set_ylim(-0.03, 1.05)

    if len(merged):
        top_risk_idx = int(np.nanargmax(np.abs(y)))
        top_attn_idx = int(np.nanargmax(attn_y))
        ax1.scatter([x[top_risk_idx]], [y[top_risk_idx]], color="#991b1b", s=70, zorder=5, label="Largest |risk contribution|")
        ax2.scatter([x[top_attn_idx]], [attn_y[top_attn_idx]], color="#166534", s=70, zorder=5, label="Highest attention post")

    title = (
        f"{user_id} | {case_row['error_type']} | "
        f"prob={float(case_row['probability']):.3f}, threshold={float(case_row['threshold']):.3f}"
    )
    ax1.set_title(title)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    fig.tight_layout()
    fig.savefig(user_dir / "occlusion_contribution_attention_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    top_cols = ["user_id", "position", "risk_contribution", "attention_score_norm", "masked_probability", "text", "image_path"]
    merged.assign(abs_risk_contribution=merged["risk_contribution"].abs()).sort_values(
        "abs_risk_contribution", ascending=False
    )[top_cols + ["abs_risk_contribution"]].head(15).to_csv(
        user_dir / "top_occlusion_contribution_posts.csv", index=False, encoding="utf-8-sig"
    )
    return user_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="configs/combos/bert_clip_contextvecnet.yaml")
    parser.add_argument("--group", default="final_preprocessing_v2")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--case_csv", default="interpretability_outputs/bert_clip_w64_main_xai/recommended_attention_cases.csv")
    parser.add_argument("--attention_root", default="interpretability_outputs/bert_clip_w64_main_xai/attention_cases")
    parser.add_argument("--output_dir", default="interpretability_outputs/bert_clip_w64_main_xai/occlusion_attention_cases")
    args = parser.parse_args()

    cases = pd.read_csv(args.case_csv)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    attention_root = Path(args.attention_root)

    summary = []
    for fold, fold_cases in cases.groupby("fold"):
        loaded_args = make_args(args, int(fold))
        model = build_model(loaded_args)
        dataset = CombinedTwitterDataset(args=loaded_args, kind="test")
        for _, case_row in fold_cases.iterrows():
            user_dir = plot_case(case_row, model, dataset, loaded_args, attention_root, out_root)
            summary.append({"user_id": case_row["author"], "fold": int(fold), "error_type": case_row["error_type"], "output_dir": str(user_dir)})
    pd.DataFrame(summary).to_csv(out_root / "occlusion_attention_case_summary.csv", index=False, encoding="utf-8-sig")
    print("WROTE", out_root)


if __name__ == "__main__":
    main()