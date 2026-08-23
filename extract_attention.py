import argparse
import csv
import os
import re
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
from models.layers.attention import BertAttention
from utils import load_args, load_checkpoint


device = "cuda" if torch.cuda.is_available() else "cpu"


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
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
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

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
    model.float()

    checkpoint = load_checkpoint(args)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def custom_collate_fn(batch, args):
    user_names = [item["author"] for item in batch]
    labels = [item["label"] for item in batch]
    texts = [item["texts"] for item in batch]
    dates = [item["time"] for item in batch]
    images = [item["images"] for item in batch]
    images_paths = [item["images_paths"] for item in batch]
    padding_amount = [item["padding_amount"] for item in batch]
    text_mask = [item["text_mask"] for item in batch]
    image_mask = [item["image_mask"] for item in batch]

    sample = {
        "author": user_names,
        "images": images,
        "images_paths": images_paths,
        "texts": texts,
        "time": torch.tensor(dates, dtype=torch.float),
        "label": torch.tensor(labels, dtype=torch.float).to(device).view(-1, 1),
        "padding_amount": padding_amount,
        "text_mask": torch.tensor(text_mask, dtype=torch.float),
        "image_mask": torch.tensor(image_mask, dtype=torch.float),
    }

    if args.modality == "text":
        sample["image_mask"] = torch.zeros_like(sample["image_mask"])
    elif args.modality == "image":
        sample["text_mask"] = torch.zeros_like(sample["text_mask"])

    return sample


class AttentionRecorder:
    def __init__(self):
        self.records = []
        self.call_counts = {}
        self.handles = []

    def reset(self):
        self.records = []
        self.call_counts = {}

    def register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, BertAttention) and "multi_modal_transformer.layers" in name:
                handle = module.softmax.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, name):
        def hook(module, inputs, output):
            call_idx = self.call_counts.get(name, 0)
            self.call_counts[name] = call_idx + 1
            self.records.append(
                {
                    "name": name,
                    "call_idx": call_idx,
                    "kind": attention_kind(name, call_idx),
                    "attention": output.detach().float().cpu().numpy(),
                }
            )

        return hook


def attention_kind(name, call_idx):
    layer_match = re.search(r"layers\.(\d+)", name)
    layer = layer_match.group(1) if layer_match else "x"

    if "visual_attention" in name:
        direction = "text_to_image" if call_idx % 2 == 0 else "image_to_text"
    elif "lang_self_att" in name:
        direction = "text_self"
    elif "visn_self_att" in name:
        direction = "image_self"
    else:
        direction = "unknown"

    return f"layer{layer}_{direction}"


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def masked_mean_attention(attention, query_mask=None, key_mask=None):
    matrix = attention[0].mean(axis=0)

    if query_mask is not None:
        q = query_mask.astype(bool)
        matrix = matrix[q, :]
    if key_mask is not None:
        k = key_mask.astype(bool)
        matrix = matrix[:, k]
    return matrix


def save_heatmap(matrix, path, title, xlabel, ylabel):
    if matrix.size == 0:
        return

    width = max(6, min(14, matrix.shape[1] * 0.22))
    height = max(4, min(12, matrix.shape[0] * 0.22))

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def shorten_text(text, max_len=120):
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def record_timeline_rows(
    user_dir,
    user_id,
    texts,
    image_paths,
    text_mask,
    image_mask,
    records,
    output_csv,
):
    valid_text = text_mask.astype(bool)
    valid_image = image_mask.astype(bool)
    n = len(texts)

    text_received = np.zeros(n, dtype=float)
    image_received = np.zeros(n, dtype=float)
    text_self_received = np.zeros(n, dtype=float)
    image_self_received = np.zeros(n, dtype=float)

    for record in records:
        attn = record["attention"][0].mean(axis=0)
        kind = record["kind"]

        if kind.endswith("text_to_image"):
            image_received[: attn.shape[1]] += attn.mean(axis=0)
        elif kind.endswith("image_to_text"):
            text_received[: attn.shape[1]] += attn.mean(axis=0)
        elif kind.endswith("text_self"):
            text_self_received[: attn.shape[1]] += attn.mean(axis=0)
        elif kind.endswith("image_self"):
            image_self_received[: attn.shape[1]] += attn.mean(axis=0)

    rows = []
    for idx in range(n):
        rows.append(
            {
                "user_id": user_id,
                "position": idx,
                "text_valid": int(valid_text[idx]) if idx < len(valid_text) else 0,
                "image_valid": int(valid_image[idx]) if idx < len(valid_image) else 0,
                "text": shorten_text(texts[idx]),
                "image_path": image_paths[idx],
                "text_received_from_image_attention": float(text_received[idx]),
                "image_received_from_text_attention": float(image_received[idx]),
                "text_self_attention_received": float(text_self_received[idx]),
                "image_self_attention_received": float(image_self_received[idx]),
            }
        )

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return pd.DataFrame(rows)


def save_timeline_plot(timeline_df, path, title):
    valid_df = timeline_df[timeline_df["text_valid"] == 1].copy()
    if valid_df.empty:
        return

    x = valid_df["position"].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        x,
        valid_df["text_received_from_image_attention"],
        marker="o",
        label="Text received from image attention",
    )
    ax.plot(
        x,
        valid_df["image_received_from_text_attention"],
        marker="o",
        label="Image received from text attention",
    )
    ax.plot(
        x,
        valid_df["text_self_attention_received"],
        marker="o",
        alpha=0.75,
        label="Text self-attention received",
    )
    ax.set_title(title)
    ax.set_xlabel("Timeline position")
    ax.set_ylabel("Mean attention received")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def select_dataset_indices(dataset, target_users, max_samples):
    selected = []
    for idx, user_path in enumerate(dataset.users):
        user_id = os.path.basename(user_path)
        if target_users and user_id not in target_users:
            continue
        selected.append(idx)
        if not target_users and len(selected) >= max_samples:
            break
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ContextVecNet attention heatmaps.")
    parser.add_argument("--config_file", default="configs/combos/multi_only.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", default="modality_ablation")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--kind", default="test", choices=["train", "valid", "val", "test"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--users", nargs="*", default=None, help="Specific user ids to analyze.")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--position_embeddings", type=str, default=None)
    parser.add_argument("--image_embeddings_type", type=str, default=None)
    parser.add_argument("--text_embeddings_type", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--mode", type=str, default="dryrun")
    return parser.parse_args()


def main():
    args = parse_args()
    args, _ = load_args(args)
    args.batch_size = 1

    output_dir = Path(
        args.output_dir
        or f"attention_outputs/{args.group}_{args.name}_{args.kind}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    recorder = AttentionRecorder()
    recorder.register(model)

    dataset = CombinedTwitterDataset(args=args, kind=args.kind)
    selected_indices = select_dataset_indices(dataset, set(args.users or []), args.max_samples)

    print("selected samples:", len(selected_indices))
    if not selected_indices:
        raise SystemExit("No matching users were found.")

    summary_rows = []

    with torch.no_grad():
        for idx in selected_indices:
            sample = dataset[idx]
            user_id = sample["author"]
            user_slug = safe_name(user_id)
            user_dir = output_dir / user_slug
            user_dir.mkdir(parents=True, exist_ok=True)

            loader = DataLoader(
                [sample],
                batch_size=1,
                shuffle=False,
                collate_fn=lambda batch: custom_collate_fn(batch, args),
            )
            batch = next(iter(loader))

            recorder.reset()
            output = model(batch)

            logit = float(output["logits"].detach().cpu().numpy().reshape(-1)[0])
            raw_prob = float(output["probas"].detach().cpu().numpy().reshape(-1)[0])
            label = int(batch["label"].detach().cpu().numpy().reshape(-1)[0])

            text_mask = batch["text_mask"].detach().cpu().numpy().reshape(1, -1)[0]
            image_mask = batch["image_mask"].detach().cpu().numpy().reshape(1, -1)[0]
            texts = batch["texts"][0]
            image_paths = batch["images_paths"][0]

            for record in recorder.records:
                attn = record["attention"]
                kind = record["kind"]

                if kind.endswith("text_to_image"):
                    matrix = masked_mean_attention(attn, text_mask, image_mask)
                    xlabel = "Image timeline position"
                    ylabel = "Text timeline position"
                elif kind.endswith("image_to_text"):
                    matrix = masked_mean_attention(attn, image_mask, text_mask)
                    xlabel = "Text timeline position"
                    ylabel = "Image timeline position"
                elif kind.endswith("text_self"):
                    matrix = masked_mean_attention(attn, text_mask, text_mask)
                    xlabel = "Text timeline position"
                    ylabel = "Text timeline position"
                elif kind.endswith("image_self"):
                    matrix = masked_mean_attention(attn, image_mask, image_mask)
                    xlabel = "Image timeline position"
                    ylabel = "Image timeline position"
                else:
                    matrix = masked_mean_attention(attn)
                    xlabel = "Key position"
                    ylabel = "Query position"

                heatmap_path = user_dir / f"{kind}.png"
                save_heatmap(
                    matrix,
                    heatmap_path,
                    title=f"{user_id} | {kind}",
                    xlabel=xlabel,
                    ylabel=ylabel,
                )

            timeline_csv = user_dir / "timeline_attention.csv"
            timeline_df = record_timeline_rows(
                user_dir=user_dir,
                user_id=user_id,
                texts=texts,
                image_paths=image_paths,
                text_mask=text_mask,
                image_mask=image_mask,
                records=recorder.records,
                output_csv=timeline_csv,
            )
            save_timeline_plot(
                timeline_df,
                user_dir / "timeline_attention.png",
                title=f"{user_id} | timeline attention summary",
            )

            summary_rows.append(
                {
                    "user_id": user_id,
                    "label": label,
                    "raw_prob": raw_prob,
                    "logit": logit,
                    "text_valid_count": int(text_mask.sum()),
                    "image_valid_count": int(image_mask.sum()),
                    "output_dir": str(user_dir),
                }
            )
            print("saved:", user_id, "->", user_dir)

    recorder.close()

    summary_path = output_dir / "attention_summary.csv"
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("summary:", summary_path)


if __name__ == "__main__":
    main()
