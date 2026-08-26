import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_trajectory(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    "position": int(float(row["position"])),
                    "risk_prob": float(row["risk_prob"]),
                    "label": int(float(row["label"])),
                    "user_id": row["user_id"],
                    "fold": int(float(row["fold"])),
                    "n_posts_used": int(float(row.get("n_posts_used", 0))),
                }
            )
    return rows


def label_name(label):
    return "positive" if int(label) == 1 else "negative"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot two model-inferred risk trajectories in one figure."
    )
    parser.add_argument("--positive_csv", required=True)
    parser.add_argument("--negative_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive_name", default=None)
    parser.add_argument("--negative_name", default=None)
    parser.add_argument("--threshold_positive", type=float, default=None)
    parser.add_argument("--threshold_negative", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--fig_width", type=float, default=10.0)
    parser.add_argument("--fig_height", type=float, default=6.0)
    parser.add_argument(
        "--y_margin",
        type=float,
        default=0.025,
        help="Extra y-axis margin around observed min/max risk.",
    )
    parser.add_argument(
        "--hide_warmup_from_scale",
        action="store_true",
        help="Set y-axis limits using only positions after warm-up.",
    )
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    pos_rows = read_trajectory(args.positive_csv)
    neg_rows = read_trajectory(args.negative_csv)

    pos_name = args.positive_name or pos_rows[0]["user_id"]
    neg_name = args.negative_name or neg_rows[0]["user_id"]

    pos_x = np.array([r["position"] for r in pos_rows])
    pos_y = np.array([r["risk_prob"] for r in pos_rows])
    neg_x = np.array([r["position"] for r in neg_rows])
    neg_y = np.array([r["risk_prob"] for r in neg_rows])

    if args.hide_warmup_from_scale:
        scale_values = np.concatenate(
            [pos_y[pos_x >= args.warmup], neg_y[neg_x >= args.warmup]]
        )
    else:
        scale_values = np.concatenate([pos_y, neg_y])

    if args.y_min is not None and args.y_max is not None:
        y_min = args.y_min
        y_max = args.y_max
    else:
        y_min = max(0.0, float(np.nanmin(scale_values)) - args.y_margin)
        y_max = min(1.0, float(np.nanmax(scale_values)) + args.y_margin)
        if y_max - y_min < 0.05:
            center = (y_min + y_max) / 2
            y_min = max(0.0, center - 0.025)
            y_max = min(1.0, center + 0.025)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    if args.warmup > 0:
        ax.axvspan(-0.5, args.warmup - 0.5, color="#f2f2f2", alpha=0.9, label="warm-up")
        ax.axvline(args.warmup - 0.5, color="#bdbdbd", linestyle=":", linewidth=1)

    ax.plot(
        pos_x,
        pos_y,
        color="#d62728",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=f"{pos_name} ({label_name(pos_rows[0]['label'])})",
    )
    ax.plot(
        neg_x,
        neg_y,
        color="#1f77b4",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=f"{neg_name} ({label_name(neg_rows[0]['label'])})",
    )

    if args.threshold_positive is not None:
        ax.axhline(
            args.threshold_positive,
            color="#d62728",
            linestyle="--",
            linewidth=1.1,
            alpha=0.65,
            label=f"{pos_name} threshold",
        )
    if args.threshold_negative is not None:
        ax.axhline(
            args.threshold_negative,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.1,
            alpha=0.65,
            label=f"{neg_name} threshold",
        )

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, max(pos_x.max(), neg_x.max()) + 0.5)
    ax.set_xlabel("Timeline position")
    ax.set_ylabel("Model-inferred positive probability")
    ax.set_title("Timeline Risk Trajectory Comparison")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    print("saved:", output)


if __name__ == "__main__":
    main()
