import csv
from pathlib import Path


PATH = Path(
    "timeline_instability_outputs/"
    "multi_w64_maskpool_posw_b2_fixpool/"
    "user_instability_metrics.csv"
)

NUMERIC = [
    "fold",
    "label",
    "baseline_prob",
    "threshold",
    "risk_mean",
    "risk_std",
    "mean_abs_delta",
    "max_abs_delta",
    "risk_range",
    "valid_post_count",
]


def load_rows():
    rows = []
    with PATH.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            for key in NUMERIC:
                if row.get(key, "") != "":
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def show(title, rows):
    print(f"\n=== {title} ===")
    for row in rows:
        print(
            "fold={fold:.0f} user={user_id} label={label:.0f} "
            "base={baseline_prob:.4f} thr={threshold:.4f} "
            "mean={risk_mean:.4f} std={risk_std:.4f} "
            "mad={mean_abs_delta:.4f} maxd={max_abs_delta:.4f} "
            "range={risk_range:.4f} valid={valid_post_count:.0f}".format(**row)
        )


def main():
    rows = load_rows()
    full = [r for r in rows if r["valid_post_count"] == 64]
    pos = [r for r in full if r["label"] == 1]
    neg = [r for r in full if r["label"] == 0]

    print("all rows:", len(rows))
    print("full 64 rows:", len(full))
    print("positive full 64:", len(pos))
    print("negative full 64:", len(neg))

    show(
        "positive full 64, highest risk_mean",
        sorted(pos, key=lambda r: r["risk_mean"], reverse=True)[:12],
    )
    show(
        "positive full 64, highest mean_abs_delta",
        sorted(pos, key=lambda r: r["mean_abs_delta"], reverse=True)[:12],
    )
    show(
        "negative full 64, lowest risk_mean",
        sorted(neg, key=lambda r: r["risk_mean"])[:12],
    )
    show(
        "negative full 64, lowest mean_abs_delta",
        sorted(neg, key=lambda r: r["mean_abs_delta"])[:12],
    )


if __name__ == "__main__":
    main()
