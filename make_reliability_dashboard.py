import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.calibration import calibration_curve

parser = argparse.ArgumentParser(description="Build a reliability dashboard from error analysis CSV.")
parser.add_argument(
    "--csv_path",
    type=str,
    default="results/ig/modality_ablation_multi_w64_fold0_maskpool_test_error_analysis.csv",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="reliability_dashboard",
)
args = parser.parse_args()

# ===== 1. 讀取 error_analysis.csv =====
csv_path = args.csv_path
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# ===== 2. Performance =====
accuracy = metrics.accuracy_score(df["y_true"], df["y_pred"])
precision = metrics.precision_score(df["y_true"], df["y_pred"], zero_division=0)
recall = metrics.recall_score(df["y_true"], df["y_pred"], zero_division=0)
f1 = metrics.f1_score(df["y_true"], df["y_pred"], zero_division=0)
auc = metrics.roc_auc_score(df["y_true"], df["y_prob"])

performance_df = pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "auc": auc,
    "n_samples": len(df),
    "n_errors": int((df["correct"] == False).sum()),
    "n_high_conf_errors": int(df["high_conf_error"].sum())
}])

print("\n===== Performance =====")
print(performance_df)

performance_df.to_csv(
    os.path.join(output_dir, "performance_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ===== 3. Confidence distribution =====
plt.figure(figsize=(8, 5))
plt.hist(df[df["correct"] == True]["confidence"], bins=20, alpha=0.6, label="Correct")
plt.hist(df[df["correct"] == False]["confidence"], bins=20, alpha=0.6, label="Error")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("Confidence Distribution: Correct vs Error")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confidence_distribution_correct_vs_error.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
df.boxplot(column="confidence", by="error_type")
plt.title("Confidence by Error Type")
plt.suptitle("")
plt.xlabel("Error Type")
plt.ylabel("Confidence")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confidence_by_error_type.png"), dpi=300)
plt.close()

# ===== 4. Calibration curve =====
prob_true, prob_pred = calibration_curve(df["y_true"], df["y_prob"], n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Reliability / Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=300)
plt.close()

# ===== 5. Error slicing =====
error_type_count = df["error_type"].value_counts().reset_index()
error_type_count.columns = ["error_type", "count"]

error_type_count.to_csv(
    os.path.join(output_dir, "error_type_count.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n===== Error Type Count =====")
print(error_type_count)

# ===== 6. Subgroup metrics function =====
def subgroup_metrics(data, group_col):
    rows = []

    for group_name, g in data.groupby(group_col):
        if len(g) == 0:
            continue

        row = {
            "subgroup_variable": group_col,
            "group": group_name,
            "count": len(g),
            "accuracy": metrics.accuracy_score(g["y_true"], g["y_pred"]),
            "precision": metrics.precision_score(g["y_true"], g["y_pred"], zero_division=0),
            "recall": metrics.recall_score(g["y_true"], g["y_pred"], zero_division=0),
            "f1": metrics.f1_score(g["y_true"], g["y_pred"], zero_division=0),
            "error_rate": 1 - metrics.accuracy_score(g["y_true"], g["y_pred"]),
            "high_conf_error_count": int(g["high_conf_error"].sum()),
            "mean_confidence": g["confidence"].mean(),
        }

        if len(g["y_true"].unique()) == 2:
            row["auc"] = metrics.roc_auc_score(g["y_true"], g["y_prob"])
        else:
            row["auc"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)

subgroup_tables = []

def safe_qcut(series, q, label_prefix):
    try:
        # 先嘗試 qcut
        cat = pd.qcut(series, q=q, duplicates="drop")
        n_bins = len(cat.cat.categories)

        labels = [f"{label_prefix}_{i}" for i in range(n_bins)]

        return pd.qcut(series, q=n_bins, labels=labels, duplicates="drop")

    except Exception as e:
        print(f"{label_prefix} qcut failed:", e)
        return pd.Series(["all"] * len(series))

# text_valid_count 分組
df["text_valid_group"] = safe_qcut(df["text_valid_count"], 3, "text")
df["image_valid_group"] = safe_qcut(df["image_valid_count"], 3, "image")
df["padding_group"] = safe_qcut(df["padding_amount"], 3, "padding")
subgroup_tables.append(subgroup_metrics(df, "text_valid_group"))
subgroup_tables.append(subgroup_metrics(df, "image_valid_group"))
subgroup_tables.append(subgroup_metrics(df, "padding_group"))
# if "text_valid_count" in df.columns:
#     df["text_valid_group"] = pd.qcut(
#         df["text_valid_count"],
#         q=3,
#         labels=["low_text", "mid_text", "high_text"],
#         duplicates="drop"
#     )
#     subgroup_tables.append(subgroup_metrics(df, "text_valid_group"))

# # image_valid_count 分組
# if "image_valid_count" in df.columns:
#     df["image_valid_group"] = pd.qcut(
#         df["image_valid_count"],
#         q=3,
#         labels=["low_image", "mid_image", "high_image"],
#         duplicates="drop"
#     )
#     subgroup_tables.append(subgroup_metrics(df, "image_valid_group"))

# # padding_amount 分組
# if "padding_amount" in df.columns:
#     try:
#         df["padding_group"] = pd.qcut(
#             df["padding_amount"],
#             q=3,
#             labels=["low_padding", "mid_padding", "high_padding"],
#             duplicates="drop"
#         )
#         subgroup_tables.append(subgroup_metrics(df, "padding_group"))
#     except Exception as e:
#         print("padding_amount 無法分組：", e)

if subgroup_tables:
    subgroup_df = pd.concat(subgroup_tables, ignore_index=True)
else:
    subgroup_df = pd.DataFrame()

subgroup_df.to_csv(
    os.path.join(output_dir, "subgroup_metrics.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n===== Subgroup Metrics =====")
print(subgroup_df)

# ===== 7. High-confidence errors =====
high_conf_errors = df[df["high_conf_error"] == True].sort_values(
    "confidence",
    ascending=False
)

high_conf_errors.to_csv(
    os.path.join(output_dir, "high_confidence_errors.csv"),
    index=False,
    encoding="utf-8-sig"
)

top3 = high_conf_errors.head(3)
top3.to_csv(
    os.path.join(output_dir, "top3_high_confidence_errors.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n===== Top 3 High-confidence Errors =====")
print(top3)

# ===== 8. 輸出 Excel dashboard =====
with pd.ExcelWriter(os.path.join(output_dir, "reliability_dashboard.xlsx")) as writer:
    performance_df.to_excel(writer, sheet_name="performance", index=False)
    error_type_count.to_excel(writer, sheet_name="error_type_count", index=False)
    subgroup_df.to_excel(writer, sheet_name="subgroup_metrics", index=False)
    high_conf_errors.to_excel(writer, sheet_name="high_conf_errors", index=False)
    top3.to_excel(writer, sheet_name="top3_cases", index=False)
    df.to_excel(writer, sheet_name="all_predictions", index=False)

print("\nSaved reliability dashboard to:", output_dir)
