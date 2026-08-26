"""Surrogate SHAP analysis for predicted depression probability."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

OUT_DIR = Path("interpretability_outputs/bert_clip_w64_main_xai/probability_shap")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALIGN_PATH = Path("results/final_preprocessing_v2/image_text_alignment_analysis/user_level_alignment_delta.csv")
XAI_PATH = Path("interpretability_outputs/bert_clip_w64_main_xai/bert_clip_w64_user_level_xai_table.csv")

align = pd.read_csv(ALIGN_PATH)
xai = pd.read_csv(XAI_PATH)

# Merge model outputs/alignment features with sequence coverage features.
df = align.merge(
    xai[[
        "fold",
        "author",
        "text_valid_count",
        "image_valid_count",
        "text_valid_ratio",
        "image_valid_ratio",
        "padding_amount",
    ]],
    on=["fold", "author"],
    how="left",
)

# User-level ratios derived from alignment counts.
df["high_alignment_ratio"] = df["high_post_count"] / df["valid_post_count"].replace(0, np.nan)
df["low_alignment_ratio"] = df["low_post_count"] / df["valid_post_count"].replace(0, np.nan)
df["similarity_iqr"] = df["q3_similarity"] - df["q1_similarity"]
df["abs_delta_p"] = df["delta_p"].abs()

feature_cols = [
    "text_probability",
    "delta_p",
    "abs_delta_p",
    "mean_similarity",
    "median_similarity",
    "sd_similarity",
    "similarity_iqr",
    "min_similarity",
    "max_similarity",
    "high_alignment_ratio",
    "low_alignment_ratio",
    "valid_post_count",
    "text_valid_count",
    "image_valid_count",
    "text_valid_ratio",
    "image_valid_ratio",
    "padding_amount",
]

target_col = "multimodal_probability"
model_df = df[["fold", "author", "label", target_col] + feature_cols].copy()
model_df = model_df.replace([np.inf, -np.inf], np.nan)
for col in feature_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
    model_df[col] = model_df[col].fillna(model_df[col].median())
model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)

X = model_df[feature_cols]
y = model_df[target_col]

surrogate = RandomForestRegressor(
    n_estimators=800,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42,
)
surrogate.fit(X, y)
pred = surrogate.predict(X)
metrics = pd.DataFrame([
    {
        "target": target_col,
        "n": len(model_df),
        "r2_in_sample": r2_score(y, pred),
        "mae_in_sample": mean_absolute_error(y, pred),
        "target_mean": y.mean(),
        "target_std": y.std(),
    }
])
metrics.to_csv(OUT_DIR / "surrogate_probability_metrics.csv", index=False, encoding="utf-8-sig")

perm = permutation_importance(
    surrogate,
    X,
    y,
    n_repeats=50,
    random_state=42,
    scoring="r2",
)
perm_df = pd.DataFrame(
    {
        "feature": feature_cols,
        "rf_feature_importance": surrogate.feature_importances_,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }
).sort_values("permutation_importance_mean", ascending=False)
perm_df.to_csv(OUT_DIR / "surrogate_probability_feature_importance.csv", index=False, encoding="utf-8-sig")

explainer = shap.TreeExplainer(surrogate)
shap_values = explainer.shap_values(X)
values = np.asarray(shap_values)
if values.ndim != 2:
    raise ValueError(f"Unexpected SHAP shape for regressor: {values.shape}")

importance = pd.DataFrame(
    {
        "feature": feature_cols,
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
    }
).sort_values("mean_abs_shap", ascending=False)
importance.to_csv(OUT_DIR / "shap_probability_importance.csv", index=False, encoding="utf-8-sig")

model_df.to_csv(OUT_DIR / "probability_shap_input_table.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 5))
shap.summary_plot(values, X, show=False, max_display=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_probability_summary.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 4.5))
bar_df = importance.head(12).sort_values("mean_abs_shap")
plt.barh(bar_df["feature"], bar_df["mean_abs_shap"], color="#2563eb", alpha=0.85)
plt.xlabel("Mean |SHAP value|")
plt.title("SHAP Feature Importance for Predicted Depression Probability")
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_probability_importance_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print("WROTE", OUT_DIR)
print(metrics.to_string(index=False))
print(importance.head(12).to_string(index=False))