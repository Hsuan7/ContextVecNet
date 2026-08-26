from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

OUT_DIR = Path("interpretability_outputs/bert_clip_w64_main_xai/probability_shap")
INPUT = OUT_DIR / "probability_shap_input_table.csv"
df = pd.read_csv(INPUT)

def fit_and_plot(feature_cols, target_col, prefix, title, max_display=None):
    data = df[[target_col] + feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[target_col]).reset_index(drop=True)
    X = data[feature_cols]
    y = data[target_col]

    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X, y)
    pred = model.predict(X)
    metrics = pd.DataFrame([{
        "figure": prefix,
        "target": target_col,
        "n": len(data),
        "r2_in_sample": r2_score(y, pred),
        "mae_in_sample": mean_absolute_error(y, pred),
    }])
    metrics.to_csv(OUT_DIR / f"{prefix}_surrogate_metrics.csv", index=False, encoding="utf-8-sig")

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X))
    if values.ndim != 2:
        raise ValueError(values.shape)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUT_DIR / f"{prefix}_importance.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8.5, max(4.8, 0.42 * (max_display or len(feature_cols)) + 1.5)))
    shap.summary_plot(values, X, show=False, max_display=max_display or len(feature_cols))
    plt.title(title, pad=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix}_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.4, 4.8))
    bar = importance.head(max_display or len(feature_cols)).sort_values("mean_abs_shap")
    plt.barh(bar["feature"], bar["mean_abs_shap"], color="#2563eb", alpha=0.86)
    plt.xlabel("Mean |SHAP value|")
    plt.title(title.replace("SHAP Summary", "SHAP Importance"))
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix}_importance_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(prefix)
    print(metrics.to_string(index=False))
    print(importance.to_string(index=False))

# Keep only the features that are interpretable for the thesis story.
main_features = [
    "text_probability",
    "delta_p",
    "sd_similarity",
    "similarity_iqr",
    "max_similarity",
    "median_similarity",
    "high_alignment_ratio",
    "low_alignment_ratio",
]
fit_and_plot(
    main_features,
    "multimodal_probability",
    "shap_probability_refined_main",
    "Refined SHAP Summary for Predicted Depression Probability",
    max_display=8,
)

# Zoomed view: remove the dominant text baseline so the visual/image-text features are readable.
secondary_features = [
    "delta_p",
    "sd_similarity",
    "similarity_iqr",
    "max_similarity",
    "median_similarity",
    "high_alignment_ratio",
    "low_alignment_ratio",
    "mean_similarity",
]
fit_and_plot(
    secondary_features,
    "multimodal_probability",
    "shap_probability_secondary_zoom",
    "Secondary SHAP Summary Excluding Text Probability",
    max_display=8,
)