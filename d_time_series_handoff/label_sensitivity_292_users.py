from pathlib import Path

import numpy as np
import pandas as pd


DST_ROOT = Path(r"D:\時間序列\ContextVecNet_Instagram_filtered_new")
OUTPUT_DIR = Path(r"D:\時間序列\labeled\label_sensitivity_292_users")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2021-01-01 00:00:00+00:00"
END_DATE = "2026-06-01 23:59:59+00:00"
WINDOW_DAYS = 14
MIN_POSTS = 3
HIGH_RISK_SCORE_THRESHOLD = 0.7
USE_PSEUDO_LABEL_IF_AVAILABLE = True

MIN_HIGH_RISK_GRID = [1, 2, 3]
MEAN_SCORE_THRESHOLD_GRID = [0.5, 0.6, 0.7]
MAIN_RULE = {"min_high_risk": 2, "mean_score_threshold": 0.6}


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        wins = 0.0
        for p in pos:
            wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        return float(wins / (len(pos) * len(neg)))


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def load_292_user_metadata(dst_root):
    frames = []
    for split, y_true in [("positive", 1), ("negative", 0)]:
        split_dir = dst_root / split
        for user_dir in sorted(split_dir.iterdir()):
            metadata_path = user_dir / "metadata.csv"
            if not metadata_path.exists():
                continue
            d = pd.read_csv(metadata_path)
            d["folder_split"] = split
            d["y_true"] = y_true
            d["user_dir"] = user_dir.name
            frames.append(d)

    if not frames:
        raise FileNotFoundError(f"No metadata.csv files found under {dst_root}")

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["username", "post_id", "taken_at", "p_depression_tcal"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["username"] = df["username"].fillna(df["user_dir"]).astype(str)
    df["taken_at"] = pd.to_datetime(df["taken_at"], utc=True, errors="coerce")
    df["p_depression_tcal"] = pd.to_numeric(df["p_depression_tcal"], errors="coerce")
    if "pseudo_label" in df.columns:
        df["pseudo_label"] = pd.to_numeric(df["pseudo_label"], errors="coerce")

    df = df.dropna(subset=["username", "post_id", "taken_at", "p_depression_tcal"]).copy()
    df = df.drop_duplicates(subset=["username", "post_id"]).copy()
    df = df[(df["taken_at"] >= pd.Timestamp(START_DATE)) & (df["taken_at"] <= pd.Timestamp(END_DATE))].copy()

    if USE_PSEUDO_LABEL_IF_AVAILABLE and "pseudo_label" in df.columns:
        df["is_high_risk"] = np.where(
            df["pseudo_label"].notna(),
            (df["pseudo_label"] == 1).astype(int),
            (df["p_depression_tcal"] >= HIGH_RISK_SCORE_THRESHOLD).astype(int),
        )
    else:
        df["is_high_risk"] = (df["p_depression_tcal"] >= HIGH_RISK_SCORE_THRESHOLD).astype(int)

    return df.sort_values(["username", "taken_at", "post_id"]).reset_index(drop=True)


def label_single_user_by_window(df_user, min_high_risk, mean_score_threshold):
    g = df_user.sort_values("taken_at").reset_index(drop=True)
    positive_windows = []
    best_window_score = 0.0
    best_window_mean = float(g["p_depression_tcal"].mean()) if len(g) else np.nan
    best_window_high_risk = 0
    times = g["taken_at"].tolist()
    scores = g["p_depression_tcal"].to_numpy(dtype=float)
    high_flags = g["is_high_risk"].to_numpy(dtype=int)
    score_cumsum = np.concatenate([[0.0], np.cumsum(scores)])
    high_cumsum = np.concatenate([[0], np.cumsum(high_flags)])
    j = 0

    for i in range(len(g)):
        window_start = times[i]
        window_end = window_start + pd.Timedelta(days=WINDOW_DAYS)
        while j < len(g) and times[j] < window_end:
            j += 1

        n_posts = j - i
        n_high = int(high_cumsum[j] - high_cumsum[i])
        mean_score = float((score_cumsum[j] - score_cumsum[i]) / n_posts) if n_posts else np.nan

        high_ratio = n_high / min_high_risk if min_high_risk else 0.0
        mean_ratio = mean_score / mean_score_threshold if mean_score_threshold else 0.0
        window_score = min(high_ratio, mean_ratio)
        if window_score > best_window_score:
            best_window_score = float(window_score)
            best_window_mean = mean_score
            best_window_high_risk = n_high

        if n_posts >= MIN_POSTS and n_high >= min_high_risk and mean_score >= mean_score_threshold:
            positive_windows.append(
                {
                    "username": g.loc[0, "username"],
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "n_posts": int(n_posts),
                    "n_high_risk_posts": int(n_high),
                    "mean_p_depression_tcal": mean_score,
                    "post_ids": "|".join(g.iloc[i:j]["post_id"].astype(str).tolist()),
                }
            )

    return {
        "username": g.loc[0, "username"],
        "y_true": int(g["y_true"].iloc[0]),
        "folder_split": g["folder_split"].iloc[0],
        "num_posts_total": int(len(g)),
        "num_high_risk_posts_total": int(g["is_high_risk"].sum()),
        "mean_p_depression_tcal_total": float(g["p_depression_tcal"].mean()),
        "user_label": int(len(positive_windows) > 0),
        "num_positive_windows": int(len(positive_windows)),
        "best_window_score": best_window_score,
        "best_window_mean": best_window_mean,
        "best_window_high_risk": int(best_window_high_risk),
    }, positive_windows


def run_one_scheme(df, min_high_risk, mean_score_threshold):
    scheme_name = f"high{min_high_risk}_mean{str(mean_score_threshold).replace('.', 'p')}"
    user_rows = []
    window_rows = []

    for _, g in df.groupby("username", sort=True):
        user_row, positive_windows = label_single_user_by_window(g, min_high_risk, mean_score_threshold)
        user_row["scheme_name"] = scheme_name
        user_row["window_days"] = WINDOW_DAYS
        user_row["min_posts"] = MIN_POSTS
        user_row["min_high_risk"] = min_high_risk
        user_row["mean_score_threshold"] = mean_score_threshold
        user_rows.append(user_row)

        for row in positive_windows:
            row["scheme_name"] = scheme_name
            row["min_high_risk"] = min_high_risk
            row["mean_score_threshold"] = mean_score_threshold
            window_rows.append(row)

    user_df = pd.DataFrame(user_rows)
    windows_df = pd.DataFrame(window_rows)

    user_df.to_csv(OUTPUT_DIR / f"user_summary_{scheme_name}.csv", index=False, encoding="utf-8-sig")
    windows_df.to_csv(OUTPUT_DIR / f"positive_windows_{scheme_name}.csv", index=False, encoding="utf-8-sig")

    y_true = user_df["y_true"].astype(int)
    y_pred = user_df["user_label"].astype(int)
    metrics = binary_metrics(y_true, y_pred)

    n_users = int(user_df["username"].nunique())
    n_positive_users = int(user_df["user_label"].sum())
    class_ratio = n_positive_users / n_users if n_users else np.nan

    return {
        "scheme_name": scheme_name,
        "window_days": WINDOW_DAYS,
        "min_posts": MIN_POSTS,
        "min_high_risk": min_high_risk,
        "mean_score_threshold": mean_score_threshold,
        "n_users": n_users,
        "n_true_positive_users": int((user_df["y_true"] == 1).sum()),
        "n_true_negative_users": int((user_df["y_true"] == 0).sum()),
        "n_pred_positive_users": n_positive_users,
        "pred_positive_rate": float(class_ratio),
        "auc": safe_auc(y_true, user_df["best_window_score"]),
        "f1": metrics["f1"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
    }


def add_stability_conclusion(summary_df):
    main_mask = (
        (summary_df["min_high_risk"] == MAIN_RULE["min_high_risk"])
        & (summary_df["mean_score_threshold"] == MAIN_RULE["mean_score_threshold"])
    )
    main = summary_df.loc[main_mask].iloc[0]

    def conclude(row):
        rate_ok = abs(row["pred_positive_rate"] - main["pred_positive_rate"]) <= 0.05
        f1_ok = abs(row["f1"] - main["f1"]) <= 0.03
        recall_ok = abs(row["recall"] - main["recall"]) <= 0.05
        if rate_ok and f1_ok and recall_ok:
            return "stable"
        return "sensitive"

    summary_df["stable_vs_main_rule"] = summary_df.apply(conclude, axis=1)
    summary_df.loc[main_mask, "stable_vs_main_rule"] = "main_rule"
    return summary_df


def main():
    df = load_292_user_metadata(DST_ROOT)
    all_rows = []
    for min_high_risk in MIN_HIGH_RISK_GRID:
        for mean_score_threshold in MEAN_SCORE_THRESHOLD_GRID:
            all_rows.append(run_one_scheme(df, min_high_risk, mean_score_threshold))

    summary_df = add_stability_conclusion(pd.DataFrame(all_rows))
    summary_path = OUTPUT_DIR / "sensitivity_summary_292_users.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Loaded users: {df['username'].nunique()}")
    print(f"Output: {summary_path}")
    print(
        summary_df[
            [
                "scheme_name",
                "n_pred_positive_users",
                "pred_positive_rate",
                "auc",
                "f1",
                "recall",
                "stable_vs_main_rule",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
