from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = r"D:\時間序列\DECEN_TS\auto_labeled_with_scores_combined.csv"
OUTPUT_DIR = Path(r"D:\時間序列\labeled\label_sensitivity_from_input_csv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2021-01-01 00:00:00+00:00"
END_DATE = "2026-06-01 23:59:59+00:00"
WINDOW_DAYS = 14

MIN_POSTS_IN_WINDOW = 3
HIGH_RISK_SCORE_THRESHOLD = 0.7
USE_PSEUDO_LABEL_IF_AVAILABLE = True

MIN_HIGH_RISK_GRID = [1, 2, 3]
MEAN_SCORE_THRESHOLD_GRID = [0.5, 0.6, 0.7]

MAIN_RULE = {
    "min_posts": 3,
    "min_high_risk": 2,
    "mean_score_threshold": 0.6,
}

STABLE_MAX_RATE_DIFF = 0.05
STABLE_MIN_JACCARD = 0.90


def scheme_name(min_high_risk, mean_score_threshold):
    mean_part = str(mean_score_threshold).replace(".", "p")
    return f"high{min_high_risk}_mean{mean_part}"


def load_input_csv(input_csv):
    usecols = [
        "username",
        "post_id",
        "taken_at",
        "p_depression_tcal",
        "pseudo_label",
    ]
    df = pd.read_csv(input_csv, usecols=lambda c: c in usecols)
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["username", "post_id", "taken_at", "p_depression_tcal"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["username"] = df["username"].astype(str).str.strip()
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
    times = g["taken_at"].tolist()
    scores = g["p_depression_tcal"].to_numpy(dtype=float)
    high_flags = g["is_high_risk"].to_numpy(dtype=int)
    score_cumsum = np.concatenate([[0.0], np.cumsum(scores)])
    high_cumsum = np.concatenate([[0], np.cumsum(high_flags)])

    positive_windows = []
    best_window_mean = np.nan
    best_window_high_risk = 0
    best_window_n_posts = 0
    best_window_rule_ratio = 0.0
    j = 0

    for i in range(len(g)):
        window_start = times[i]
        window_end = window_start + pd.Timedelta(days=WINDOW_DAYS)
        while j < len(g) and times[j] < window_end:
            j += 1

        n_posts = j - i
        n_high = int(high_cumsum[j] - high_cumsum[i])
        mean_score = float((score_cumsum[j] - score_cumsum[i]) / n_posts) if n_posts else np.nan

        post_ratio = n_posts / MIN_POSTS_IN_WINDOW if MIN_POSTS_IN_WINDOW else 0.0
        high_ratio = n_high / min_high_risk if min_high_risk else 0.0
        mean_ratio = mean_score / mean_score_threshold if mean_score_threshold else 0.0
        rule_ratio = min(post_ratio, high_ratio, mean_ratio)
        if rule_ratio > best_window_rule_ratio:
            best_window_rule_ratio = float(rule_ratio)
            best_window_mean = mean_score
            best_window_high_risk = n_high
            best_window_n_posts = n_posts

        if (
            n_posts >= MIN_POSTS_IN_WINDOW
            and n_high >= min_high_risk
            and mean_score >= mean_score_threshold
        ):
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

    user_label = int(len(positive_windows) > 0)
    user_summary = {
        "username": g.loc[0, "username"],
        "num_posts_total": int(len(g)),
        "num_high_risk_posts_total": int(g["is_high_risk"].sum()),
        "mean_p_depression_tcal_total": float(g["p_depression_tcal"].mean()),
        "user_label": user_label,
        "num_positive_windows": int(len(positive_windows)),
        "best_window_rule_ratio": best_window_rule_ratio,
        "best_window_n_posts": int(best_window_n_posts),
        "best_window_high_risk": int(best_window_high_risk),
        "best_window_mean": best_window_mean,
    }
    return user_summary, positive_windows


def run_one_scheme(df, min_high_risk, mean_score_threshold):
    name = scheme_name(min_high_risk, mean_score_threshold)
    user_rows = []
    window_rows = []

    for _, g in df.groupby("username", sort=True):
        user_summary, positive_windows = label_single_user_by_window(
            g,
            min_high_risk=min_high_risk,
            mean_score_threshold=mean_score_threshold,
        )
        user_summary["scheme_name"] = name
        user_summary["window_days"] = WINDOW_DAYS
        user_summary["min_posts"] = MIN_POSTS_IN_WINDOW
        user_summary["min_high_risk"] = min_high_risk
        user_summary["mean_score_threshold"] = mean_score_threshold
        user_rows.append(user_summary)

        for row in positive_windows:
            row["scheme_name"] = name
            row["window_days"] = WINDOW_DAYS
            row["min_posts"] = MIN_POSTS_IN_WINDOW
            row["min_high_risk"] = min_high_risk
            row["mean_score_threshold"] = mean_score_threshold
            window_rows.append(row)

    user_df = pd.DataFrame(user_rows)
    windows_df = pd.DataFrame(window_rows)
    user_df.to_csv(OUTPUT_DIR / f"user_summary_{name}.csv", index=False, encoding="utf-8-sig")
    windows_df.to_csv(OUTPUT_DIR / f"positive_windows_{name}.csv", index=False, encoding="utf-8-sig")
    return user_df


def add_overlap_stability(summary_df, user_labels_by_scheme):
    main_name = scheme_name(MAIN_RULE["min_high_risk"], MAIN_RULE["mean_score_threshold"])
    if main_name not in user_labels_by_scheme:
        raise ValueError(f"Main rule {main_name} was not found in parameter grid.")

    main_positive_users = user_labels_by_scheme[main_name]
    main_rate = float(summary_df.loc[summary_df["scheme_name"] == main_name, "positive_rate"].iloc[0])

    overlap_rows = []
    for _, row in summary_df.iterrows():
        name = row["scheme_name"]
        positive_users = user_labels_by_scheme[name]
        overlap = positive_users & main_positive_users
        added = positive_users - main_positive_users
        removed = main_positive_users - positive_users
        union = positive_users | main_positive_users
        jaccard = len(overlap) / len(union) if union else 1.0
        rate_diff = abs(float(row["positive_rate"]) - main_rate)

        if name == main_name:
            conclusion = "main_rule"
        elif rate_diff <= STABLE_MAX_RATE_DIFF and jaccard >= STABLE_MIN_JACCARD:
            conclusion = "stable"
        else:
            conclusion = "sensitive"

        overlap_rows.append(
            {
                "scheme_name": name,
                "overlap_with_main": int(len(overlap)),
                "added_vs_main": int(len(added)),
                "removed_vs_main": int(len(removed)),
                "jaccard_vs_main": float(jaccard),
                "positive_rate_diff_vs_main": float(rate_diff),
                "stability_conclusion": conclusion,
            }
        )

    overlap_df = pd.DataFrame(overlap_rows)
    return summary_df.merge(overlap_df, on="scheme_name", how="left")


def main():
    df = load_input_csv(INPUT_CSV)
    all_summary_rows = []
    user_labels_by_scheme = {}

    for min_high_risk in MIN_HIGH_RISK_GRID:
        for mean_score_threshold in MEAN_SCORE_THRESHOLD_GRID:
            user_df = run_one_scheme(df, min_high_risk, mean_score_threshold)
            name = scheme_name(min_high_risk, mean_score_threshold)
            positive_users = set(user_df.loc[user_df["user_label"] == 1, "username"])
            user_labels_by_scheme[name] = positive_users

            n_users = int(user_df["username"].nunique())
            n_positive_users = int(user_df["user_label"].sum())
            all_summary_rows.append(
                {
                    "scheme_name": name,
                    "window_days": WINDOW_DAYS,
                    "min_posts": MIN_POSTS_IN_WINDOW,
                    "min_high_risk": min_high_risk,
                    "mean_score_threshold": mean_score_threshold,
                    "n_users": n_users,
                    "n_positive_users": n_positive_users,
                    "n_negative_users": int(n_users - n_positive_users),
                    "positive_rate": float(n_positive_users / n_users) if n_users else np.nan,
                    "avg_positive_windows_per_positive_user": float(
                        user_df.loc[user_df["user_label"] == 1, "num_positive_windows"].mean()
                    )
                    if n_positive_users
                    else 0.0,
                }
            )

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df = add_overlap_stability(summary_df, user_labels_by_scheme)

    summary_path = OUTPUT_DIR / "sensitivity_summary_distribution_stability.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Loaded posts: {len(df)}")
    print(f"Loaded users: {df['username'].nunique()}")
    print(f"Output: {summary_path}")
    print(
        summary_df[
            [
                "scheme_name",
                "n_positive_users",
                "positive_rate",
                "overlap_with_main",
                "added_vs_main",
                "removed_vs_main",
                "jaccard_vs_main",
                "stability_conclusion",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
