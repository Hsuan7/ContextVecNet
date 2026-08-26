"""Generate a minimal manual-review subset for image-text case analysis.

The full case_review_samples.csv contains multiple case-type samples. This script
creates a smaller "minimum viable" review file:
- multimodal false positives: 10
- multimodal false negatives: 10
- low similarity users: 20
- largest absolute delta_p users: 20

Rows are de-duplicated by fold + author. A row can keep multiple review reasons.
"""

from pathlib import Path

import pandas as pd


BASE = Path("results/final_preprocessing_v2/image_text_alignment_analysis")
USER_PATH = BASE / "user_level_alignment_delta.csv"
POST_PATH = BASE / "post_level_clip_similarity.csv"
OUTPUT_PATH = BASE / "case_review_samples_minimal.csv"

MANUAL_COLUMNS = [
    "manual_text_image_consistency",
    "manual_image_extra_signal",
    "manual_image_effect",
    "manual_inconsistency_taxonomy_A_to_H",
    "manual_identifiable_info",
    "manual_notes",
]


def add_reason(existing, reason):
    if not existing:
        return reason
    reasons = existing.split(";")
    if reason not in reasons:
        reasons.append(reason)
    return ";".join(reasons)


def select_rows(user_df):
    selections = [
        (
            "multimodal_false_positive_top10",
            user_df[user_df["multimodal_error_type"] == "FP"].sort_values(
                "multimodal_probability", ascending=False
            ).head(10),
        ),
        (
            "multimodal_false_negative_top10",
            user_df[user_df["multimodal_error_type"] == "FN"].sort_values(
                "multimodal_probability", ascending=True
            ).head(10),
        ),
        (
            "low_similarity_top20",
            user_df[user_df["user_alignment_group"] == "Low"].sort_values(
                "mean_similarity", ascending=True
            ).head(20),
        ),
        (
            "largest_abs_delta_p_top20",
            user_df.assign(abs_delta_p=user_df["delta_p"].abs()).sort_values(
                "abs_delta_p", ascending=False
            ).head(20),
        ),
    ]

    chosen = {}
    for reason, frame in selections:
        for _, row in frame.iterrows():
            key = (int(row["fold"]), str(row["author"]))
            row_dict = row.to_dict()
            if key not in chosen:
                row_dict["minimal_review_reason"] = reason
                chosen[key] = row_dict
            else:
                chosen[key]["minimal_review_reason"] = add_reason(
                    chosen[key].get("minimal_review_reason", ""), reason
                )
    return pd.DataFrame(chosen.values())


def add_representative_posts(cases, post_df):
    valid_posts = post_df[post_df["valid_for_similarity"]].copy()
    low_post = (
        valid_posts.sort_values("clip_similarity")
        .groupby(["fold", "author"], as_index=False)
        .head(1)
        .rename(
            columns={
                "caption": "lowest_similarity_caption",
                "image_path": "lowest_similarity_image_path",
                "clip_similarity": "lowest_post_similarity",
            }
        )
    )
    high_post = (
        valid_posts.sort_values("clip_similarity", ascending=False)
        .groupby(["fold", "author"], as_index=False)
        .head(1)
        .rename(
            columns={
                "caption": "highest_similarity_caption",
                "image_path": "highest_similarity_image_path",
                "clip_similarity": "highest_post_similarity",
            }
        )
    )
    cases = cases.merge(
        low_post[
            [
                "fold",
                "author",
                "lowest_similarity_caption",
                "lowest_similarity_image_path",
                "lowest_post_similarity",
            ]
        ],
        on=["fold", "author"],
        how="left",
    )
    cases = cases.merge(
        high_post[
            [
                "fold",
                "author",
                "highest_similarity_caption",
                "highest_similarity_image_path",
                "highest_post_similarity",
            ]
        ],
        on=["fold", "author"],
        how="left",
    )
    return cases


def main():
    user_df = pd.read_csv(USER_PATH)
    post_df = pd.read_csv(POST_PATH)
    cases = select_rows(user_df)
    cases = add_representative_posts(cases, post_df)
    for col in MANUAL_COLUMNS:
        if col not in cases.columns:
            cases[col] = ""
    first_cols = [
        "minimal_review_reason",
        "fold",
        "author",
        "label",
        "multimodal_probability",
        "text_probability",
        "delta_p",
        "multimodal_error_type",
        "user_alignment_group",
        "mean_similarity",
    ]
    remaining = [col for col in cases.columns if col not in first_cols]
    cases = cases[first_cols + remaining]
    cases.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"rows={len(cases)}")
    print(cases["minimal_review_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
