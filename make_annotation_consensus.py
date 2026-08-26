#!/usr/bin/env python3
"""Prepare and finalize consensus annotations for minimal image-text case review."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE = Path("results/final_preprocessing_v2/image_text_alignment_analysis")
DEFAULT_RATER1 = BASE / "case_review_samples_minimal_annotated.csv"
DEFAULT_RATER2 = BASE / "case_review_samples_minimal_annotated_rater2.csv"
DEFAULT_DRAFT = BASE / "case_review_samples_minimal_consensus_draft.csv"
DEFAULT_REVIEW = BASE / "case_review_samples_minimal_consensus_review.csv"
DEFAULT_DISAGREE = BASE / "case_review_samples_minimal_consensus_disagreements.csv"
DEFAULT_REVIEW_ANNOTATED = BASE / "case_review_samples_minimal_consensus_review_annotated.csv"
DEFAULT_FINAL = BASE / "case_review_samples_minimal_consensus_final.csv"

KEY_COLS = ["fold", "author"]
CONSENSUS_FIELDS = [
    "manual_text_image_consistency",
    "manual_image_extra_signal",
    "manual_image_effect",
]
OPTIONAL_FIELDS = [
    "manual_inconsistency_taxonomy_A_to_H",
    "manual_identifiable_info",
    "manual_notes",
]
CONTEXT_COLS = [
    "case_type",
    "minimal_review_reason",
    "label",
    "text_probability",
    "multimodal_probability",
    "delta_p",
    "mean_similarity",
    "user_alignment_group",
    "multimodal_error_type",
    "lowest_similarity_caption",
    "highest_similarity_caption",
    "lowest_similarity_image_path",
    "highest_similarity_image_path",
]


def clean_manual_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CONSENSUS_FIELDS + OPTIONAL_FIELDS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def read_annotated(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {name}: {path}")
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLS + CONSENSUS_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    if df.duplicated(KEY_COLS).any():
        dup = df.loc[df.duplicated(KEY_COLS, keep=False), KEY_COLS]
        raise ValueError(f"{name} has duplicate fold/author keys:\n{dup}")
    return clean_manual_columns(df)


def prepare(args: argparse.Namespace) -> None:
    r1 = read_annotated(args.rater1, "rater1")
    r2 = read_annotated(args.rater2, "rater2")

    draft = r1.copy()
    r2_fields = r2[KEY_COLS + CONSENSUS_FIELDS].rename(
        columns={field: f"{field}_rater2" for field in CONSENSUS_FIELDS}
    )
    draft = draft.merge(r2_fields, on=KEY_COLS, how="left", validate="one_to_one")

    disagreement_rows = []
    needs_review = pd.Series(False, index=draft.index)
    disagreement_field_values = []

    for idx, row in draft.iterrows():
        disagreed_fields = []
        for field in CONSENSUS_FIELDS:
            r1_value = str(row.get(field, "")).strip()
            r2_value = str(row.get(f"{field}_rater2", "")).strip()
            draft.at[idx, f"{field}_rater1"] = r1_value
            if r1_value == r2_value:
                draft.at[idx, field] = r1_value
            else:
                draft.at[idx, field] = ""
                needs_review.at[idx] = True
                disagreed_fields.append(field)
                disagreement_rows.append(
                    {
                        **{k: row.get(k, "") for k in KEY_COLS},
                        "case_type": row.get("case_type", row.get("minimal_review_reason", "")),
                        "label": row.get("label", ""),
                        "delta_p": row.get("delta_p", ""),
                        "field": field,
                        "rater1": r1_value,
                        "rater2": r2_value,
                    }
                )
        disagreement_field_values.append(";".join(disagreed_fields))

    draft["needs_consensus_review"] = needs_review
    draft["disagreement_fields"] = disagreement_field_values

    review = draft.loc[needs_review].copy()
    keep_cols = []
    for col in KEY_COLS + CONTEXT_COLS + CONSENSUS_FIELDS + OPTIONAL_FIELDS:
        if col in review.columns and col not in keep_cols:
            keep_cols.append(col)
    for field in CONSENSUS_FIELDS:
        for suffix in ["_rater1", "_rater2"]:
            col = f"{field}{suffix}"
            if col in review.columns:
                keep_cols.append(col)
    for col in ["needs_consensus_review", "disagreement_fields"]:
        if col in review.columns:
            keep_cols.append(col)
    review = review[keep_cols]

    disagreement_df = pd.DataFrame(disagreement_rows)
    args.draft.parent.mkdir(parents=True, exist_ok=True)
    draft.to_csv(args.draft, index=False)
    review.to_csv(args.review, index=False)
    disagreement_df.to_csv(args.disagreements, index=False)

    print(f"saved draft: {args.draft} rows={len(draft)}")
    print(f"saved review subset: {args.review} rows={len(review)}")
    print(f"saved disagreements: {args.disagreements} rows={len(disagreement_df)}")
    if len(review):
        print("review fold/author rows:")
        print(review[KEY_COLS + ["disagreement_fields"]].to_string(index=False))


def finalize(args: argparse.Namespace) -> None:
    draft = pd.read_csv(args.draft)
    review = pd.read_csv(args.review_output)
    draft = clean_manual_columns(draft)
    review = clean_manual_columns(review)

    if review.duplicated(KEY_COLS).any():
        raise ValueError("review output has duplicate fold/author keys")

    review_indexed = review.set_index(KEY_COLS)
    final = draft.copy()
    final_index = final.set_index(KEY_COLS).index

    for idx, row in final.iterrows():
        key = tuple(row[k] for k in KEY_COLS)
        if key not in review_indexed.index:
            continue
        review_row = review_indexed.loc[key]
        for field in CONSENSUS_FIELDS:
            value = str(review_row.get(field, "")).strip()
            if value:
                final.at[idx, field] = value

    missing = {}
    for field in CONSENSUS_FIELDS:
        blank = final[field].fillna("").astype(str).str.strip() == ""
        if blank.any():
            missing[field] = int(blank.sum())
    if missing:
        raise ValueError(f"Consensus still has blank required fields: {missing}")

    # Keep provenance columns, but mark the file as finalized.
    final["consensus_status"] = "final"
    final.to_csv(args.final, index=False)
    print(f"saved final consensus: {args.final} rows={len(final)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="create consensus draft and disagreement review files")
    p_prepare.add_argument("--rater1", type=Path, default=DEFAULT_RATER1)
    p_prepare.add_argument("--rater2", type=Path, default=DEFAULT_RATER2)
    p_prepare.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    p_prepare.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    p_prepare.add_argument("--disagreements", type=Path, default=DEFAULT_DISAGREE)
    p_prepare.set_defaults(func=prepare)

    p_finalize = sub.add_parser("finalize", help="merge reviewed disagreement rows back into final consensus file")
    p_finalize.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    p_finalize.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_ANNOTATED)
    p_finalize.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    p_finalize.set_defaults(func=finalize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
