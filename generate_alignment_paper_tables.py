"""Generate paper-ready tables for image-text alignment analyses.

Inputs are produced by analyze_image_text_alignment.py under:
results/final_preprocessing_v2/image_text_alignment_analysis/
"""

from pathlib import Path

import pandas as pd


RESULTS_ROOT = Path("results/final_preprocessing_v2")
ALIGNMENT_DIR = RESULTS_ROOT / "image_text_alignment_analysis"


def fmt(value, digits=4):
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def mean_std(row, metric):
    return f"{fmt(row[f'{metric}_mean'])} +/- {fmt(row[f'{metric}_std'])}"


def markdown_table(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_inputs():
    required = [
        RESULTS_ROOT / "method_and_modality_comparison.csv",
        ALIGNMENT_DIR / "post_level_clip_similarity.csv",
        ALIGNMENT_DIR / "user_level_alignment_delta.csv",
        ALIGNMENT_DIR / "alignment_group_user_performance.csv",
        ALIGNMENT_DIR / "alignment_thresholds.csv",
        ALIGNMENT_DIR / "case_review_samples.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required analysis files:\n" + "\n".join(missing)
        )
    return {
        "method": pd.read_csv(required[0]),
        "post": pd.read_csv(required[1]),
        "user": pd.read_csv(required[2]),
        "perf": pd.read_csv(required[3]),
        "thresholds": pd.read_csv(required[4]),
        "cases": pd.read_csv(required[5]),
    }


def build_modality_table(method):
    method_order = [
        ("text_clip", "Text-only CLIP", "caption text embedding", "文字語意基準"),
        ("image_clip", "Image-only CLIP", "image embedding", "影像弱標籤訊號"),
        (
            "concat",
            "Text+Image concat",
            "text + image concatenation",
            "簡單融合 baseline",
        ),
        (
            "contextvecnet",
            "Multimodal ContextVecNet (CLIP/CLIP)",
            "text + image + cross-attention + temporal module",
            "原始 CLIP/CLIP 多模態架構 baseline",
        ),
        (
            "text_bert",
            "Text-only BERT-base Chinese",
            "BERT caption embedding",
            "強文字 baseline",
        ),
        (
            "bert_clip",
            "BERT+Image CLIP",
            "BERT text + CLIP image",
            "主要多模態比較模型",
        ),
    ]
    rows = []
    for key, name, inputs, interpretation in method_order:
        sub = method[method["method"] == key]
        if sub.empty:
            rows.append(
                {
                    "模型": name,
                    "輸入": inputs,
                    "AUC": "尚未完成",
                    "F1": "尚未完成",
                    "Accuracy": "尚未完成",
                    "Precision": "尚未完成",
                    "Recall": "尚未完成",
                    "Interpretation": interpretation,
                }
            )
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "模型": name,
                "輸入": inputs,
                "AUC": mean_std(row, "auc"),
                "F1": mean_std(row, "f1"),
                "Accuracy": mean_std(row, "accuracy"),
                "Precision": mean_std(row, "precision"),
                "Recall": mean_std(row, "recall"),
                "Interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def build_similarity_stats(post):
    valid = post[post["valid_for_similarity"] & post["clip_similarity"].notna()].copy()

    def stat_row(name, frame):
        scores = frame["clip_similarity"].dropna()
        return {
            "群組": name,
            "N": int(scores.size),
            "Mean": fmt(scores.mean()),
            "SD": fmt(scores.std(ddof=1)),
            "Median": fmt(scores.median()),
            "Q1": fmt(scores.quantile(0.25)),
            "Q3": fmt(scores.quantile(0.75)),
        }

    rows = [
        stat_row("全部有效貼文", valid),
        stat_row("Positive users", valid[valid["label"] == 1]),
        stat_row("Negative users", valid[valid["label"] == 0]),
    ]
    for error_type in ["TP", "FP", "FN", "TN"]:
        rows.append(
            stat_row(
                error_type,
                valid[valid["multimodal_error_type"] == error_type],
            )
        )
    for group in ["High", "Medium", "Low"]:
        rows.append(
            stat_row(
                f"{group} alignment",
                valid[valid["alignment_group"] == group],
            )
        )
    return pd.DataFrame(rows)


def build_alignment_performance(perf):
    interpretations = {
        "High": (
            "圖文語意較一致；本資料中 Text-only BERT 表現較高，"
            "顯示文字語意已足夠強，圖片未必進一步提升。"
        ),
        "Medium": "圖文有部分關聯；BERT+Image CLIP 的 AUC/F1 略高，代表圖片可能提供有限輔助。",
        "Low": (
            "圖文語意較不一致；BERT+Image CLIP 的 recall/F1 略高但 precision 較低，"
            "顯示圖片可能同時帶來輔助與噪音。"
        ),
    }
    rows = []
    for group in ["High", "Medium", "Low"]:
        for model in ["Text-only BERT", "BERT+Image CLIP"]:
            sub = perf[(perf["user_alignment_group"] == group) & (perf["model"] == model)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append(
                {
                    "Alignment group": group,
                    "Model": model,
                    "N users": int(row["n_users"]),
                    "AUC": fmt(row["auc"]),
                    "F1": fmt(row["f1"]),
                    "Accuracy": fmt(row["accuracy"]),
                    "Precision": fmt(row["precision"]),
                    "Recall": fmt(row["recall"]),
                    "Interpretation": interpretations[group]
                    if model == "BERT+Image CLIP"
                    else "",
                }
            )
    return pd.DataFrame(rows)


def build_delta_stats(user):
    groups = [
        ("全部使用者", user),
        ("Positive users", user[user["label"] == 1]),
        ("Negative users", user[user["label"] == 0]),
        ("High alignment users", user[user["user_alignment_group"] == "High"]),
        ("Medium alignment users", user[user["user_alignment_group"] == "Medium"]),
        ("Low alignment users", user[user["user_alignment_group"] == "Low"]),
        ("TP", user[user["multimodal_error_type"] == "TP"]),
        ("FP", user[user["multimodal_error_type"] == "FP"]),
        ("FN", user[user["multimodal_error_type"] == "FN"]),
        ("TN", user[user["multimodal_error_type"] == "TN"]),
    ]
    rows = []
    for name, frame in groups:
        scores = frame["delta_p"].dropna()
        rows.append(
            {
                "群組": name,
                "N": int(scores.size),
                "Mean delta_p": fmt(scores.mean()),
                "SD": fmt(scores.std(ddof=1)),
                "Median": fmt(scores.median()),
                "Min": fmt(scores.min()),
                "Max": fmt(scores.max()),
                "|delta_p| >= 0.10": int((scores.abs() >= 0.10).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_top_delta_cases(user):
    top = user.assign(abs_delta_p=user["delta_p"].abs()).sort_values(
        "abs_delta_p", ascending=False
    )
    top = top[
        [
            "fold",
            "author",
            "label",
            "text_probability",
            "multimodal_probability",
            "delta_p",
            "user_alignment_group",
            "multimodal_error_type",
        ]
    ].head(20)
    for col in ["text_probability", "multimodal_probability", "delta_p"]:
        top[col] = top[col].map(fmt)
    return top


def build_case_summary(cases):
    summary = cases["case_type"].value_counts().rename_axis("案例類型").reset_index(
        name="案例數"
    )
    summary["建議用途"] = summary["案例類型"].map(
        {
            "high_alignment_correct": "圖文一致且模型正確，用於說明圖片與文字一致時的代表案例。",
            "high_alignment_error": "圖文一致但模型錯誤，用於檢查即使一致仍誤判的限制。",
            "low_alignment_correct": "圖文不一致但模型正確，用於檢查模型是否主要依賴文字訊號。",
            "low_alignment_error": "圖文不一致且模型錯誤，用於分析影像噪音或弱標籤限制。",
            "largest_abs_delta_p": "加入圖片後風險分數變化最大，用於分析圖片影響方向。",
            "multimodal_false_positive": "多模態 FP，用於分析圖片是否推高負類風險。",
            "multimodal_false_negative": "多模態 FN，用於分析圖片是否稀釋正類文字訊號。",
        }
    ).fillna("人工檢視樣本")
    return summary


def build_taxonomy_template():
    return pd.DataFrame(
        [
            ["A", "Caption 高風險、圖片中性", "文字有憂鬱語意，但圖片是風景、食物、日常物品", "圖片可能稀釋文字訊號"],
            ["B", "Caption 中性、圖片低落氛圍", "文字無明顯風險，但圖片暗色、孤獨、空景", "圖片可能提高風險分數"],
            ["C", "圖片為裝飾或無關內容", "貼圖、廣告、品牌照、與 caption 主題無關", "影像模態可能成為噪音"],
            ["D", "圖片含文字截圖", "圖片本身包含文字訊息，如限動、對話截圖", "CLIP image encoder 未必能完整理解，可考慮 OCR"],
            ["E", "圖文反諷或語氣不一致", "caption 有幽默、反諷或情緒轉折", "模型容易誤解真實語意"],
            ["F", "多圖或影片代表影格不足", "第一張圖片或中間影格不能代表整篇貼文", "影像特徵不完整"],
            ["G", "缺圖或低品質圖片", "黑圖、模糊、過暗、截斷", "影像訊號可靠度低"],
            ["H", "平台風格影響", "濾鏡、自拍、打卡、品牌風格", "模型可能學到非心理訊號"],
        ],
        columns=["類型", "說明", "常見情境", "可能影響"],
    )


def main():
    data = load_inputs()
    tables = {
        "paper_table_1_modality_method_comparison.csv": build_modality_table(
            data["method"]
        ),
        "paper_table_2_clip_similarity_statistics.csv": build_similarity_stats(
            data["post"]
        ),
        "paper_table_3_alignment_group_performance.csv": build_alignment_performance(
            data["perf"]
        ),
        "paper_table_4_delta_p_statistics.csv": build_delta_stats(data["user"]),
        "paper_table_5_top_delta_p_cases.csv": build_top_delta_cases(data["user"]),
        "paper_table_6_case_review_sampling_summary.csv": build_case_summary(
            data["cases"]
        ),
        "paper_table_7_inconsistency_taxonomy_template.csv": build_taxonomy_template(),
    }
    for filename, frame in tables.items():
        frame.to_csv(ALIGNMENT_DIR / filename, index=False)

    thresholds = data["thresholds"].iloc[0]
    lines = [
        "# 多模態圖片弱標籤補強分析統計表格",
        "",
        f"- 有效貼文數：{int(thresholds['valid_post_count']):,}",
        f"- 使用者數：{int(thresholds['user_count']):,}",
        f"- Low alignment：similarity <= {thresholds['post_similarity_q25']:.4f}",
        f"- High alignment：similarity >= {thresholds['post_similarity_q75']:.4f}",
        "",
    ]
    titles = [
        ("表 1. Text-only / Image-only / Multimodal 比較", tables["paper_table_1_modality_method_comparison.csv"]),
        ("表 2. CLIP image-text similarity 統計", tables["paper_table_2_clip_similarity_statistics.csv"]),
        ("表 3. Alignment group 下的使用者層級模型表現", tables["paper_table_3_alignment_group_performance.csv"]),
        ("表 4. 圖片影響分數 delta_p 統計", tables["paper_table_4_delta_p_statistics.csv"]),
        ("表 5. |delta_p| 最大案例 Top 20", tables["paper_table_5_top_delta_p_cases.csv"]),
        ("表 6. 圖文不一致案例抽樣統計", tables["paper_table_6_case_review_sampling_summary.csv"]),
        ("表 7. 圖文不一致 taxonomy 人工檢視模板", tables["paper_table_7_inconsistency_taxonomy_template.csv"]),
    ]
    for title, frame in titles:
        lines.extend([f"## {title}", "", markdown_table(frame), ""])
    lines.extend(
        [
            "## 可搭配圖表",
            "",
            "- `clip_similarity_histogram.png`",
            "- `similarity_by_error_type_boxplot.png`",
            "- `alignment_group_auc_bar.png`",
            "",
        ]
    )
    report_path = ALIGNMENT_DIR / "paper_ready_statistical_tables.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    for filename in tables:
        print(f"Wrote {ALIGNMENT_DIR / filename}")


if __name__ == "__main__":
    main()
