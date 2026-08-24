import ast
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# 0. 路徑與參數設定
# =========================================================
INPUT_POST_CSV = Path(
    r"D:\時間序列\DECEN_TS\auto_labeled_with_scores_combined.csv")
INPUT_POS_WINDOWS_CSV = Path(r"D:\時間序列\labeled\positive_windows_14d.csv")

OUTPUT_DIR = Path(r"D:\時間序列\window_level_plus_prev_context_32")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_WINDOW_DATASET = OUTPUT_DIR / "window_dataset.csv"
OUTPUT_USER_SPLIT = OUTPUT_DIR / "window_user_split.csv"

WINDOW_DAYS = 14
MODEL_WINDOW_SIZE = 32

# 正視窗條件（應與你原標註一致）
MIN_POSTS_IN_WINDOW = 3
MIN_HIGH_RISK_POSTS = 2
MEAN_SCORE_THRESHOLD = 0.6
HIGH_RISK_SCORE_THRESHOLD = 0.7
USE_PSEUDO_LABEL_IF_AVAILABLE = True

# 負視窗條件（第一版用較乾淨的負樣本）
NEG_MIN_POSTS = 3
NEG_MAX_HIGH_RISK_POSTS = 0
NEG_MAX_MEAN_SCORE = 0.4

# 每位使用者最多保留幾個視窗
MAX_POS_WINDOWS_PER_USER = 3
MAX_NEG_WINDOWS_PER_USER = 3

# 視窗去重
JACCARD_THRESHOLD = 0.8

# split 比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# 1. 基本工具函式
# =========================================================
def parse_post_ids(value):
    """
    將字串形式的 list 轉成真正的 Python list
    例如 "['a', 'b']" -> ['a', 'b']
    """
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if not s:
        return []
    try:
        return list(ast.literal_eval(s))
    except Exception:
        return []


def jaccard_similarity(list_a, list_b):
    set_a = set(list_a)
    set_b = set(list_b)
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = set_a | set_b
    inter = set_a & set_b
    if len(union) == 0:
        return 0.0
    return len(inter) / len(union)


def is_high_risk_post(row,
                      use_pseudo_label_if_available=True,
                      high_risk_score_threshold=0.7):
    if use_pseudo_label_if_available and "pseudo_label" in row.index and pd.notna(row["pseudo_label"]):
        return int(row["pseudo_label"] == 1)
    return int(pd.notna(row["p_depression_tcal"]) and row["p_depression_tcal"] >= high_risk_score_threshold)


# def select_posts_for_model(window_df, model_window_size=16):
#     """
#     第一版規則：只取 14 天視窗內的貼文，
#     若超過 window_size，就取時間排序後最後 K 篇。
#     """
#     g = window_df.sort_values(["taken_at", "post_id"]).copy()
#     selected_ids = g["post_id"].astype(str).tolist()

#     if len(selected_ids) > model_window_size:
#         selected_ids = selected_ids[-model_window_size:]

#     selected_post_count = len(selected_ids)
#     padding_needed = max(0, model_window_size - selected_post_count)

#     return selected_ids, selected_post_count, padding_needed
def select_posts_for_model_with_prev_context(
    df_all_user_posts,
    username,
    window_start,
    window_end,
    anchor_post_ids,
    model_window_size=16,
):
    """
    規則：
    1. 先取 14 天視窗內的 anchor posts
    2. 若 anchor posts 數量 >= window_size，取最後 K 篇
    3. 若不足，往 window_start 之前補最近貼文，直到滿足 window_size
    4. 若仍不足，後續交給模型 padding

    注意：
    - 標籤仍由原本 14 天視窗決定
    - 補進來的貼文只作為 context，不參與重新標記
    """

    # 該 user 全部貼文
    g_user = df_all_user_posts[df_all_user_posts["username"]
                               == username].copy()
    g_user["post_id"] = g_user["post_id"].astype(str)
    g_user = g_user.sort_values(["taken_at", "post_id"]).reset_index(drop=True)

    anchor_post_ids = [str(x) for x in anchor_post_ids]

    # 1) 先取 14 天視窗內貼文
    anchor_df = g_user[g_user["post_id"].isin(anchor_post_ids)].copy()
    anchor_df = anchor_df.sort_values(
        ["taken_at", "post_id"]).reset_index(drop=True)

    selected_ids = anchor_df["post_id"].astype(str).tolist()

    # 2) 若已經超過 window_size，就只取最後 K 篇
    if len(selected_ids) >= model_window_size:
        selected_ids = selected_ids[-model_window_size:]
        selected_post_count = len(selected_ids)
        padding_needed = 0
        return selected_ids, selected_post_count, padding_needed

    # 3) 不足時，往前補 window_start 之前最近貼文
    prev_df = g_user[g_user["taken_at"] < window_start].copy()
    prev_df = prev_df.sort_values(
        ["taken_at", "post_id"]).reset_index(drop=True)

    prev_ids = prev_df["post_id"].astype(str).tolist()

    # 從最近的往前補，所以取最後幾篇
    need = model_window_size - len(selected_ids)
    prev_context_ids = prev_ids[-need:] if need > 0 else []

    # 最後序列順序：前面是較早的補充 context，後面是 anchor posts
    selected_ids = prev_context_ids + selected_ids

    # 若補完後還是超過（理論上不會，但保險）
    if len(selected_ids) > model_window_size:
        selected_ids = selected_ids[-model_window_size:]

    selected_post_count = len(selected_ids)
    padding_needed = max(0, model_window_size - selected_post_count)

    return selected_ids, selected_post_count, padding_needed


# =========================================================
# 2. 讀取 post-level 資料
# =========================================================
df = pd.read_csv(INPUT_POST_CSV)
df.columns = [c.strip() for c in df.columns]

required_cols = ["username", "post_id", "taken_at", "p_depression_tcal"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"缺少必要欄位: {col}")

df["taken_at"] = pd.to_datetime(df["taken_at"], utc=True, errors="coerce")
df["p_depression_tcal"] = pd.to_numeric(
    df["p_depression_tcal"], errors="coerce")

if "pseudo_label" in df.columns:
    df["pseudo_label"] = pd.to_numeric(df["pseudo_label"], errors="coerce")

df = df.dropna(subset=["username", "post_id",
               "taken_at", "p_depression_tcal"]).copy()
df = df.drop_duplicates(subset=["username", "post_id"]).copy()
df = df.sort_values(["username", "taken_at", "post_id"]).reset_index(drop=True)

df["is_high_risk"] = df.apply(
    lambda r: is_high_risk_post(
        r,
        use_pseudo_label_if_available=USE_PSEUDO_LABEL_IF_AVAILABLE,
        high_risk_score_threshold=HIGH_RISK_SCORE_THRESHOLD,
    ),
    axis=1
)


# =========================================================
# 3. 讀取正視窗（若你想直接沿用已產生好的 positive_windows_14d.csv）
# =========================================================
pos_win_df = pd.read_csv(INPUT_POS_WINDOWS_CSV)
pos_win_df.columns = [c.strip() for c in pos_win_df.columns]

required_pos_cols = [
    "username", "window_start", "window_end",
    "n_posts", "n_high_risk_posts", "mean_p_depression_tcal", "post_ids"
]
for col in required_pos_cols:
    if col not in pos_win_df.columns:
        raise ValueError(f"positive_windows_14d.csv 缺少必要欄位: {col}")

pos_win_df["window_start"] = pd.to_datetime(
    pos_win_df["window_start"], utc=True, errors="coerce")
pos_win_df["window_end"] = pd.to_datetime(
    pos_win_df["window_end"], utc=True, errors="coerce")
pos_win_df["anchor_post_ids"] = pos_win_df["post_ids"].apply(parse_post_ids)


# =========================================================
# 4. 正視窗去重
# =========================================================
def deduplicate_windows_for_user(df_user_windows, label_type="positive"):
    """
    對同一 user 的 windows 進行去重：
    若 anchor_post_ids 的 Jaccard >= threshold，則視為重複，只保留一個
    """
    if len(df_user_windows) == 0:
        return df_user_windows.copy()

    g = df_user_windows.sort_values(
        ["window_start", "window_end"]).copy().reset_index(drop=True)
    kept_rows = []

    for _, row in g.iterrows():
        current_ids = row["anchor_post_ids"]

        duplicate_idx = None
        for k_idx, kept in enumerate(kept_rows):
            score = jaccard_similarity(current_ids, kept["anchor_post_ids"])
            if score >= JACCARD_THRESHOLD:
                duplicate_idx = k_idx
                break

        if duplicate_idx is None:
            kept_rows.append(row.to_dict())
        else:
            old = kept_rows[duplicate_idx]

            # 選擇保留哪一個
            keep_new = False
            if label_type == "positive":
                # 優先：高風險貼文數高 -> 平均分數高 -> 起始時間早
                if row["n_high_risk_posts"] > old["n_high_risk_posts"]:
                    keep_new = True
                elif row["n_high_risk_posts"] == old["n_high_risk_posts"]:
                    if row["mean_p_depression_tcal"] > old["mean_p_depression_tcal"]:
                        keep_new = True
                    elif row["mean_p_depression_tcal"] == old["mean_p_depression_tcal"]:
                        if row["window_start"] < old["window_start"]:
                            keep_new = True
            else:
                # negative: 優先平均分數低 -> 高風險貼文少 -> 起始時間早
                if row["mean_p_depression_tcal"] < old["mean_p_depression_tcal"]:
                    keep_new = True
                elif row["mean_p_depression_tcal"] == old["mean_p_depression_tcal"]:
                    if row["n_high_risk_posts"] < old["n_high_risk_posts"]:
                        keep_new = True
                    elif row["n_high_risk_posts"] == old["n_high_risk_posts"]:
                        if row["window_start"] < old["window_start"]:
                            keep_new = True

            if keep_new:
                kept_rows[duplicate_idx] = row.to_dict()

    return pd.DataFrame(kept_rows)


dedup_pos_list = []
for username, g in pos_win_df.groupby("username"):
    g = g.rename(columns={
        "n_posts": "n_posts_in_14d"
    }).copy()
    dedup_g = deduplicate_windows_for_user(g, label_type="positive")
    dedup_g["window_label"] = 1
    dedup_g["window_type"] = "positive_14d"
    dedup_pos_list.append(dedup_g)

dedup_pos_df = pd.concat(
    dedup_pos_list, ignore_index=True) if dedup_pos_list else pd.DataFrame()


# =========================================================
# 5. 建立負視窗候選
# =========================================================
def generate_negative_windows(df_user):
    """
    對單一 user 產生負視窗候選：
    - 使用同樣 14 天滑動視窗
    - 但只保留乾淨負樣本：
        n_posts >= NEG_MIN_POSTS
        n_high_risk_posts <= NEG_MAX_HIGH_RISK_POSTS
        mean_score < NEG_MAX_MEAN_SCORE
    """
    g = df_user.sort_values("taken_at").copy().reset_index(drop=True)
    rows = []

    n = len(g)
    for i in range(n):
        window_start = g.loc[i, "taken_at"]
        window_end = window_start + pd.Timedelta(days=WINDOW_DAYS)

        window_df = g[(g["taken_at"] >= window_start) &
                      (g["taken_at"] < window_end)].copy()

        n_posts = len(window_df)
        n_high = int(window_df["is_high_risk"].sum())
        mean_score = float(
            window_df["p_depression_tcal"].mean()) if n_posts > 0 else np.nan

        is_negative_window = (
            (n_posts >= NEG_MIN_POSTS) and
            (n_high <= NEG_MAX_HIGH_RISK_POSTS) and
            (mean_score < NEG_MAX_MEAN_SCORE)
        )

        if is_negative_window:
            rows.append({
                "username": g.loc[0, "username"],
                "window_start": window_start,
                "window_end": window_end,
                "n_posts_in_14d": int(n_posts),
                "n_high_risk_posts": int(n_high),
                "mean_p_depression_tcal": float(mean_score),
                "anchor_post_ids": list(window_df["post_id"].astype(str)),
                "window_label": 0,
                "window_type": "negative_14d",
            })

    return pd.DataFrame(rows)


neg_list = []
for username, g in df.groupby("username"):
    neg_g = generate_negative_windows(g)
    if len(neg_g) > 0:
        dedup_neg = deduplicate_windows_for_user(neg_g, label_type="negative")
        neg_list.append(dedup_neg)

neg_df = pd.concat(neg_list, ignore_index=True) if neg_list else pd.DataFrame()


# =========================================================
# 6. 限制每位 user 的視窗數量
# =========================================================
def cap_windows_per_user(df_windows, max_windows_per_user):
    if len(df_windows) == 0:
        return df_windows.copy()

    kept = []
    for username, g in df_windows.groupby("username"):
        g = g.sort_values(["window_start", "window_end"]).copy()

        if len(g) > max_windows_per_user:
            # 先簡單保留前 max_windows_per_user 個
            g = g.iloc[:max_windows_per_user].copy()

        kept.append(g)

    return pd.concat(kept, ignore_index=True)


dedup_pos_df = cap_windows_per_user(dedup_pos_df, MAX_POS_WINDOWS_PER_USER)
neg_df = cap_windows_per_user(neg_df, MAX_NEG_WINDOWS_PER_USER)


# =========================================================
# 7. 以 user 為單位切分 train / val / test
# =========================================================
pos_users = set(dedup_pos_df["username"].unique()
                ) if len(dedup_pos_df) > 0 else set()
all_users = set(df["username"].unique())

user_split_df = pd.DataFrame({"username": sorted(all_users)})
user_split_df["user_has_positive_window"] = user_split_df["username"].isin(
    pos_users).astype(int)

# 先 train / temp
train_users, temp_users = train_test_split(
    user_split_df,
    test_size=(1 - TRAIN_RATIO),
    random_state=RANDOM_SEED,
    stratify=user_split_df["user_has_positive_window"]
)

# 再 val / test
temp_ratio = VAL_RATIO + TEST_RATIO
val_size_in_temp = VAL_RATIO / temp_ratio

val_users, test_users = train_test_split(
    temp_users,
    test_size=(1 - val_size_in_temp),
    random_state=RANDOM_SEED,
    stratify=temp_users["user_has_positive_window"]
)

train_users = train_users.copy()
val_users = val_users.copy()
test_users = test_users.copy()

train_users["split"] = "train"
val_users["split"] = "val"
test_users["split"] = "test"

user_split_df = pd.concat(
    [train_users, val_users, test_users], ignore_index=True)


# =========================================================
# 8. 合併正負視窗，並指派 split
# =========================================================
window_df = pd.concat([dedup_pos_df, neg_df], ignore_index=True)

if len(window_df) == 0:
    raise ValueError("沒有任何 window 樣本產生，請檢查條件設定。")

window_df = window_df.merge(
    user_split_df[["username", "split", "user_has_positive_window"]],
    on="username",
    how="left"
)

# 若有 split 缺失，表示 user split 出問題
if window_df["split"].isna().any():
    raise ValueError("有部分 window 無法對應到 split，請檢查 user split。")


# =========================================================
# 9. 產生 selected_post_ids
# =========================================================
# 先建立 post lookup
post_lookup = df[["username", "post_id", "taken_at"]].copy()
post_lookup["post_id"] = post_lookup["post_id"].astype(str)
selected_post_ids_all = []
selected_post_count_all = []
padding_needed_all = []

anchor_post_count_all = []
prev_context_count_all = []

for _, row in window_df.iterrows():
    username = row["username"]
    window_start = pd.to_datetime(
        row["window_start"], utc=True, errors="coerce")
    window_end = pd.to_datetime(row["window_end"], utc=True, errors="coerce")
    anchor_ids = row["anchor_post_ids"]

    anchor_count = len(anchor_ids)

    selected_ids, selected_count, padding_needed = select_posts_for_model_with_prev_context(
        df_all_user_posts=df,
        username=username,
        window_start=window_start,
        window_end=window_end,
        anchor_post_ids=anchor_ids,
        model_window_size=MODEL_WINDOW_SIZE,
    )

    prev_context_count = max(0, selected_count - anchor_count)

    selected_post_ids_all.append(json.dumps(selected_ids, ensure_ascii=False))
    selected_post_count_all.append(selected_count)
    padding_needed_all.append(padding_needed)
    anchor_post_count_all.append(anchor_count)
    prev_context_count_all.append(prev_context_count)

window_df["selected_post_ids"] = selected_post_ids_all
window_df["selected_post_count"] = selected_post_count_all
window_df["padding_needed"] = padding_needed_all
window_df["anchor_post_count"] = anchor_post_count_all
window_df["prev_context_count"] = prev_context_count_all
window_df["source_rule"] = "14d_plus_prev_context"
# selected_post_ids_all = []
# selected_post_count_all = []
# padding_needed_all = []

# for _, row in window_df.iterrows():
#     username = row["username"]
#     anchor_ids = row["anchor_post_ids"]

#     g_posts = df[df["username"] == username].copy()
#     g_posts["post_id"] = g_posts["post_id"].astype(str)

#     window_posts = g_posts[g_posts["post_id"].isin(anchor_ids)].copy()
#     window_posts = window_posts.sort_values(["taken_at", "post_id"])

#     selected_ids, selected_count, padding_needed = select_posts_for_model(
#         window_posts,
#         model_window_size=MODEL_WINDOW_SIZE
#     )

#     selected_post_ids_all.append(json.dumps(selected_ids, ensure_ascii=False))
#     selected_post_count_all.append(selected_count)
#     padding_needed_all.append(padding_needed)

# window_df["selected_post_ids"] = selected_post_ids_all
# window_df["selected_post_count"] = selected_post_count_all
# window_df["padding_needed"] = padding_needed_all
# window_df["source_rule"] = "14d_exact_lastK"


# =========================================================
# 10. window_id
# =========================================================
window_df = window_df.reset_index(drop=True)
window_df["window_id"] = window_df.index.map(lambda x: f"w_{x:06d}")

# anchor_post_ids 轉成字串方便存檔
window_df["anchor_post_ids"] = window_df["anchor_post_ids"].apply(
    lambda x: json.dumps(list(x), ensure_ascii=False)
)


# =========================================================
# 11. 補 user split 摘要欄位
# =========================================================
pos_count_map = window_df[window_df["window_label"]
                          == 1].groupby("username").size().to_dict()
neg_count_map = window_df[window_df["window_label"]
                          == 0].groupby("username").size().to_dict()

user_split_df["num_positive_windows"] = user_split_df["username"].map(
    pos_count_map).fillna(0).astype(int)
user_split_df["num_negative_windows"] = user_split_df["username"].map(
    neg_count_map).fillna(0).astype(int)


# =========================================================
# 12. 輸出欄位整理
# =========================================================
window_df = window_df[[
    "window_id",
    "username",
    "split",
    "window_label",
    "window_type",
    "window_start",
    "window_end",
    "n_posts_in_14d",
    "n_high_risk_posts",
    "mean_p_depression_tcal",
    "anchor_post_ids",
    "anchor_post_count",
    "selected_post_ids",
    "selected_post_count",
    "prev_context_count",
    "padding_needed",
    "source_rule",
]].copy()

user_split_df = user_split_df[[
    "username",
    "split",
    "user_has_positive_window",
    "num_positive_windows",
    "num_negative_windows",
]].copy()


# =========================================================
# 13. 存檔
# =========================================================
window_df.to_csv(OUTPUT_WINDOW_DATASET, index=False, encoding="utf-8-sig")
user_split_df.to_csv(OUTPUT_USER_SPLIT, index=False, encoding="utf-8-sig")

print("=== 完成 window-level dataset 建構 ===")
print(f"window_dataset rows: {len(window_df)}")
print(f"users total: {user_split_df['username'].nunique()}")
print(f"positive windows: {(window_df['window_label'] == 1).sum()}")
print(f"negative windows: {(window_df['window_label'] == 0).sum()}")
print(f"train rows: {(window_df['split'] == 'train').sum()}")
print(f"val rows: {(window_df['split'] == 'val').sum()}")
print(f"test rows: {(window_df['split'] == 'test').sum()}")
print(f"\n輸出：\n{OUTPUT_WINDOW_DATASET}\n{OUTPUT_USER_SPLIT}")
