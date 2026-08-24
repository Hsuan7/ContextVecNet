import os
import ast
import json
from pathlib import Path

import pandas as pd
import numpy as np

from datasets.preprocessing import preprocess_image
from datasets.time_dataset_learn import TimeDatasetLearn


# =========================================================
# 2. 小工具
# =========================================================
def parse_list_column(value):
    """
    支援：
    - '["a", "b"]'
    - "['a', 'b']"
    - list 本身
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    s = str(value).strip()
    if not s:
        return []

    try:
        out = ast.literal_eval(s)
        if isinstance(out, list):
            return [str(x) for x in out]
    except Exception:
        pass

    try:
        out = json.loads(s)
        if isinstance(out, list):
            return [str(x) for x in out]
    except Exception:
        pass

    return []


from pathlib import Path
import pandas as pd


def parse_local_media_paths(value):
    """
    支援 local_media_path 可能是：
    - 單一路徑
    - 多個路徑以 ; 分隔
    """
    if pd.isna(value):
        return []

    s = str(value).strip()
    if not s:
        return []

    parts = [x.strip().strip('"').strip("'") for x in s.split(";")]
    parts = [p for p in parts if p]
    return parts


def resolve_image_path(local_media_path, media_type=None):
    """
    第一版規則：
    - local_media_path 若有多個檔案，用 ; 切開
    - 只取第一個有效圖片
    - 影片直接略過
    """
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    candidates = parse_local_media_paths(local_media_path)

    for item in candidates:
        p = Path(str(item))
        suffix = p.suffix.lower()

        if suffix in video_exts:
            continue

        if suffix in image_exts:
            # 直接存在
            if p.exists() and p.is_file():
                return str(p)

            # 若是相對路徑，嘗試補成 /mnt/d/時間序列/depress_dataset/...
            # 你可以依實際資料根目錄調整
            alt = Path("/mnt/d/時間序列/depress_dataset") / p
            if alt.exists() and alt.is_file():
                return str(alt)

    return None


# =========================================================
# 3. Window-level Dataset
# =========================================================
class WindowLevelDataset(TimeDatasetLearn):
    """
    一筆樣本 = 一個 window

    讀：
    - window_dataset.csv：決定每筆 window 的 label / selected_post_ids / split
    - post-level csv：用 post_id 找 caption / taken_at / local_media_path

    回傳：
    - 與 ContextVecNet 現有 pipeline 相容的 sample dict
    """

    def __init__(self, args, kind="train"):
        """
        kind: train / val / test
        這裡假設你會在 args 裡新增兩個路徑：
        - args.window_dataset_csv
        - args.post_level_csv
        """
        self.args = args
        self.kind = kind
        self.window_size = self.args.window_size

        # 你需要在 args 中提供這兩個路徑
        window_csv = getattr(args, "window_dataset_csv", None)
        post_csv = getattr(args, "post_level_csv", None)

        if window_csv is None:
            raise ValueError("args.window_dataset_csv 未設定")
        if post_csv is None:
            raise ValueError("args.post_level_csv 未設定")

        self.window_df = pd.read_csv(window_csv)
        self.post_df = pd.read_csv(post_csv)

        # 基本清理
        self.window_df.columns = [c.strip() for c in self.window_df.columns]
        self.post_df.columns = [c.strip() for c in self.post_df.columns]

        required_window_cols = [
            "window_id",
            "username",
            "split",
            "window_label",
            "selected_post_ids",
        ]
        for col in required_window_cols:
            if col not in self.window_df.columns:
                raise ValueError(f"window_dataset.csv 缺少必要欄位: {col}")

        required_post_cols = [
            "username",
            "post_id",
            "caption",
            "taken_at",
        ]
        for col in required_post_cols:
            if col not in self.post_df.columns:
                raise ValueError(f"post-level csv 缺少必要欄位: {col}")

        # split 過濾
        split_name = kind
        if kind == "valid":
            split_name = "val"

        self.window_df = self.window_df[self.window_df["split"] == split_name].copy(
        )
        self.window_df = self.window_df.reset_index(drop=True)
        # 兼容 main_maple.py 與舊 trainer 邏輯
        self.labels = self.window_df["window_label"].astype(int).tolist()
        self.users = self.window_df["window_id"].astype(str).tolist()

        # post-level 處理
        self.post_df["post_id"] = self.post_df["post_id"].astype(str)
        self.post_df["username"] = self.post_df["username"].astype(
            str).str.strip()
        self.post_df["caption"] = self.post_df["caption"].fillna(
            "").astype(str)
        self.post_df["taken_at"] = pd.to_datetime(
            self.post_df["taken_at"], utc=True, errors="coerce"
        )

        # 建 post lookup：用 (username, post_id) 查貼文
        # 若同一 pair 重複，只保留第一筆
        self.post_df = self.post_df.drop_duplicates(
            subset=["username", "post_id"]).copy()
        self.post_lookup = {
            (row["username"], row["post_id"]): row
            for _, row in self.post_df.iterrows()
        }

        print("==== WindowLevelDataset Debug ====")
        print("kind =", kind)
        print("window_csv =", window_csv)
        print("post_csv =", post_csv)
        print("num windows =", len(self.window_df))

    def __len__(self):
        return len(self.window_df)

    def __getitem__(self, idx):
        row = self.window_df.iloc[idx]

        window_id = str(row["window_id"])
        username = str(row["username"]).strip()
        label = int(row["window_label"])

        selected_post_ids = parse_list_column(row["selected_post_ids"])

        texts = []
        dates = []
        images = []
        images_paths = []

        for pid in selected_post_ids:
            key = (username, str(pid))
            post = self.post_lookup.get(key, None)

            if post is None:
                # 找不到貼文，就跳過；最後若全空再補 placeholder
                continue

            caption = post["caption"]
            taken_at = post["taken_at"]

            if pd.isna(taken_at):
                continue

            # 時間轉 unix timestamp
            timestamp = int(round(pd.Timestamp(taken_at).timestamp()))

            # 圖片路徑（如果你的 post csv 有 local_media_path / media_type）
            local_media_path = post["local_media_path"] if "local_media_path" in post.index else None
            media_type = post["media_type"] if "media_type" in post.index else None

            img_path = resolve_image_path(local_media_path, media_type)
            preprocessed_img, updated_path = preprocess_image(
                img_path,
                training=self.kind == "train",
            )

            texts.append(caption)
            dates.append(timestamp)
            images.append(preprocessed_img)
            images_paths.append(updated_path)

        # 若 selected_post_ids 都失敗，至少補一筆 placeholder，避免空序列炸掉
        if len(texts) == 0:
            texts = [""]
            dates = [0]
            dummy_img, dummy_path = preprocess_image(
                None,
                training=self.kind == "train",
            )
            images = [dummy_img]
            images_paths = [dummy_path]

        # 交給父類別做排序 / windowing / padding / masks
        sample = self.load_multimodal(
            images=images,
            texts=texts,
            dates=dates,
            label=label,
            user_name=f"{username}__{window_id}",
            images_paths=images_paths,
        )

        # 額外塞一些 window-level metadata，方便 debug / analysis
        sample["window_id"] = window_id
        sample["username"] = username
        sample["window_label"] = label
        sample["selected_post_ids"] = selected_post_ids

        return sample
