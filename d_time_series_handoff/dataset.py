import os
import re
import json
import shutil
from pathlib import Path
from typing import Optional, List

import cv2
import pandas as pd


# =========================================================
# 1. 路徑設定
# =========================================================
POST_CSV = Path(r"D:\時間序列\labeled\post_with_user_label.csv")
OUTPUT_ROOT = Path(r"D:\時間序列\ContextVecNet_Instagram")
MEDIA_ROOT = Path(r"D:\時間序列\depress_dataset\downloaded_media")

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv"]


# =========================================================
# 2. 基本工具函式
# =========================================================
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def is_video_file(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def extract_shortcode(post_url: str) -> Optional[str]:
    if not isinstance(post_url, str):
        return None
    m = re.search(r"/(?:p|reel|tv)/([^/]+)/?", post_url)
    return m.group(1) if m else None


def parse_local_media_paths(value) -> List[Path]:
    """
    支援：
    - 單一路徑字串
    - 用 ; 串接的多路徑
    - 空值
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    value = str(value).strip()
    if value == "":
        return []

    parts = [p.strip() for p in value.split(";") if p.strip()]
    return [Path(p) for p in parts]


def normalize_path_for_local_file(p: Path) -> Path:
    """
    若 CSV 中是相對路徑 downloaded_media\\user\\xxx.jpg
    就補到 MEDIA_ROOT 的父層去組出完整路徑
    """
    if p.is_absolute():
        return p

    p_str = str(p).replace("\\", os.sep).replace("/", os.sep)

    # 若本身就包含 downloaded_media 開頭
    if p_str.startswith("downloaded_media" + os.sep) or p_str == "downloaded_media":
        media_parent = MEDIA_ROOT.parent
        return media_parent / Path(p_str)

    # 否則視為 MEDIA_ROOT 之下的相對路徑
    return MEDIA_ROOT / p


def extract_middle_frame(video_path: Path, output_dir: Path) -> Optional[Path]:
    print(f"[VIDEO DEBUG] trying: {video_path}")

    if not video_path.exists() or not video_path.is_file():
        print("[VIDEO DEBUG] path not exists or not file")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[VIDEO DEBUG] VideoCapture open failed")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[VIDEO DEBUG] frame_count={frame_count}, fps={fps}")

    if frame_count <= 0:
        print("[VIDEO DEBUG] invalid frame_count")
        cap.release()
        return None

    middle_idx = frame_count // 2
    print(f"[VIDEO DEBUG] middle_idx={middle_idx}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)

    success, frame = cap.read()
    print(
        f"[VIDEO DEBUG] read success={success}, frame is None={frame is None}")
    cap.release()

    if not success or frame is None:
        print("[VIDEO DEBUG] failed to read middle frame")
        return None

    output_path = output_dir / f"{video_path.stem}_midframe.jpg"

    success, buffer = cv2.imencode(".jpg", frame)
    print(f"[VIDEO DEBUG] imencode success={success}, output={output_path}")

    if not success:
        return None

    try:
        buffer.tofile(str(output_path))   # Windows 中文路徑較穩
    except Exception as e:
        print(f"[VIDEO DEBUG] tofile failed: {e}")
        return None

    if not output_path.exists():
        print("[VIDEO DEBUG] output file not created")
        return None

    return output_path

# =========================================================
# 3. 為單篇貼文挑代表圖
# 規則：
# - Photo：直接用該 jpg
# - Album：若多張 jpg，取第一張
# - Album 若只有 mp4，擷取中間影格
# - Video：擷取中間影格
# - 若 local_media_path 無效，去 downloaded_media/{username} 內用 shortcode 找
# =========================================================


def resolve_representative_image(row: pd.Series) -> Optional[Path]:
    username = safe_str(row.get("username", ""))
    media_type = safe_str(row.get("media_type", "")).lower()
    post_url = safe_str(row.get("post_url", ""))
    shortcode = extract_shortcode(post_url)

    frame_dir = OUTPUT_ROOT / "_video_frames" / username
    local_candidates = parse_local_media_paths(
        row.get("local_media_path", None))
    local_candidates = [normalize_path_for_local_file(
        p) for p in local_candidates]

    # -----------------------------------------------------
    # 1) local_media_path 優先
    # Photo / Album：取第一張有效圖片
    # -----------------------------------------------------
    for p in local_candidates:
        if p.exists() and p.is_file() and is_image_file(p):
            print(f"[DEBUG] {username} | {shortcode} -> IMAGE {p.name}")
            return p

    # -----------------------------------------------------
    # 2) local_media_path 沒有圖，但有影片
    # Video / Album(mp4 only)：取中間影格
    # -----------------------------------------------------
    for p in local_candidates:
        if p.exists() and p.is_file() and is_video_file(p):
            frame_path = extract_middle_frame(p, frame_dir)
            if frame_path is not None:
                print(
                    f"[DEBUG] {username} | {shortcode} -> VIDEO_FRAME {frame_path.name}")
                return frame_path
            print(f"[DEBUG] {username} | {shortcode} -> VIDEO_EXTRACT_FAILED")
            return None

    # -----------------------------------------------------
    # 3) local_media_path 不可用時，去使用者資料夾用 shortcode 找
    # -----------------------------------------------------
    user_dir = MEDIA_ROOT / username
    if not user_dir.exists():
        print(f"[DEBUG] {username} | {shortcode} -> NO_USER_DIR")
        return None

    all_files = sorted(list(user_dir.glob("*")))

    if shortcode:
        matched = [p for p in all_files if shortcode in p.name]

        # 3-1) 先找圖片
        matched_images = [p for p in matched if is_image_file(p)]
        if matched_images:
            chosen = sorted(matched_images)[0]
            print(
                f"[DEBUG] {username} | {shortcode} -> MATCH_IMAGE {chosen.name}")
            return chosen

        # 3-2) 再找影片
        matched_videos = [p for p in matched if is_video_file(p)]
        if matched_videos:
            chosen_video = sorted(matched_videos)[0]
            frame_path = extract_middle_frame(chosen_video, frame_dir)
            if frame_path is not None:
                print(
                    f"[DEBUG] {username} | {shortcode} -> MATCH_VIDEO_FRAME {frame_path.name}")
                return frame_path
            print(
                f"[DEBUG] {username} | {shortcode} -> MATCH_VIDEO_EXTRACT_FAILED")
            return None

    # -----------------------------------------------------
    # 4) 不再做 fallback 到使用者第一張圖
    # 找不到就是缺圖，後面交給 dataset 做黑圖 + mask
    # -----------------------------------------------------
    print(f"[DEBUG] {username} | {shortcode} -> NONE")
    return None


def infer_image_source_type(row: pd.Series, chosen_path: Optional[Path]) -> str:
    if chosen_path is None:
        return "none"

    media_type = safe_str(row.get("media_type", "")).lower()
    name = chosen_path.name.lower()

    if "_midframe.jpg" in name:
        if "album" in media_type:
            return "album_video_middle_frame"
        return "video_middle_frame"

    if "photo" in media_type:
        return "photo"
    if "album" in media_type:
        return "album_first_image"
    return "image"


def clean_caption(x) -> str:
    x = safe_str(x)
    return x.replace("\r", " ").replace("\n", " ").strip()


# =========================================================
# 4. 產出 ContextVecNet 需要的資料結構
# =========================================================
def build_contextvecnet_dataset():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(POST_CSV)

    # -----------------------------------------------------
    # 基本清理
    # -----------------------------------------------------
    df["username"] = df["username"].astype(str).str.strip()
    df["post_id"] = df["post_id"].astype(str).str.strip()
    df["taken_at"] = pd.to_datetime(df["taken_at"], utc=True, errors="coerce")
    df["caption"] = df["caption"].apply(clean_caption)

    # user_label 只保留 0/1
    df = df[df["user_label"].isin([0, 1])].copy()

    # 去掉沒時間或沒 username / post_id 的資料
    df = df.dropna(subset=["taken_at"])
    df = df[(df["username"] != "") & (df["post_id"] != "")].copy()

    total_posts_after_clean = len(df)
    total_users = df["username"].nunique()
    positive_users = df[df["user_label"] == 1]["username"].nunique()
    negative_users = df[df["user_label"] == 0]["username"].nunique()

    print("total_posts_after_clean:", total_posts_after_clean)
    print("total_users:", total_users)
    print("positive_users:", positive_users)
    print("negative_users:", negative_users)

    # -----------------------------------------------------
    # 建立輸出資料夾
    # ContextVecNet_Instagram/
    #   positive/
    #     userA/
    #       metadata.csv
    #       images/
    #   negative/
    #     userB/
    #       metadata.csv
    #       images/
    # -----------------------------------------------------
    pos_root = OUTPUT_ROOT / "positive"
    neg_root = OUTPUT_ROOT / "negative"
    pos_root.mkdir(exist_ok=True)
    neg_root.mkdir(exist_ok=True)

    posts_with_image = 0
    posts_without_image = 0
    video_posts_without_image = 0
    users_written = 0

    grouped = df.sort_values(
        ["username", "taken_at", "post_id"]).groupby("username")

    for username, g in grouped:
        g = g.sort_values(["taken_at", "post_id"]).copy()

        # 以第一個 user_label 為主
        user_label = int(g["user_label"].iloc[0])
        split_root = pos_root if user_label == 1 else neg_root

        user_dir = split_root / username
        image_out_dir = user_dir / "images"
        user_dir.mkdir(parents=True, exist_ok=True)
        image_out_dir.mkdir(parents=True, exist_ok=True)

        rows_out = []
        timeline_records = []

        for _, row in g.iterrows():
            img_src = resolve_representative_image(row)

            if img_src is not None and img_src.exists():
                # 統一複製到 user/images/
                ext = img_src.suffix.lower()
                out_name = f"{row['post_id']}{ext}"
                img_dst = image_out_dir / out_name

                if not img_dst.exists():
                    shutil.copy2(img_src, img_dst)

                image_rel_path = str(img_dst.relative_to(
                    user_dir)).replace("\\", "/")
                posts_with_image += 1
            else:
                image_rel_path = ""
                posts_without_image += 1
                media_type = safe_str(row.get("media_type", "")).lower()
                if "video" in media_type:
                    video_posts_without_image += 1

            image_source_type = infer_image_source_type(row, img_src)

            # 補上：收集 timeline 資料 (與舊程式格式一致)
            timeline_records.append({
                "id": str(row["post_id"]),
                "text": row["caption"],
                "created_at": row["taken_at"].isoformat(),
                "image_path": image_rel_path,
                "image_source_type": image_source_type
            })

            rows_out.append({
                "username": row["username"],
                "post_id": row["post_id"],
                "taken_at": row["taken_at"].isoformat(),
                "caption": row["caption"],
                "media_type": safe_str(row.get("media_type", "")),
                "image_path": image_rel_path,   # 相對於 user_dir
                "image_source_type": image_source_type,
                "user_label": int(row["user_label"]),
                "p_depression_t1": row.get("p_depression_t1", None),
                "p_depression_tcal": row.get("p_depression_tcal", None),
                "logit_margin": row.get("logit_margin", None),
                "pseudo_label": row.get("pseudo_label", None),
                "is_high_risk": row.get("is_high_risk", None),
            })

        meta_df = pd.DataFrame(rows_out)
        meta_df.to_csv(user_dir / "metadata.csv",
                       index=False, encoding="utf-8-sig")
        # -----------------------------------------------------
        # 補上：寫入 timeline.txt (與舊程式邏輯一致)
        # -----------------------------------------------------
        timeline_path = user_dir / "timeline.txt"
        with open(timeline_path, "w", encoding="utf-8") as f:
            for rec in timeline_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        users_written += 1

    user_with_img = meta_df["image_path"].astype(str).str.len().gt(0).sum()
    user_total = len(meta_df)
    print(
        f"[USER SUMMARY] {username}: {user_with_img}/{user_total} posts with image")

    print("posts_with_image:", posts_with_image)
    print("posts_without_image:", posts_without_image)
    print("video_posts_without_image:", video_posts_without_image)
    print("users_written:", users_written)


if __name__ == "__main__":
    build_contextvecnet_dataset()


# import os
# import json
# import shutil
# from pathlib import Path
# from typing import Optional, List
# import pandas as pd
# import re
# import cv2

# # =========================================================
# # 1. 路徑設定
# # =========================================================
# CSV_PATH = Path(r"D:\時間序列\labeled\post_with_user_label.csv")
# OUTPUT_ROOT = Path(r"D:\時間序列\ContextVecNet_Instagram")
# MEDIA_ROOT = Path(r"D:\時間序列\depress_dataset\downloaded_media")
# # =========================================================
# # 2. user_label 映射：1 -> positive, 0 -> negative
# # =========================================================
# def map_user_label_to_split(value) -> Optional[str]:
#     if pd.isna(value):
#         return None
#     try:
#         v = int(value)
#     except Exception:
#         return None

#     if v == 1:
#         return "positive"
#     if v == 0:
#         return "negative"
#     return None


# # =========================================================
# # 3. 文字清理
# # =========================================================
# def clean_text(text) -> str:
#     if pd.isna(text):
#         return ""
#     return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


# # =========================================================
# # 4. 時間轉 ISO 格式
# # =========================================================
# def parse_taken_at(value) -> Optional[str]:
#     if pd.isna(value):
#         return None

#     dt = pd.to_datetime(value, errors="coerce")
#     if pd.isna(dt):
#         return None

#     return dt.strftime("%Y-%m-%dT%H:%M:%S")

# def extract_middle_frame(video_path: Path, output_dir: Path) -> Optional[Path]:
#     """
#     從影片擷取中間影格，存成 jpg
#     """
#     if not video_path.exists() or not video_path.is_file():
#         return None

#     output_dir.mkdir(parents=True, exist_ok=True)

#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         return None

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if frame_count <= 0:
#         cap.release()
#         return None

#     middle_idx = frame_count // 2
#     cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)

#     success, frame = cap.read()
#     cap.release()

#     if not success or frame is None:
#         return None

#     output_path = output_dir / f"{video_path.stem}_midframe.jpg"
#     ok = cv2.imwrite(str(output_path), frame)
#     if not ok:
#         return None

#     return output_path


# IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
# VIDEO_EXTS = [".mp4", ".mov", ".avi"]


# def is_image_file(p: Path):
#     return p.suffix.lower() in IMAGE_EXTS


# def is_video_file(p: Path):
#     return p.suffix.lower() in VIDEO_EXTS


# def extract_shortcode(post_url: str) -> Optional[str]:
#     if not isinstance(post_url, str):
#         return None
#     m = re.search(r"/p/([^/]+)/", post_url)
#     return m.group(1) if m else None


# def parse_local_media_paths(path_str):
#     if not isinstance(path_str, str) or path_str.strip() == "":
#         return []
#     paths = [p.strip() for p in path_str.split(";")]
#     return [Path(p) for p in paths]


# def extract_middle_frame(video_path: Path, output_dir: Path) -> Optional[Path]:
#     if not video_path.exists() or not video_path.is_file():
#         return None

#     output_dir.mkdir(parents=True, exist_ok=True)

#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         return None

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if frame_count <= 0:
#         cap.release()
#         return None

#     middle_idx = frame_count // 2
#     cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)

#     success, frame = cap.read()
#     cap.release()

#     if not success or frame is None:
#         return None

#     output_path = output_dir / f"{video_path.stem}_midframe.jpg"
#     ok = cv2.imwrite(str(output_path), frame)
#     if not ok:
#         return None

#     return output_path


# def resolve_representative_image(row) -> Optional[Path]:
#     username = str(row.get("username", "")).strip()
#     media_type = str(row.get("media_type", "")).lower()
#     post_url = row.get("post_url", "")
#     shortcode = extract_shortcode(post_url)

#     # 你可以自己調整這個輸出資料夾
#     frame_dir = MEDIA_ROOT / "_video_frames" / username

#     selected = None
#     local_candidates = parse_local_media_paths(row.get("local_media_path"))

#     # =========================================================
#     # 1. local_media_path 優先
#     # Photo / Album：先找第一張圖片
#     # =========================================================
#     for p in local_candidates:
#         if p.exists() and is_image_file(p):
#             selected = p
#             print(f"[DEBUG] {username} | {shortcode} → {selected.name}")
#             return selected

#     # =========================================================
#     # 2. 若沒有圖片，但有影片，擷取中間影格
#     # =========================================================
#     for p in local_candidates:
#         if p.exists() and is_video_file(p):
#             frame_path = extract_middle_frame(p, frame_dir)
#             if frame_path is not None:
#                 print(
#                     f"[DEBUG] {username} | {shortcode} → VIDEO FRAME {frame_path.name}")
#                 return frame_path
#             print(
#                 f"[DEBUG] {username} | {shortcode} → VIDEO BUT EXTRACT FAILED")
#             return None

#     # =========================================================
#     # 3. 備援：去使用者資料夾找 shortcode
#     # =========================================================
#     user_dir = MEDIA_ROOT / username
#     if not user_dir.exists():
#         print(f"[DEBUG] {username} | {shortcode} → NO DIR")
#         return None

#     all_files = list(user_dir.glob("*"))

#     if shortcode:
#         matched = [p for p in all_files if shortcode in p.name]

#         images = [p for p in matched if is_image_file(p)]
#         if images:
#             selected = sorted(images)[0]
#             print(f"[DEBUG] {username} | {shortcode} → {selected.name}")
#             return selected

#         videos = [p for p in matched if is_video_file(p)]
#         if videos:
#             frame_path = extract_middle_frame(sorted(videos)[0], frame_dir)
#             if frame_path is not None:
#                 print(
#                     f"[DEBUG] {username} | {shortcode} → VIDEO FRAME {frame_path.name}")
#                 return frame_path
#             print(
#                 f"[DEBUG] {username} | {shortcode} → VIDEO MATCHED BUT EXTRACT FAILED")
#             return None

#     # =========================================================
#     # 4. fallback
#     # Photo / Album 才抓第一張圖片
#     # =========================================================
#     if "photo" in media_type or "album" in media_type:
#         images = [p for p in all_files if is_image_file(p)]
#         if images:
#             selected = sorted(images)[0]
#             print(
#                 f"[DEBUG] {username} | {shortcode} → FALLBACK {selected.name}")
#             return selected

#     # Video 類型 fallback：抓第一個影片做影格
#     if "video" in media_type:
#         videos = [p for p in all_files if is_video_file(p)]
#         if videos:
#             frame_path = extract_middle_frame(sorted(videos)[0], frame_dir)
#             if frame_path is not None:
#                 print(
#                     f"[DEBUG] {username} | {shortcode} → FALLBACK VIDEO FRAME {frame_path.name}")
#                 return frame_path

#     print(f"[DEBUG] {username} | {shortcode} → NONE")
#     return None

# # =========================================================
# # 9. 主程式：建立 ContextVecNet 可用資料集
# # =========================================================


# def build_contextvecnet_dataset():
#     if not CSV_PATH.exists():
#         raise FileNotFoundError(f"找不到 CSV：{CSV_PATH}")

#     OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(CSV_PATH)

#     required_cols = ["username", "post_id",
#                      "caption", "taken_at", "user_label"]
#     missing = [c for c in required_cols if c not in df.columns]
#     if missing:
#         raise ValueError(f"CSV 缺少必要欄位：{missing}")

#     # 基本清理
#     df["split_label"] = df["user_label"].apply(map_user_label_to_split)
#     df["username"] = df["username"].astype(str).str.strip()
#     df["post_id"] = df["post_id"].astype(str).str.strip()
#     df["caption_clean"] = df["caption"].apply(clean_text)
#     df["created_at_iso"] = df["taken_at"].apply(parse_taken_at)

#     # 保留 user_label / username / post_id / taken_at 可用的列
#     df = df[
#         df["split_label"].notna() &
#         df["username"].ne("") &
#         df["post_id"].ne("") &
#         df["created_at_iso"].notna()
#     ].copy()

#     # 同一使用者內，同一 post_id 只保留一筆
#     df = df.drop_duplicates(subset=["username", "post_id"], keep="first")

#     summary = {
#         "total_posts_after_clean": int(len(df)),
#         "total_users": int(df["username"].nunique()),
#         "positive_users": int(df.loc[df["split_label"] == "positive", "username"].nunique()),
#         "negative_users": int(df.loc[df["split_label"] == "negative", "username"].nunique()),
#         "posts_with_image": 0,
#         "posts_without_image": 0,
#         "video_posts_without_image": 0,
#         "users_written": 0,
#     }

#     grouped = df.groupby(["split_label", "username"], sort=True)

#     for split_label, username in grouped.groups.keys():
#         user_df = grouped.get_group((split_label, username)).copy()
#         user_df = user_df.sort_values("created_at_iso")

#         user_out_dir = OUTPUT_ROOT / split_label / username
#         user_out_dir.mkdir(parents=True, exist_ok=True)

#         timeline_records = []

#         for _, row in user_df.iterrows():
#             post_id = str(row["post_id"]).strip()
#             caption = row["caption_clean"]
#             created_at = row["created_at_iso"]
#             media_type = str(row.get("media_type", "")).strip().lower()

#             img_src = resolve_representative_image(row)
#             img_written = False

#             if img_src is not None and img_src.exists():
#                 dst_img = user_out_dir / f"{post_id}.jpg"
#                 try:
#                     shutil.copy2(img_src, dst_img)
#                     img_written = True
#                     summary["posts_with_image"] += 1
#                 except Exception as e:
#                     print(f"[WARN] 複製圖片失敗: {img_src} -> {dst_img} | {e}")
#                     summary["posts_without_image"] += 1
#             else:
#                 summary["posts_without_image"] += 1
#                 if any(k in media_type for k in ["video", "clips", "reel", "mp4", "mov"]):
#                     summary["video_posts_without_image"] += 1

#             # 不論有沒有圖，都保留 timeline
#             timeline_records.append({
#                 "id": post_id,
#                 "text": caption,
#                 "created_at": created_at
#             })

#         timeline_path = user_out_dir / "timeline.txt"
#         with open(timeline_path, "w", encoding="utf-8") as f:
#             for rec in timeline_records:
#                 f.write(json.dumps(rec, ensure_ascii=False) + "\n")

#         summary["users_written"] += 1

#     summary_path = OUTPUT_ROOT / "build_summary.json"
#     with open(summary_path, "w", encoding="utf-8") as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)

#     print("=== 建立完成 ===")
#     for k, v in summary.items():
#         print(f"{k}: {v}")
#     print(f"輸出資料夾：{OUTPUT_ROOT}")
#     print(f"摘要檔：{summary_path}")


# if __name__ == "__main__":
#     build_contextvecnet_dataset()
