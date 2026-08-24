import os
import shutil
import stat
from pathlib import Path
import pandas as pd
import json

# ===== 設定 =====
SRC_ROOT = Path(r"D:\時間序列\ContextVecNet_Instagram")
DST_ROOT = Path(r"D:\時間序列\ContextVecNet_Instagram_filtered_new_2")

MIN_POSTS = 16  # 👉 可以改 5 / 10 / 20
MIN_DAYS_HISTORY = 30  # 最少要有多少天的貼文歷史

# 清空舊資料（更強健的刪除，處理唯讀檔與 Windows 可能的錯誤）


def _on_rm_error(func, path, exc_info):
    # 對於唯讀檔或權限問題，改為可寫後重試
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        # 若仍失敗，讓上層處理並繼續
        raise


if DST_ROOT.exists():
    try:
        shutil.rmtree(DST_ROOT, onerror=_on_rm_error)
    except Exception as e:
        print(f"Warning: failed to fully remove {DST_ROOT}: {e}")
        print("Proceeding and attempting to recreate target directory.")

DST_ROOT.mkdir(parents=True, exist_ok=True)

BALANCE_NEGATIVE_TO_POSITIVE = True  # 是否讓 negative 使用者數對齊 positive

total_users = 0
kept_users = 0
removed_users = 0
split_stats = {
    "positive": {"total": 0, "kept": 0, "removed": 0},
    "negative": {"total": 0, "kept": 0, "removed": 0},
}
kept_users_info = {
    "positive": [],
    "negative": [],
}

for split in ["positive", "negative"]:
    src_split = SRC_ROOT / split
    for user_dir in src_split.iterdir():
        if not user_dir.is_dir():
            continue

        total_users += 1
        split_stats[split]["total"] += 1

        meta_path = user_dir / "metadata.csv"
        if not meta_path.exists():
            continue

        df = pd.read_csv(meta_path)

        # ===== 1️⃣ 保留有圖片的貼文 =====
        df["image_path"] = df["image_path"].fillna("")
        df_filtered = df[df["image_path"].str.strip() != ""].copy()

        # ===== 2️⃣ 過濾貼文太少的 user =====
        if len(df_filtered) < MIN_POSTS:
            removed_users += 1
            split_stats[split]["removed"] += 1
            continue

        # ===== 2.5️⃣ 檢查貼文歷史跨度（天數） =====
        # 需要最少 MIN_DAYS_HISTORY 天的歷史
        df_filtered["taken_at_parsed"] = pd.to_datetime(
            df_filtered["taken_at"], errors="coerce"
        )
        valid_dates = df_filtered["taken_at_parsed"].dropna()
        if len(valid_dates) == 0:
            removed_users += 1
            split_stats[split]["removed"] += 1
            continue

        days_span = (valid_dates.max() - valid_dates.min()).days
        if days_span < MIN_DAYS_HISTORY:
            removed_users += 1
            split_stats[split]["removed"] += 1
            continue

        kept_users_info[split].append({
            "user_dir": user_dir,
            "df": df_filtered,
            "num_posts": len(df_filtered),
        })
        kept_users += 1
        split_stats[split]["kept"] += 1

# ===== 3️⃣ 依 positive 使用者數對齊 negative =====
# if BALANCE_NEGATIVE_TO_POSITIVE:
#     positive_count = len(kept_users_info["positive"])
#     negative_count = len(kept_users_info["negative"])

#     if negative_count > positive_count:
#         kept_users_info["negative"] = sorted(
#             kept_users_info["negative"],
#             key=lambda x: x["num_posts"],
#             reverse=True,
#         )[:positive_count]

#         removed_by_balance = negative_count - positive_count
#         removed_users += removed_by_balance
#         split_stats["negative"]["removed"] += removed_by_balance
#         split_stats["negative"]["kept"] = positive_count
#         kept_users -= removed_by_balance
#         print(
#             f"Balance negative -> positive: keep top {positive_count} negative users by post count")
#     else:
#         print("Negative user count is less than or equal to positive count; no balancing applied.")

# ===== 4️⃣ 複製資料到目標資料夾 =====
for split in ["positive", "negative"]:
    dst_split = DST_ROOT / split
    dst_split.mkdir(parents=True, exist_ok=True)

    for info in kept_users_info[split]:
        user_dir = info["user_dir"]
        df_filtered = info["df"]
        dst_user_dir = dst_split / user_dir.name

        # 先建立使用者目的資料夾
        dst_user_dir.mkdir(parents=True, exist_ok=True)

        for _, row in df_filtered.iterrows():
            img_rel = row["image_path"]
            if pd.isna(img_rel) or str(img_rel).strip() == "":
                continue

            # 若 image_path 為絕對路徑，轉成相對路徑以免覆寫根目錄
            rel = Path(str(img_rel))
            if rel.is_absolute():
                rel = Path(*rel.parts[1:])

            src_img = user_dir / rel
            dst_img = dst_user_dir / rel
            try:
                dst_img.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: failed to create dir {dst_img.parent}: {e}")

            if src_img.exists():
                try:
                    shutil.copy2(src_img, dst_img)
                except Exception:
                    # 嘗試用較低階的複製方式作為回退
                    try:
                        with open(src_img, "rb") as fsrc:
                            with open(dst_img, "wb") as fdst:
                                shutil.copyfileobj(fsrc, fdst)
                    except Exception as e:
                        print(f"Warning: failed to copy {src_img} -> {dst_img}: {e}")
            else:
                print(f"Warning: src image not found: {src_img}")
        df_filtered.to_csv(
            dst_user_dir / "metadata.csv",
            index=False,
            encoding="utf-8-sig"
        )

        with open(dst_user_dir / "timeline.txt", "w", encoding="utf-8") as f:
            for _, row in df_filtered.iterrows():
                rec = {
                    "id": str(row["post_id"]),
                    "text": row["caption"],
                    "created_at": row["taken_at"],
                    "image_path": row["image_path"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("\n===== Summary =====")
print("Total users:", total_users)
print("Kept users:", kept_users)
print("Removed users:", removed_users)
print("\n===== Split Summary =====")
for split in ["positive", "negative"]:
    stats = split_stats[split]
    print(
        f"{split}: total={stats['total']}, kept={stats['kept']}, removed={stats['removed']}"
    )
print("Output:", DST_ROOT)
