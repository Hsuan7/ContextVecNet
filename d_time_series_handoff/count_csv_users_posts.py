import csv
from pathlib import Path
from typing import Dict, Set, Tuple


BASE_DIR = Path(__file__).resolve().parent
FILES = {
    "憂鬱資料": BASE_DIR / "depress_dataset" / "貼文資料.csv",
    "非憂鬱資料": BASE_DIR / "depress_dataset" / "非憂鬱標籤貼文.csv",
}


def count_posts_and_users(csv_path: Path) -> Tuple[int, Set[str], int]:
    post_count = 0
    users: Set[str] = set()
    blank_username_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None or "username" not in reader.fieldnames:
            raise ValueError(f"{csv_path} 找不到 username 欄位")

        for row in reader:
            post_count += 1
            username = (row.get("username") or "").strip()

            if username:
                users.add(username)
            else:
                blank_username_count += 1

    return post_count, users, blank_username_count


def main() -> None:
    results: Dict[str, Tuple[int, Set[str], int]] = {}

    for label, csv_path in FILES.items():
        if not csv_path.is_file():
            raise FileNotFoundError(f"找不到檔案：{csv_path}")
        results[label] = count_posts_and_users(csv_path)

    first_label, second_label = FILES
    first_posts, first_users, first_blanks = results[first_label]
    second_posts, second_users, second_blanks = results[second_label]
    overlapping_users = first_users & second_users

    print(f"{first_label}：")
    print(f"  貼文數：{first_posts:,}")
    print(f"  唯一使用者數：{len(first_users):,}")
    print(f"  username 空白的貼文數：{first_blanks:,}")

    print(f"\n{second_label}：")
    print(f"  貼文數：{second_posts:,}")
    print(f"  唯一使用者數：{len(second_users):,}")
    print(f"  username 空白的貼文數：{second_blanks:,}")

    print(f"\n兩份資料重複使用者數：{len(overlapping_users):,}")

    if overlapping_users:
        output_path = BASE_DIR / "重複使用者.csv"
        with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["username"])
            writer.writerows([username] for username in sorted(overlapping_users))
        print(f"重複使用者名單：{output_path}")


if __name__ == "__main__":
    main()
