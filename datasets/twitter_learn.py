import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from datasets.time_dataset_learn import TimeDatasetLearn
from datasets.preprocessing import preprocess_image
import random
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "MultiModalDataset"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/mnt/d/時間序列/ContextVecNet_Instagram_filtered_new"
TEXT_FILE_NAME = "timeline.txt"

# 改 增加validation threshold sweep
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def threshold_sweep_report(y_true, y_score, thresholds=None, zero_division=0):
    """
    對 validation 分數做 threshold sweep。

    參數
    ----
    y_true : list / np.ndarray
        真實標籤，0/1
    y_score : list / np.ndarray
        正類（positive class）的預測分數或機率
    thresholds : list[float]
        要掃描的 threshold 清單
    zero_division : int
        當 precision/recall 無法定義時的回傳值

    回傳
    ----
    rows : list[dict]
        每個 threshold 的指標結果
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    rows = []

    print("\n===== Validation Threshold Sweep =====")
    print(f"{'thr':>6} | {'acc':>8} | {'prec':>8} | {'rec':>8} | {'f1':>8} | {'pos_pred':>8}")
    print("-" * 64)

    for thr in thresholds:
        # y_score = 1 - out
        y_pred = (y_score >= thr).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=zero_division)
        rec = recall_score(y_true, y_pred, zero_division=zero_division)
        f1 = f1_score(y_true, y_pred, zero_division=zero_division)
        pos_pred = int(y_pred.sum())

        row = {
            "threshold": thr,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "num_positive_pred": pos_pred,
        }
        rows.append(row)

        print(
            f"{thr:6.2f} | {acc:8.4f} | {prec:8.4f} | {rec:8.4f} | {f1:8.4f} | {pos_pred:8d}")

    best_by_f1 = max(rows, key=lambda x: x["f1"])
    print("-" * 64)
    print("Best threshold by F1:", best_by_f1)

    return rows


class CombinedTwitterDataset(TimeDatasetLearn):
    """
    A dataset class that combines Twitter user data with multimodal content (images and texts).

    This dataset extends TimeDatasetLearn to load and process Twitter data by reading user directories,
    splitting them into training, validation, and test folds, and applying image preprocessing along with
    multimodal data aggregation.

    Attributes:
        args: Configuration parameters including window_size, num_folds, fold, etc.
        kind (str): The dataset split type ("train", "valid", or "test").
        window_size (int): The window size for multimodal data loading.
        users (list): List of user directory paths included in the dataset.
        labels (list): List of labels corresponding to each user (1 for positive, 0 for negative).
        positive_users (list): List of directories for positive users in the current fold.
        negative_users (list): List of directories for negative users in the current fold.
    """

    def __init__(self, args, kind="train"):
        """
        Initialize the CombinedTwitterDataset by splitting users into folds and setting up labels.

        The dataset collects user directories for both positive and negative samples from the specified
        data path. Depending on the 'kind' parameter, it selects a portion of the data for validation/test
        or the remaining data for training.

        Args:
            args: An object containing configuration parameters such as window_size, num_folds, fold, etc.
            kind (str, optional): The dataset split type ("train", "valid", or "test"). Defaults to "train".
        """
        self.args = args
        self.kind = kind
        self.window_size = self.args.window_size

        print("DATA_PATH =", DATA_PATH)
        print("exists(DATA_PATH) =", os.path.exists(DATA_PATH))

        positive_users = sorted(glob.glob(f"{DATA_PATH}/positive/*"))
        negative_users = sorted(glob.glob(f"{DATA_PATH}/negative/*"))

        rng = random.Random(42)
        rng.shuffle(positive_users)
        rng.shuffle(negative_users)

        print("num positive_users =", len(positive_users))
        print("num negative_users =", len(negative_users))

        n_pos = len(positive_users)
        n_neg = len(negative_users)

        pos_fold_size = n_pos // self.args.num_folds
        neg_fold_size = n_neg // self.args.num_folds

        def get_fold(users, fold_id, fold_size):
            start = fold_id * fold_size
            end = (fold_id + 1) * \
                fold_size if fold_id < self.args.num_folds - 1 else len(users)
            return users[start:end]

        # Leakage-free rotating 3/1/1 split:
        # - validation fold: checkpoint selection, Platt fitting, threshold sweep
        # - test fold: final evaluation only
        # - training folds: the remaining three folds
        val_fold = self.args.fold
        test_fold = (self.args.fold + 1) % self.args.num_folds

        pos_val = get_fold(positive_users, val_fold, pos_fold_size)
        neg_val = get_fold(negative_users, val_fold, neg_fold_size)
        pos_test = get_fold(positive_users, test_fold, pos_fold_size)
        neg_test = get_fold(negative_users, test_fold, neg_fold_size)

        val_users = set(pos_val + neg_val)
        test_users = set(pos_test + neg_test)

        if self.kind in ["valid", "val"]:
            self.users = pos_val + neg_val
        elif self.kind == "test":
            self.users = pos_test + neg_test
        else:
            self.users = [
                u for u in positive_users + negative_users
                if u not in val_users and u not in test_users
            ]

        print(
            f"split kind={self.kind} outer_fold={self.args.fold} "
            f"val_fold={val_fold} test_fold={test_fold}"
        )
        # ===== keep only valid users =====
        self.users = [
            user_path for user_path in self.users
            if os.path.isfile(f"{user_path}/{TEXT_FILE_NAME}")
        ]

        # ===== labels =====
        self.labels = [
            1 if user_path.split("/")[-2] == "positive" else 0
            for user_path in self.users
        ]
        # ##### 有切分(原本testc04val重疊)
        # if self.kind in ["valid", "test"]:
        #     positive_users_fold = positive_users[start_idx_fold:end_idx_fold]
        #     negative_users_fold = negative_users[start_idx_fold:end_idx_fold]
        #     self.users = positive_users_fold + negative_users_fold
        # else:
        #     positive_users_fold = (
        #         positive_users[:start_idx_fold] + positive_users[end_idx_fold:]
        #     )
        #     negative_users_fold = (
        #         negative_users[:start_idx_fold] + negative_users[end_idx_fold:]
        #     )

        #     self.users = positive_users_fold + negative_users_fold

        # self.users = [
        #     user_path for user_path in self.users
        #     if os.path.isfile(f"{user_path}/{TEXT_FILE_NAME}")
        # ]
        # self.labels = [
        #     1 if user_path.split("/")[-2] == "positive" else 0
        #     for user_path in self.users
        # ]
        #####
        # 沒有切分，train / valid / test 都讀到同一批 173 位使用
        # self.users = positive_users + negative_users
        # self.users = [
        #     user_path for user_path in self.users
        #     if os.path.isfile(f"{user_path}/{TEXT_FILE_NAME}")
        # ]
        # self.labels = [
        #     1 if user_path.split("/")[-2] == "positive" else 0
        #     for user_path in self.users
        # ]
        #####

        # self.positive_users = positive_users_fold #都不要
        # self.negative_users = negative_users_fold

    def __len__(self):
        """
        Return the total number of user samples in the dataset.

        Returns:
            int: The number of user directories included in the dataset.
        """
        return len(self.users)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        This method loads text and image data for a given user by reading the user's timeline file.
        If the timeline file does not exist, it loads available images and returns an empty text list.
        The method applies image preprocessing and aggregates data using the load_multimodal function
        from the parent TimeDatasetLearn class.

        Args:
            idx (int): The index of the user sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - "author": The user identifier.
                - "texts": A list of text entries (empty if no timeline exists).
                - "images_paths": A list of image file paths or placeholders ("Blank image") if missing.
                - "images": A list of preprocessed image tensors.
                - "label": The label for the user (1 for positive, 0 for negative).
                - Additional keys from load_multimodal when a timeline is available.
        """
        user_path = self.users[idx]
        label = self.labels[idx]
        user = user_path.split("/")[-1]
        label_name = "positive" if label == 1 else "negative"

        # Load text and images (similar to dataloader 1)
        if not os.path.isfile(f"{user_path}/{TEXT_FILE_NAME}"):
            print(f"user {user} does not have timeline")
            image_paths = sorted(
                glob.glob(f"{DATA_PATH}/{label_name}/{user}/*.jpg"))
            images = [
                Image.open(img_path) if os.path.isfile(img_path) else None
                for img_path in image_paths
            ]
            sample = {
                "author": user,
                "texts": [],
                "images_paths": image_paths,
                "images": images,
                "label": label,
            }
            return sample

        user_timeline = pd.read_json(
            f"{user_path}/{TEXT_FILE_NAME}", lines=True)
        images = []
        image_paths = []
        texts = user_timeline["text"].tolist()
        for id, row in user_timeline.iterrows():

            # 優先使用 timeline.txt 裡的 image_path
            image_rel_path = row.get("image_path", "")

            if pd.notna(image_rel_path) and str(image_rel_path).strip() != "":
                img_path = os.path.join(user_path, str(image_rel_path))
            else:
                # fallback：舊格式，避免沒有 image_path 的資料壞掉
                img_path = f"{user_path}/{row['id']}.jpg"

            if not os.path.isfile(img_path):
                img_path = None

            preprocessed_img, updated_path = preprocess_image(
                img_path,
                training=self.kind == "train",
            )
            images.append(preprocessed_img)
            image_paths.append(updated_path)

        dates = [
            int(round(date.timestamp()))
            for date in user_timeline["created_at"].tolist()
        ]

        if idx < 3:
            print("DEBUG user:", user)
            print("DEBUG first image_paths:", image_paths[:5])
            print("DEBUG blank count:", sum(
                [p == "Blank image" for p in image_paths]))

        sample = self.load_multimodal(
            images=images,
            texts=texts,
            label=label,
            dates=dates,
            user_name=user,
            images_paths=image_paths,
        )

        return sample
