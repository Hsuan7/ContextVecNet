import torch
from scipy import stats
import json
import numpy as np
import os

import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn import metrics
from evaluators import BaseEvaluator

import nomenclature

from datasets.twitter_learn import CombinedTwitterDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def custom_collate_fn(batch):
    """
    Custom collate function for handling variable size images in the dataset.
    Returns a batch where images are kept as a list.
    """
    # Extract items from the batch
    user_names = [item["author"] for item in batch]
    labels = [item["label"] for item in batch]
    texts = [item["texts"] for item in batch]
    dates = [item["time"] for item in batch]
    images = [item["images"] for item in batch]
    images_paths = [item["images_paths"] for item in batch]
    padding_amount = [item["padding_amount"] for item in batch]
    text_mask = [item["text_mask"] for item in batch]
    image_mask = [item["image_mask"] for item in batch]

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float32).clone().detach()
    text_mask_tensor = torch.tensor(text_mask, dtype=torch.float32).clone().detach()
    image_mask_tensor = torch.tensor(image_mask, dtype=torch.float32).clone().detach()
    dates_tensor = torch.tensor(dates, dtype=torch.float32).clone().detach()

    # Create a sample dictionary
    sample = {
        "author": user_names,
        "images": images,  # Keep images as a list of variable sizes
        "images_paths": images_paths,
        "texts": texts,
        "time": dates_tensor,
        "label": labels_tensor.view(-1, 1),
        "padding_amount": padding_amount,
        "text_mask": text_mask_tensor,
        "image_mask": image_mask_tensor,
    }

    return sample


class MultimodalEvaluator(BaseEvaluator):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.num_runs = 1
        self.dataset = CombinedTwitterDataset
        self.test_dataset = self.dataset(args=args, kind="test")
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=False,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        self.eval_calibration = getattr(args, "eval_calibration", {}) or {}
        self.platt_a = self.eval_calibration.get("platt_a")
        self.platt_b = self.eval_calibration.get("platt_b")
        self.threshold = self.eval_calibration.get("threshold")

        if self.threshold is None:
            self.threshold = 0.5
            print("[WARN] Validation best threshold not found; using 0.5 for test.")
        else:
            self.threshold = float(self.threshold)

        if self.platt_a is None or self.platt_b is None:
            self.platt_a = None
            self.platt_b = None
            print("[WARN] Validation Platt parameters not found; using raw model probabilities for test.")
        else:
            self.platt_a = float(self.platt_a)
            self.platt_b = float(self.platt_b)

    def _apply_validation_calibration(self, logits, raw_probas):
        if self.platt_a is None or self.platt_b is None:
            return raw_probas

        z = np.clip(self.platt_a * logits + self.platt_b, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def trainer_evaluate(self, step):
        print("Running Evaluation.")
        results = self.evaluate(save=False)
        return results[-1]["f1"]

    def evaluate(self, save=True):
        y_preds = []
        y_preds_proba = []
        true_labels = []

        for _ in range(self.num_runs):
            y_pred = []
            y_pred_proba = []
            true_label = []

            with torch.no_grad(), torch.autocast(
                device_type=nomenclature.device.type, dtype=torch.float32
            ):
                for i, batch in enumerate(
                    tqdm(self.test_dataloader, total=len(self.test_dataloader))
                ):
                    output = self.model(batch)

                    raw_preds = np.vstack(output["probas"].detach().cpu().numpy()).ravel()
                    logits = np.vstack(output["logits"].detach().cpu().numpy()).ravel()
                    preds = self._apply_validation_calibration(logits, raw_preds)
                    labels = np.vstack(batch["label"].detach().cpu().numpy()).ravel()

                    y_pred.extend((preds >= self.threshold).astype(int))
                    y_pred_proba.extend(preds)
                    true_label.extend(labels)

            y_preds.append(y_pred)
            y_preds_proba.append(y_pred_proba)
            true_labels.append(true_label)

        y_preds = np.array(y_preds)
        y_preds_proba = np.array(y_preds_proba)
        true_labels = np.array(true_labels)

        # Flatten arrays
        y_preds = y_preds.flatten()
        y_preds_proba = y_preds_proba.flatten()
        true_labels = true_labels.flatten()

        # Compute metrics
        accuracy = metrics.accuracy_score(true_labels, y_preds)
        precision = metrics.precision_score(true_labels, y_preds, average="binary")
        recall = metrics.recall_score(true_labels, y_preds, average="binary")
        f1 = metrics.f1_score(true_labels, y_preds, average="binary")
        auc = metrics.roc_auc_score(true_labels, y_preds_proba)

        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Threshold: {self.threshold:.6f}")
        print(
            "Calibration:",
            "platt" if self.platt_a is not None and self.platt_b is not None else "raw",
        )

        results = pd.DataFrame.from_dict(
            {
                "f1": [f1],
                "recall": [recall],
                "precision": [precision],
                "auc": [auc],
                "accuracy": [accuracy],
                "name": [f"{self.args.group}:{self.args.name}"],
                "dataset": [self.args.dataset],
                "text_embedding": [self.args.text_embeddings_type],
                "image_embedding": [self.args.image_embeddings_type],
                "window_size": [self.args.window_size],
                "position_embedding": [self.args.position_embeddings],
                "fold": [self.args.fold],
                "modality": [self.args.modality],
                "threshold": [self.threshold],
                "platt_a": [self.platt_a],
                "platt_b": [self.platt_b],
            }
        )

        if save:
            os.makedirs(f"results/{self.args.output_dir}", exist_ok=True)
            results.to_csv(
                f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}.csv",
                index=False,
            )

        return results
