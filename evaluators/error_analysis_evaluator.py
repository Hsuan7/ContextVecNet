import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn import metrics

import nomenclature
from evaluators import BaseEvaluator
from datasets.twitter_learn import CombinedTwitterDataset


def custom_collate_fn(batch):
    user_names = [item["author"] for item in batch]
    labels = [item["label"] for item in batch]
    texts = [item["texts"] for item in batch]
    dates = [item["time"] for item in batch]
    images = [item["images"] for item in batch]
    images_paths = [item["images_paths"] for item in batch]
    padding_amount = [item["padding_amount"] for item in batch]
    text_mask = [item["text_mask"] for item in batch]
    image_mask = [item["image_mask"] for item in batch]

    labels_tensor = torch.tensor(labels, dtype=torch.float32).clone().detach()
    text_mask_tensor = torch.tensor(text_mask, dtype=torch.float32).clone().detach()
    image_mask_tensor = torch.tensor(image_mask, dtype=torch.float32).clone().detach()
    dates_tensor = torch.tensor(dates, dtype=torch.float32).clone().detach()

    sample = {
        "author": user_names,
        "images": images,
        "images_paths": images_paths,
        "texts": texts,
        "time": dates_tensor,
        "label": labels_tensor.view(-1, 1),
        "padding_amount": padding_amount,
        "text_mask": text_mask_tensor,
        "image_mask": image_mask_tensor,
    }

    return sample


def get_error_type(y_true, y_pred):
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 0:
        return "FN"


class ErrorAnalysisEvaluator(BaseEvaluator):
    def __init__(self, args, model, kind="test"):
        super().__init__(args, model)

        self.dataset = CombinedTwitterDataset
        self.eval_dataset = self.dataset(args=args, kind=kind)

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=False,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        self.kind = kind

    def evaluate_and_save_error_analysis(
        self,
        threshold=0.3667688318495692,
        high_conf_threshold=0.8,
        output_file=None
    ):
        self.model.eval()

        rows = []

        with torch.no_grad(), torch.autocast(
            device_type=nomenclature.device.type,
            dtype=torch.float32
        ):
            for batch_idx, batch in enumerate(
                tqdm(self.eval_dataloader, total=len(self.eval_dataloader))
            ):
                output = self.model(batch)["probas"]

                probs = np.vstack(output.detach().cpu().numpy()).ravel()
                labels = np.vstack(batch["label"].detach().cpu().numpy()).ravel()

                preds = (probs >= threshold).astype(int)
                confidences = np.maximum(probs, 1 - probs)

                authors = batch["author"]
                padding_amount = batch["padding_amount"]

                text_mask = batch["text_mask"].detach().cpu().numpy()
                image_mask = batch["image_mask"].detach().cpu().numpy()

                text_valid_count = text_mask.sum(axis=1)
                image_valid_count = image_mask.sum(axis=1)

                for i in range(len(labels)):
                    y_true_i = int(labels[i])
                    y_prob_i = float(probs[i])
                    y_pred_i = int(preds[i])
                    confidence_i = float(confidences[i])
                    correct_i = y_true_i == y_pred_i
                    error_type_i = get_error_type(y_true_i, y_pred_i)

                    rows.append({
                        "user_id": authors[i],
                        "y_true": y_true_i,
                        "y_prob": y_prob_i,
                        "y_pred": y_pred_i,
                        "confidence": confidence_i,
                        "correct": correct_i,
                        "error_type": error_type_i,
                        "high_conf_error": (not correct_i) and (confidence_i >= high_conf_threshold),
                        "text_valid_count": float(text_valid_count[i]),
                        "image_valid_count": float(image_valid_count[i]),
                        "padding_amount": padding_amount[i],
                        "threshold": threshold,
                        "high_conf_threshold": high_conf_threshold,
                        "possible_source": "",
                        "observable_log": ""
                    })

        df_error = pd.DataFrame(rows)

        accuracy = metrics.accuracy_score(df_error["y_true"], df_error["y_pred"])
        precision = metrics.precision_score(df_error["y_true"], df_error["y_pred"], zero_division=0)
        recall = metrics.recall_score(df_error["y_true"], df_error["y_pred"], zero_division=0)
        f1 = metrics.f1_score(df_error["y_true"], df_error["y_pred"], zero_division=0)
        auc = metrics.roc_auc_score(df_error["y_true"], df_error["y_prob"])

        print("\n===== Performance =====")
        print(f"Data kind : {self.kind}")
        print(f"Threshold : {threshold}")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"AUC       : {auc:.4f}")

        print("\n===== Error Type Count =====")
        print(df_error["error_type"].value_counts())

        high_conf_errors = df_error[df_error["high_conf_error"]].sort_values(
            "confidence",
            ascending=False
        )

        print("\n===== High-confidence Errors =====")
        print(high_conf_errors.head(10))

        if output_file is None:
            output_dir = f"results/{self.args.output_dir}"
            os.makedirs(output_dir, exist_ok=True)
            safe_group = str(self.args.group).replace(":", "_")
            safe_name = str(self.args.name).replace(":", "_")
            output_file = f"{output_dir}/{safe_group}_{safe_name}_{self.kind}_error_analysis2.csv"

        df_error.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nSaved error analysis file to: {output_file}")

        metrics_summary = pd.DataFrame([{
            "kind": self.kind,
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "n_samples": len(df_error),
            "n_high_conf_errors": len(high_conf_errors),
            "model_name": f"{self.args.group}:{self.args.name}",
            "dataset": self.args.dataset,
            "fold": self.args.fold,
            "modality": self.args.modality,
            "window_size": self.args.window_size,
            "text_embedding": self.args.text_embeddings_type,
            "image_embedding": self.args.image_embeddings_type,
            "position_embedding": self.args.position_embeddings,
        }])

        metrics_file = output_file.replace(".csv", "_metrics_summary.csv")
        metrics_summary.to_csv(metrics_file, index=False, encoding="utf-8-sig")
        print(f"Saved metrics summary file to: {metrics_file}")

        return df_error, metrics_summary