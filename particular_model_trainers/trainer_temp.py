import pickle
import torch.nn as nn
import torch
import os
import csv
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from .acumen_trainer import AcumenTrainer
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


class Trainer(AcumenTrainer):
    def __init__(self, args, model, class_weights=None):
        super().__init__()
        self.args = args

        self.weights = torch.Tensor(
            np.array([1.0, 1.0]).astype(np.float32)).cuda()
        if class_weights is not None:
            self.weights = torch.from_numpy(
                class_weights.astype(np.float32)).cuda()

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.model = model

    def configure_optimizers(self, lr=0.00001):
        self._optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )

        return self._optimizer

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss_non_reduced = self.criterion(output["logits"], batch["label"])

        if self.weights is None:
            loss = torch.mean(loss_non_reduced)
        else:
            batch_weights = torch.cat(
                [1 - batch["label"], batch["label"]], dim=-1)
            batch_weights = batch_weights * self.weights
            batch_weights = batch_weights.sum(dim=-1)
            loss = batch_weights.view(-1) * loss_non_reduced.view(-1)
            loss = torch.mean(loss)

        self.log("train/loss", loss.item(), on_step=True)

        return loss

    def validation_step(self, batch, i):
        output = self.model(batch)
        loss = self.criterion(output["logits"], batch["label"])
        loss = torch.mean(loss)

        self.log("val_loss", loss.item(), on_step=True, force_log=True)

        return (
            output["logits"].detach().cpu().numpy(),
            output["probas"].detach().cpu().numpy(),
            batch["label"].detach().cpu().numpy(),
        )
        print("=== ENTER validation_epoch_end WITH CALIBRATION ===")

    def validation_epoch_end(self, outputs):
        # =========================================================
        # 0) 解析 outputs
        # validation_step 現在需回傳:
        # (logits, probas, labels)
        # =========================================================
        logits, probas, labels = zip(*outputs)

        logits = np.vstack(logits).astype(np.float32).reshape(-1)
        probas = np.vstack(probas).astype(np.float32).reshape(-1)
        labels = np.vstack(labels).astype(np.float32).reshape(-1)

        # =========================================================
        # 1) 先看原本未 calibration 的 default 0.5 結果
        # =========================================================
        default_pred = np.round(probas).astype(int)

        acc = metrics.accuracy_score(labels, default_pred)
        print("::: Accuracy", acc)
        self.log("val/accuracy", acc, on_step=False, force_log=True)

        fpr, tpr, _ = metrics.roc_curve(labels, probas, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("::: AUC", auc)
        self.log("val/auc", auc, on_step=False, force_log=True)

        precision = metrics.precision_score(
            labels, default_pred, zero_division=0)
        self.log("val/precision", precision, on_step=False, force_log=True)
        print("::: Precision", precision)

        recall = metrics.recall_score(
            labels, default_pred, zero_division=0)
        self.log("val/recall", recall, on_step=False, force_log=True)
        print("::: Recall", recall)

        f1 = metrics.f1_score(labels, default_pred, zero_division=0)
        print("::: F1", f1)

        # =========================================================
        # 2) Debug: 原始 probas 分布
        # =========================================================
        print("\n===== DEBUG CLASS MEAN (Before Calibration) =====")
        pos_mask = labels == 1
        neg_mask = labels == 0

        if pos_mask.sum() > 0:
            pos_mean = probas[pos_mask].mean()
        else:
            pos_mean = float("nan")

        if neg_mask.sum() > 0:
            neg_mean = probas[neg_mask].mean()
        else:
            neg_mean = float("nan")

        print("positive mean:", round(float(pos_mean), 4)
              if not np.isnan(pos_mean) else pos_mean)
        print("negative mean:", round(float(neg_mean), 4)
              if not np.isnan(neg_mean) else neg_mean)

        print("\n===== DEBUG SCORE DISTRIBUTION (Before Calibration) =====")
        print("out min:", round(float(probas.min()), 4))
        print("out max:", round(float(probas.max()), 4))
        print("out mean:", round(float(probas.mean()), 4))
        print("labels unique:", np.unique(labels, return_counts=True))
        print("sample out:", np.round(probas[:20], 4))
        print("sample labels:", labels[:20])

        # =========================================================
        # 3) Temperature Scaling
        # =========================================================
        print("\n===== Temperature Scaling =====")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
        labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

        # 單一溫度參數
        temperature = torch.nn.Parameter(torch.ones(1, device=device))

        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

        def _eval():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                logits_t / temperature, labels_t)
            loss.backward()
            return loss

        optimizer.step(_eval)

        T_value = float(temperature.item())
        print(f"Learned Temperature: {T_value:.6f}")

        calibrated_probas = torch.sigmoid(
            logits_t / temperature).detach().cpu().numpy()

        # =========================================================
        # 4) Debug: calibration 後分布
        # =========================================================
        print("\n===== DEBUG CLASS MEAN (After Calibration) =====")
        if pos_mask.sum() > 0:
            pos_mean_cal = calibrated_probas[pos_mask].mean()
        else:
            pos_mean_cal = float("nan")

        if neg_mask.sum() > 0:
            neg_mean_cal = calibrated_probas[neg_mask].mean()
        else:
            neg_mean_cal = float("nan")
        print("positive mean:", round(float(pos_mean_cal), 4)
              if not np.isnan(pos_mean_cal) else pos_mean_cal)
        print("negative mean:", round(float(neg_mean_cal), 4)
              if not np.isnan(neg_mean_cal) else neg_mean_cal)

        print("\n===== DEBUG SCORE DISTRIBUTION (After Calibration) =====")
        print("out min:", round(float(calibrated_probas.min()), 4))
        print("out max:", round(float(calibrated_probas.max()), 4))
        print("out mean:", round(float(calibrated_probas.mean()), 4))
        print("sample out:", np.round(calibrated_probas[:20], 4))

        # =========================================================
        # 5) Threshold sweep：改用 calibrated probas
        # =========================================================
        print("\n===== Threshold Sweep (After Calibration) =====")

        thr_min = float(calibrated_probas.min())
        thr_max = float(calibrated_probas.max())

        if abs(thr_max - thr_min) < 1e-8:
            thresholds = np.array([thr_min], dtype=np.float32)
        else:
            thresholds = np.linspace(thr_min, thr_max, 21)

        print(">>> DEBUG threshold range:", thr_min, "~", thr_max)
        print(">>> DEBUG threshold list:", np.round(thresholds, 4).tolist())

        best_thr = None
        best_f1 = -1.0
        best_acc = None
        best_prec = None
        best_rec = None

        for thr in thresholds:
            y_pred = (calibrated_probas >= thr).astype(int)

            acc_thr = metrics.accuracy_score(labels, y_pred)
            prec_thr = metrics.precision_score(
                labels, y_pred, zero_division=0)
            rec_thr = metrics.recall_score(labels, y_pred, zero_division=0)
            f1_thr = metrics.f1_score(labels, y_pred, zero_division=0)

            print(
                f"thr={thr:.4f} | acc={acc_thr:.3f} | "
                f"prec={prec_thr:.3f} | rec={rec_thr:.3f} | f1={f1_thr:.3f}"
            )

            if f1_thr > best_f1:
                best_f1 = f1_thr
                best_thr = float(thr)
                best_acc = acc_thr
                best_prec = prec_thr
                best_rec = rec_thr

        print("\n>>> BEST THRESHOLD")
        print(
            f"best_thr={best_thr:.4f} | "
            f"acc={best_acc:.3f} | "
            f"prec={best_prec:.3f} | "
            f"rec={best_rec:.3f} | "
            f"f1={best_f1:.3f}"
        )

        # =========================================================
        # 6) 繪製 Calibration Curve / Reliability Diagram
        # =========================================================
        print("\n===== Calibration Plots =====")
        # epoch_idx = getattr(self, "epoch", None)
        epoch_idx = getattr(getattr(self, "trainer_ref", None), "epoch", None)
        plot_dir = "/home/angle/ContextVecNet/calibration_plots"
        os.makedirs(plot_dir, exist_ok=True)

        # calibration curve
        prob_true, prob_pred = calibration_curve(
            labels, calibrated_probas, n_bins=10)

        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--",
                 label="Perfect Calibration")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Probability")
        plt.title(f"Calibration Curve (epoch={epoch_idx})")
        plt.legend()
        plt.grid(True)
        calibration_curve_path = os.path.join(
            plot_dir, f"calibration_curve_epoch_{epoch_idx}.png")
        plt.savefig(calibration_curve_path, dpi=200, bbox_inches="tight")
        plt.close()

        # reliability histogram
        plt.figure(figsize=(6, 4))
        plt.hist(calibrated_probas, bins=20)
        plt.title(f"Reliability Histogram (epoch={epoch_idx})")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        reliability_hist_path = os.path.join(
            plot_dir, f"reliability_hist_epoch_{epoch_idx}.png")
        plt.savefig(reliability_hist_path, dpi=200, bbox_inches="tight")
        plt.close()

        print("Saved calibration curve to:", calibration_curve_path)
        print("Saved reliability histogram to:", reliability_hist_path)

        # =========================================================
        # 7) log：checkpoint 依據改成 calibration 後 best_f1
        # =========================================================
        self.log("val/temperature", T_value, on_step=False, force_log=True)
        self.log("val/best_threshold", best_thr,
                 on_step=False, force_log=True)
        self.log("val/best_accuracy", best_acc,
                 on_step=False, force_log=True)
        self.log("val/best_precision", best_prec,
                 on_step=False, force_log=True)
        self.log("val/best_recall", best_rec,
                 on_step=False, force_log=True)

        # checkpoint 監控這個
        self.log("val_f1", best_f1, on_step=False, force_log=True)

        # =========================================================
        # 8) CSV 紀錄
        # =========================================================
        csv_path = "/home/angle/ContextVecNet/epoch_threshold_results.csv"
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "epoch",
                    "auc",
                    "temperature",
                    "best_threshold",
                    "best_accuracy",
                    "best_precision",
                    "best_recall",
                    "best_f1",
                    "default_accuracy",
                    "default_precision",
                    "default_recall",
                    "default_f1",
                    "positive_mean_before",
                    "negative_mean_before",
                    "positive_mean_after",
                    "negative_mean_after",
                    "out_min_before",
                    "out_max_before",
                    "out_mean_before",
                    "out_min_after",
                    "out_max_after",
                    "out_mean_after",
                ])

            writer.writerow([
                epoch_idx,
                auc,
                T_value,
                best_thr,
                best_acc,
                best_prec,
                best_rec,
                best_f1,
                acc,
                precision,
                recall,
                f1,
                pos_mean,
                neg_mean,
                pos_mean_cal,
                neg_mean_cal,
                float(probas.min()),
                float(probas.max()),
                float(probas.mean()),
                float(calibrated_probas.min()),
                float(calibrated_probas.max()),
                float(calibrated_probas.mean()),
            ])

        print("DEBUG returning val metrics")
        return {
            "val_f1": float(best_f1),
            "val_auc": float(auc),
            "val_best_threshold": float(best_thr),
            "val_best_accuracy": float(best_acc),
            "val_best_precision": float(best_prec),
            "val_best_recall": float(best_rec),
        }
# 第二版 改整個validation_step和validation_epoch_end
    # def validation_step(self, batch, i):
    #     output = self.model(batch)
    #     loss = self.criterion(output["logits"], batch["label"])
    #     loss = torch.mean(loss)

    #     self.log("val_loss", loss.item(), on_step=True, force_log=True)

    #     return (
    #         output["probas"].detach().cpu().numpy(),
    #         batch["label"].detach().cpu().numpy(),
    #     )

    # def validation_epoch_end(self, outputs):

    #             out, labels = zip(*outputs)
    #     out = np.vstack(out).astype(np.float32)
    #     labels = np.vstack(labels).astype(np.float32)

    #     # flatten 成 1D，避免 metric shape 問題
    #     out_flat = out.flatten()
    #     labels_flat = labels.flatten()

    #     # =========================================================
    #     # 1) 先看原本 0.5 threshold 下的結果（保留原始輸出習慣）
    #     # =========================================================
    #     default_pred = np.round(out_flat)

    #     acc = metrics.accuracy_score(labels_flat, default_pred)
    #     print("::: Accuracy", acc)
    #     self.log("val/accuracy", acc, on_step=False, force_log=True)

    #     fpr, tpr, _ = metrics.roc_curve(labels_flat, out_flat, pos_label=1)
    #     auc = metrics.auc(fpr, tpr)
    #     print("::: AUC", auc)
    #     self.log("val/auc", auc, on_step=False, force_log=True)

    #     precision = metrics.precision_score(
    #         labels_flat, default_pred, zero_division=0)
    #     self.log("val/precision", precision, on_step=False, force_log=True)
    #     print("::: Precision", precision)

    #     recall = metrics.recall_score(
    #         labels_flat, default_pred, zero_division=0)
    #     self.log("val/recall", recall, on_step=False, force_log=True)
    #     print("::: Recall", recall)

    #     f1 = metrics.f1_score(labels_flat, default_pred, zero_division=0)
    #     print("::: F1", f1)

    #     # =========================================================
    #     # 2) Debug: class mean / score distribution
    #     # =========================================================
    #     print("\n===== DEBUG CLASS MEAN =====")
    #     pos_mask = labels_flat == 1
    #     neg_mask = labels_flat == 0

    #     if pos_mask.sum() > 0:
    #         pos_mean = out_flat[pos_mask].mean()
    #     else:
    #         pos_mean = float("nan")

    #     if neg_mask.sum() > 0:
    #         neg_mean = out_flat[neg_mask].mean()
    #     else:
    #         neg_mean = float("nan")

    #     print("positive mean:", round(float(pos_mean), 4)
    #           if not np.isnan(pos_mean) else pos_mean)
    #     print("negative mean:", round(float(neg_mean), 4)
    #           if not np.isnan(neg_mean) else neg_mean)

    #     print("\n===== DEBUG SCORE DISTRIBUTION =====")
    #     print("out min:", round(float(out_flat.min()), 4))
    #     print("out max:", round(float(out_flat.max()), 4))
    #     print("out mean:", round(float(out_flat.mean()), 4))
    #     print("labels unique:", np.unique(labels_flat, return_counts=True))
    #     print("sample out:", np.round(out_flat[:20], 4))
    #     print("sample labels:", labels_flat[:20])

    #     # =========================================================
    #     #    3) Threshold sweep：根據當前 epoch 的分數範圍自動掃描
    #     # =========================================================
    #     print("\n===== Threshold Sweep =====")

    #     thr_min = float(out_flat.min())
    #     thr_max = float(out_flat.max())

    #     # 若所有分數幾乎一樣，避免 linspace 壞掉
    #     if abs(thr_max - thr_min) < 1e-8:
    #         thresholds = np.array([thr_min], dtype=np.float32)
    #     else:
    #         thresholds = np.linspace(thr_min, thr_max, 21)

    #     print(">>> DEBUG threshold range:", thr_min, "~", thr_max)
    #     print(">>> DEBUG threshold list:", np.round(thresholds, 4).tolist())

    #     best_thr = None
    #     best_f1 = -1.0
    #     best_acc = None
    #     best_prec = None
    #     best_rec = None

    #     for thr in thresholds:
    #         y_pred = (out_flat >= thr).astype(int)

    #         acc_thr = metrics.accuracy_score(labels_flat, y_pred)
    #         prec_thr = metrics.precision_score(
    #             labels_flat, y_pred, zero_division=0)
    #         rec_thr = metrics.recall_score(
    #             labels_flat, y_pred, zero_division=0)
    #         f1_thr = metrics.f1_score(labels_flat, y_pred, zero_division=0)

    #         print(
    #             f"thr={thr:.4f} | acc={acc_thr:.3f} | "
    #             f"prec={prec_thr:.3f} | rec={rec_thr:.3f} | f1={f1_thr:.3f}"
    #         )

    #         if f1_thr > best_f1:
    #             best_f1 = f1_thr
    #             best_thr = float(thr)
    #             best_acc = acc_thr
    #             best_prec = prec_thr
    #             best_rec = rec_thr

    #     print("\n>>> BEST THRESHOLD")
    #     print(
    #         f"best_thr={best_thr:.4f} | "
    #         f"acc={best_acc:.3f} | "
    #         f"prec={best_prec:.3f} | "
    #         f"rec={best_rec:.3f} | "
    #         f"f1={best_f1:.3f}"
    #     )

    #     # =========================================================
    #     # 4) 記錄：改用 best threshold 下的 F1 當 checkpoint 依據
    #     # =========================================================
    #     self.log("val/best_threshold", best_thr, on_step=False, force_log=True)
    #     self.log("val/best_accuracy", best_acc, on_step=False, force_log=True)
    #     self.log("val/best_precision", best_prec,
    #              on_step=False, force_log=True)
    #     self.log("val/best_recall", best_rec, on_step=False, force_log=True)

    #     # 這一行最重要：讓 checkpoint 監控的 val_f1 改成最佳 threshold 下的 F1
    #     self.log("val_f1", best_f1, on_step=False, force_log=True)

    #     csv_path = "/home/angle/ContextVecNet/epoch_threshold_results.csv"
    #     file_exists = os.path.isfile(csv_path)
    #     # 改 window 改成下面的
    #     # epoch_idx = getattr(self.trainer, "epoch", None)
    #     epoch_idx = getattr(self, "epoch", None)
    #     with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
    #         writer = csv.writer(f)

    #         if not file_exists:
    #             writer.writerow([
    #                 "epoch",
    #                 "auc",
    #                 "best_threshold",
    #                 "best_accuracy",
    #                 "best_precision",
    #                 "best_recall",
    #                 "best_f1",
    #                 "default_accuracy",
    #                 "default_precision",
    #                 "default_recall",
    #                 "default_f1",
    #                 "positive_mean",
    #                 "negative_mean",
    #                 "out_min",
    #                 "out_max",
    #                 "out_mean",
    #             ])

    #         writer.writerow([
    #             epoch_idx,
    #             auc,
    #             best_thr,
    #             best_acc,
    #             best_prec,
    #             best_rec,
    #             best_f1,
    #             acc,
    #             precision,
    #             recall,
    #             f1,
    #             pos_mean,
    #             neg_mean,
    #             float(out_flat.min()),
    #             float(out_flat.max()),
    #             float(out_flat.mean()),
    #         ])
# 第一版
#     def validation_epoch_end(self, outputs):
#         out, labels = zip(*outputs)
#         out = np.vstack(out)
#         labels = np.vstack(labels)
# ###
#         # ===== 新增這段 =====
#         print("\n===== DEBUG CLASS MEAN =====")

#         #  flatten 成 1D（避免 shape 問題）
#         out_flat = out.flatten()
#         labels_flat = labels.flatten()

#         pos_mean = out_flat[labels_flat == 1].mean()
#         neg_mean = out_flat[labels_flat == 0].mean()

#         print("positive mean:", pos_mean)
#         print("negative mean:", neg_mean)

#         acc = metrics.accuracy_score(labels, np.round(out))
#         print("::: Accuracy", acc)
#         self.log("val/accuracy", acc, on_step=False, force_log=True)

#         fpr, tpr, thresholds = metrics.roc_curve(labels, out, pos_label=1)
#         auc = metrics.auc(fpr, tpr)
#         print("::: AUC", auc)
#         self.log("val/auc", auc, on_step=False, force_log=True)

#         precision = metrics.precision_score(labels, np.round(out))
#         self.log("val/precision", precision, on_step=False, force_log=True)
#         print("::: Precision", precision)

#         recall = metrics.recall_score(labels, np.round(out))
#         self.log("val/recall", recall, on_step=False, force_log=True)
#         print("::: Recall", recall)

#         f1 = metrics.f1_score(labels, np.round(out))
#         self.log("val_f1", f1, on_step=False, force_log=True)
#         print("::: F1", f1)

#         print("\n===== DEBUG SCORE DISTRIBUTION =====")
#         print("out min:", out.min())
#         print("out max:", out.max())
#         print("out mean:", out.mean())

#         print("labels unique:", np.unique(labels, return_counts=True))

#         print("sample out:", out[:20].flatten())
#         print("sample labels:", labels[:20].flatten())

#         print("\n===== Threshold Sweep =====")

#         thr_min = float(out.min())
#         thr_max = float(out.max())
#         thresholds = np.linspace(thr_min, thr_max, 21)

#         print(">>> DEBUG threshold range:", thr_min, "~", thr_max)
#         print(">>> DEBUG threshold list:", np.round(thresholds, 4).tolist())

#         best_thr = None
#         best_f1 = -1
#         best_metrics = None

#         for thr in thresholds:
#             y_pred = (out >= thr).astype(int)

#             acc_thr = metrics.accuracy_score(labels, y_pred)
#             prec_thr = metrics.precision_score(labels, y_pred, zero_division=0)
#             rec_thr = metrics.recall_score(labels, y_pred, zero_division=0)
#             f1_thr = metrics.f1_score(labels, y_pred, zero_division=0)

#             print(
#                 f"thr={thr:.4f} | acc={acc_thr:.3f} | "
#                 f"prec={prec_thr:.3f} | rec={rec_thr:.3f} | f1={f1_thr:.3f}"
#             )

#             if f1_thr > best_f1:
#                 best_f1 = f1_thr
#                 best_thr = thr
#                 best_metrics = (acc_thr, prec_thr, rec_thr, f1_thr)

#         print("\n>>> BEST THRESHOLD")
#         print(
#             f"best_thr={best_thr:.4f} | "
#             f"acc={best_metrics[0]:.3f} | "
#             f"prec={best_metrics[1]:.3f} | "
#             f"rec={best_metrics[2]:.3f} | "
#             f"f1={best_metrics[3]:.3f}"
#         )
