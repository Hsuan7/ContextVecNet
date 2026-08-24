import pickle
import torch.nn as nn
import torch
import os
import csv
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from .acumen_trainer import AcumenTrainer
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

class Trainer(AcumenTrainer):
    def __init__(self, args, model, pos_weight=None):
        super().__init__()
        self.args = args

        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight], dtype=torch.float32).cuda()
            print("Using BCEWithLogitsLoss pos_weight =", float(pos_weight.item()))

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.model = model

    def configure_optimizers(self, lr=0.00001):
        self._optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )

        return self._optimizer

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = self.criterion(output["logits"], batch["label"])

        self.log("train/loss", loss.item(), on_step=True)

        return loss

    def validation_step(self, batch, i):
        output = self.model(batch)
        loss = self.criterion(output["logits"], batch["label"])
        loss = torch.mean(loss)

        self.log("val_loss_batch", loss.item(), on_step=True, force_log=False)

        return (
            loss.detach().cpu().item(),
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
        losses, logits, probas, labels = zip(*outputs)

        val_loss = float(np.mean(losses))
        logits = np.vstack(logits).astype(np.float32).reshape(-1)
        probas = np.vstack(probas).astype(np.float32).reshape(-1)
        labels = np.vstack(labels).astype(np.float32).reshape(-1)

        print("::: Validation loss", val_loss)
        self.log("val_loss", val_loss, on_step=False, force_log=True)

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
            pos_logit_mean = logits[pos_mask].mean()
        else:
            pos_logit_mean = float("nan")

        if neg_mask.sum() > 0:
            neg_logit_mean = logits[neg_mask].mean()
        else:
            neg_logit_mean = float("nan")

        logit_gap = pos_logit_mean - neg_logit_mean

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
        print("positive logit mean:", round(float(pos_logit_mean), 4)
              if not np.isnan(pos_logit_mean) else pos_logit_mean)
        print("negative logit mean:", round(float(neg_logit_mean), 4)
              if not np.isnan(neg_logit_mean) else neg_logit_mean)
        print("logit gap:", round(float(logit_gap), 4)
              if not np.isnan(logit_gap) else logit_gap)

        print("\n===== DEBUG SCORE DISTRIBUTION (Before Calibration) =====")
        print("out min:", round(float(probas.min()), 4))
        print("out max:", round(float(probas.max()), 4))
        print("out mean:", round(float(probas.mean()), 4))
        print("labels unique:", np.unique(labels, return_counts=True))
        print("sample out:", np.round(probas[:20], 4))
        print("sample labels:", labels[:20])

        # # =========================================================
        # # 3) Temperature Scaling
        # # =========================================================
        # print("\n===== Platt Scaling =====")

        # # logits: shape (N,)
        # # labels: shape (N,)
        # X_platt = logits.reshape(-1, 1)
        # y_platt = labels.astype(int)

        # # 防呆：validation 若只有單一類別，無法做 Platt scaling
        # if len(np.unique(y_platt)) < 2:
        #     print("[WARN] Only one class in validation labels. Skip Platt scaling.")
        #     calibrated_probas = probas.copy()
        #     platt_a = None
        #     platt_b = None
        # else:
        #     platt_model = LogisticRegression(
        #         solver="lbfgs",
        #         max_iter=1000,
        #         class_weight=None
        #     )
        #     platt_model.fit(X_platt, y_platt)

        #     calibrated_probas = platt_model.predict_proba(X_platt)[:, 1]

        #     platt_a = float(platt_model.coef_[0][0])
        #     platt_b = float(platt_model.intercept_[0])

        #     print(f"Learned Platt coef (a): {platt_a:.6f}")
        #     print(f"Learned Platt intercept (b): {platt_b:.6f}")

        # =========================================================
        # 3) Platt Scaling
        # =========================================================
        print("\n===== Platt Scaling =====")

        X_platt = logits.reshape(-1, 1)
        y_platt = labels.astype(int)

        if len(np.unique(y_platt)) < 2:
            print("[WARN] Only one class in validation labels. Skip Platt scaling.")
            calibrated_probas = probas.copy()
            platt_a = None
            platt_b = None
        else:
            platt_model = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                class_weight=None
            )
            platt_model.fit(X_platt, y_platt)

            calibrated_probas = platt_model.predict_proba(X_platt)[:, 1]

            platt_a = float(platt_model.coef_[0][0])
            platt_b = float(platt_model.intercept_[0])

            print(f"Learned Platt coef (a): {platt_a:.6f}")
            print(f"Learned Platt intercept (b): {platt_b:.6f}")

        # =========================================================
        # 4) Debug: calibration 後分布
        # =========================================================
        print("\n===== DEBUG CLASS MEAN (After Platt Scaling) =====")
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

        print("\n===== DEBUG SCORE DISTRIBUTION (After Platt Scaling) =====")
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

        # Store calibration plot data. The checkpoint callback writes the plots
        # only when this epoch becomes the best validation-loss checkpoint.
        epoch_idx = getattr(getattr(self, "trainer_ref", None), "epoch", None)
        prob_true, prob_pred = calibration_curve(
            labels, calibrated_probas, n_bins=10)
        self.latest_calibration_plot = {
            "epoch": epoch_idx,
            "prob_true": prob_true,
            "prob_pred": prob_pred,
            "calibrated_probas": calibrated_probas,
            "val_loss": val_loss,
            "val_f1": float(best_f1),
        }

        # =========================================================
        # 7) Log calibration metrics. Checkpointing now monitors epoch val_loss.
        # =========================================================
        # self.log("val/temperature", T_value, on_step=False, force_log=True)
        if platt_a is not None:
            self.log("val/platt_a", platt_a, on_step=False, force_log=True)
            self.log("val/platt_b", platt_b, on_step=False, force_log=True)
        self.log("val/best_threshold", best_thr,
                 on_step=False, force_log=True)
        self.log("val/best_accuracy", best_acc,
                 on_step=False, force_log=True)
        self.log("val/best_precision", best_prec,
                 on_step=False, force_log=True)
        self.log("val/best_recall", best_rec,
                 on_step=False, force_log=True)

        # Keep F1 for reporting; checkpointing monitors val_loss in main_maple.py.
        self.log("val_f1", best_f1, on_step=False, force_log=True)

        # =========================================================
        # 8) CSV 紀錄
        # =========================================================
        safe_group = str(self.args.group).replace(":", "_").replace("/", "_")
        safe_name = str(self.args.name).replace(":", "_").replace("/", "_")
        metrics_dir = os.path.join(
            "training_metrics",
            safe_group,
            safe_name,
        )
        csv_path = os.path.join(metrics_dir, "epoch_metrics.csv")
        file_exists = os.path.isfile(csv_path)

        try:
            os.makedirs(metrics_dir, exist_ok=True)
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow([
                        "epoch",
                        "val_loss",
                        "auc",
                        "platt_a",
                        "platt_b",
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
                        "positive_logit_mean",
                        "negative_logit_mean",
                        "logit_gap",
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
                    val_loss,
                    auc,
                    platt_a,
                    platt_b,
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
                    pos_logit_mean,
                    neg_logit_mean,
                    logit_gap,
                    pos_mean_cal,
                    neg_mean_cal,
                    float(probas.min()),
                    float(probas.max()),
                    float(probas.mean()),
                    float(calibrated_probas.min()),
                    float(calibrated_probas.max()),
                    float(calibrated_probas.mean()),
                ])
        except OSError as exc:
            print(f"[WARN] Could not write epoch metrics to {csv_path}: {exc}")

        print("DEBUG returning val metrics")
        return {
            "val_loss": val_loss,
            "val_f1": float(best_f1),
            "val_auc": float(auc),
            "val_best_threshold": float(best_thr),
            "val_best_accuracy": float(best_acc),
            "val_best_precision": float(best_prec),
            "val_best_recall": float(best_rec),
            "val_logit_gap": float(logit_gap),
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
