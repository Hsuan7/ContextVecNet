from .callback import Callback
import os
import torch
import re


class ModelCheckpoint(Callback):
    def __init__(
        self,
        monitor="val_loss",
        direction="down",
        dirpath="checkpoints/",
        save_weights_only=False,
        filename="checkpoint",
        save_best_only=True,
    ):
        self.trainer = None
        self.monitor = monitor
        self.direction = direction
        self.dirpath = dirpath
        self.save_weights_only = save_weights_only
        self.filename = filename
        self.save_best_only = save_best_only

        self.previous_best = None
        self.previous_best_path = None
        self.previous_plot_paths = []

    def _safe_name(self, value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

    def _save_best_calibration_plots(self, epoch, trainer_quantity):
        payload = getattr(self.trainer, "latest_calibration_plot", None)
        if not payload:
            return []

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args = self.trainer.args
        plot_dir = os.path.join(
            "calibration_plots",
            self._safe_name(args.group),
            self._safe_name(args.name),
        )
        os.makedirs(plot_dir, exist_ok=True)

        prefix = (
            f"{self._safe_name(args.group)}_{self._safe_name(args.name)}"
            f"_fold{args.fold}_epoch{epoch}_val_loss{trainer_quantity:.6f}"
        )

        calibration_curve_path = os.path.join(
            plot_dir, f"{prefix}_best_calibration_curve.png"
        )
        reliability_hist_path = os.path.join(
            plot_dir, f"{prefix}_best_reliability_hist.png"
        )

        plt.figure(figsize=(6, 6))
        plt.plot(payload["prob_pred"], payload["prob_true"], marker="o", label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Probability")
        plt.title(f"Calibration Curve (best val_loss, epoch={epoch})")
        plt.legend()
        plt.grid(True)
        plt.savefig(calibration_curve_path, dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(payload["calibrated_probas"], bins=20)
        plt.title(f"Reliability Histogram (best val_loss, epoch={epoch})")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.savefig(reliability_hist_path, dpi=200, bbox_inches="tight")
        plt.close()

        print("Saved best calibration curve to:", calibration_curve_path)
        print("Saved best reliability histogram to:", reliability_hist_path)
        return [calibration_curve_path, reliability_hist_path]

    def on_epoch_end(self):
        # trainer_quantity = self.trainer.logger.metrics[self.monitor]

        trainer_quantity = self.trainer.logger.metrics.get(self.monitor, None)
        if trainer_quantity is None:
            print(f"[WARN] monitor key '{self.monitor}' not found in logger.metrics, skipping checkpoint save this epoch.")
            return
        

        if self.previous_best is not None:
            if self.direction == "down":
                if self.previous_best <= trainer_quantity:
                    print(
                        f"No improvement. Current: {trainer_quantity} - Previous {self.previous_best}"
                    )
                    return
            else:
                if self.previous_best >= trainer_quantity:
                    print(
                        f"No improvement. Current: {trainer_quantity} - Previous {self.previous_best}"
                    )
                    return

        if self.previous_best_path is not None:
            os.unlink(self.previous_best_path)
        for plot_path in self.previous_plot_paths:
            if os.path.exists(plot_path):
                os.unlink(plot_path)
        self.previous_plot_paths = []

        path = os.path.join(
            self.dirpath,
            self.filename.format(
                **{"epoch": self.trainer.epoch, self.monitor: trainer_quantity}
            ),
        )

        print(f"🔥 Saving model to: {path}")

        os.makedirs(self.dirpath, exist_ok=True)

        self.previous_best = trainer_quantity
        self.previous_best_path = path

        checkpoint = {
            "model_state_dict": self.trainer.model_hook.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "epoch": self.trainer.epoch,
            "metrics": self.trainer.logger.metrics,
        }

        if self.save_weights_only:
            torch.save(checkpoint["model_state_dict"], path)
        else:
            torch.save(checkpoint, path)

        self.previous_plot_paths = self._save_best_calibration_plots(
            self.trainer.epoch, trainer_quantity
        )

    def load_checkpoint(self, checkpoint_path):
        print(f"🔄 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        self.trainer.model_hook.load_state_dict(checkpoint["model_state_dict"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.trainer.epoch = checkpoint["epoch"] + 1
        self.trainer.logger.metrics = checkpoint.get("metrics", {})
