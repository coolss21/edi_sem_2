"""
curriculum.py - Curriculum scheduler for lambda (hint loss weight) over epochs.

lambda_train(epoch) =
    0                                                           if epoch < delay_epoch
    start_lambda + (end_lambda - start_lambda) * progress^tau  otherwise

where progress = (epoch - delay_epoch) / max(1, total_epochs - delay_epoch)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CurriculumScheduler:
    """
    Controls lambda (weight of hint/distillation loss) over training epochs.

    Parameters
    ----------
    start_lambda : float  – lambda at the first active epoch
    end_lambda   : float  – lambda at the last epoch
    delay_epoch  : int    – number of warm-up epochs where lambda = 0
    total_epochs : int    – total training epochs
    tau          : float  – shape exponent
                   tau = 1.0  → linear
                   tau > 1.0  → slow start (convex)
                   tau < 1.0  → fast start (concave)
    """

    def __init__(
        self,
        start_lambda: float = 0.0,
        end_lambda: float   = 0.9,
        delay_epoch: int    = 0,
        total_epochs: int   = 20,
        tau: float          = 1.0,
    ):
        self.start_lambda = start_lambda
        self.end_lambda   = end_lambda
        self.delay_epoch  = delay_epoch
        self.total_epochs = total_epochs
        self.tau          = tau

    def get_lambda(self, epoch: int) -> float:
        """
        Return lambda for the given epoch (0-indexed).
        """
        if epoch < self.delay_epoch:
            return 0.0

        denom    = max(1, self.total_epochs - self.delay_epoch)
        progress = (epoch - self.delay_epoch) / denom
        progress = float(np.clip(progress, 0.0, 1.0))
        lam = self.start_lambda + (self.end_lambda - self.start_lambda) * (progress ** self.tau)
        return float(np.clip(lam, 0.0, 1.0))

    def schedule(self) -> list:
        """Return list of lambda values for all epochs."""
        return [self.get_lambda(e) for e in range(self.total_epochs)]

    def to_dict(self) -> dict:
        return {
            "start_lambda": self.start_lambda,
            "end_lambda":   self.end_lambda,
            "delay_epoch":  self.delay_epoch,
            "total_epochs": self.total_epochs,
            "tau":          self.tau,
        }


def plot_curriculum_schedule(
    scheduler: CurriculumScheduler,
    dataset: str,
    plots_dir: str,
    label: str = "Best AutoML",
):
    """
    Plot lambda value vs epoch and save to plots_dir.
    """
    os.makedirs(plots_dir, exist_ok=True)
    lambdas = scheduler.schedule()
    epochs  = list(range(len(lambdas)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lambdas, color="#4A90D9", linewidth=2.5, label=label)
    ax.axhline(y=scheduler.end_lambda, color="#E84393", linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"end_lambda={scheduler.end_lambda}")
    if scheduler.delay_epoch > 0:
        ax.axvline(x=scheduler.delay_epoch, color="#F5A623", linestyle=":",
                   linewidth=1.5, label=f"delay_epoch={scheduler.delay_epoch}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Lambda (hint loss weight)", fontsize=12)
    ax.set_title(
        f"Curriculum Schedule - {dataset}\n"
        f"tau={scheduler.tau}, delay={scheduler.delay_epoch}, "
        f"[{scheduler.start_lambda}->{scheduler.end_lambda}]",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(plots_dir, f"{dataset}_curriculum_schedule.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[curriculum] Saved schedule plot -> {save_path}")
    return save_path
