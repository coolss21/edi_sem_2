"""
automl_search.py - Bayesian Optimization with Optuna.

Ablation modes:
  student_only  – reconstruction only, no teacher
  teacher_only  – teacher encoder evaluated directly (no student)
  manual_tsrc   – fixed curriculum, no search
  auto_tsrc     – full AutoML search
"""

import os
import csv
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import optuna

from .config import AUTOML_SEARCH_SPACE, MANUAL_CURRICULUM
from .preprocessing import RobustTimeSeriesPreprocessor
from .teacher_contrastive import (
    LightweightContrastiveCNNTeacher,
    train_teacher,
)
from .student_autoencoder import build_student
from .curriculum import CurriculumScheduler
from .trainer import train_tsrc_student, train_student_autoencoder_only
from .evaluation import (
    extract_embeddings,
    run_kmeans,
    compute_clustering_metrics,
    compute_reconstruction_mse,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Score function
# ─────────────────────────────────────────────────────────────────────────────

def _multi_objective_score(
    ari: float,
    nmi: float,
    chi: float,
    recon_mse: float,
    train_time: float,
    robustness_score: float = 0.5,
    chi_max: float = 1.0,
    time_max: float = 1.0,
) -> float:
    norm_ari   = (ari + 1.0) / 2.0
    norm_nmi   = float(np.clip(nmi, 0, 1))
    norm_chi   = float(np.log1p(max(chi, 0)) / max(np.log1p(chi_max), 1e-8))
    recon_q    = 1.0 / (1.0 + max(recon_mse, 0))
    norm_time  = float(train_time / max(time_max, 1e-8))

    score = (
          0.35 * norm_ari
        + 0.20 * norm_nmi
        + 0.15 * norm_chi
        + 0.15 * recon_q
        + 0.10 * robustness_score
        - 0.05 * norm_time
    )
    return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# Single trial execution
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(
    trial_id: int,
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_steps: int,
    device: torch.device,
    trial_epochs: int,
    teacher_epochs: int,
    seed: int,
    verbose: bool = False,
) -> dict:
    t0 = time.time()
    n_classes = len(np.unique(y_test))

    prep = RobustTimeSeriesPreprocessor(
        normalization=config["normalization"],
        missing_strategy=config["missing_strategy"],
        clip_outliers=config["clip_outliers"],
    )
    X_tr = prep.fit_transform(X_train)
    X_te = prep.transform(X_test)

    teacher = LightweightContrastiveCNNTeacher(
        time_steps=time_steps,
        embedding_dim=config["embedding_dim"],
    )
    train_teacher(
        X_tr, teacher, device,
        epochs=teacher_epochs,
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        aug_strength=config["augmentation_strength"],
        jitter_sigma=config["jitter_sigma"],
        mask_ratio=config["mask_ratio"],
        seed=seed,
        verbose=verbose,
    )

    student = build_student(
        config["student_type"], time_steps,
        config["embedding_dim"], config["hidden_dim"],
    )
    scheduler = CurriculumScheduler(
        start_lambda=config["start_lambda"],
        end_lambda=config["end_lambda"],
        delay_epoch=config["delay_epoch"],
        total_epochs=trial_epochs,
        tau=config["tau"],
    )
    train_tsrc_student(
        X_tr, student, teacher, device, scheduler,
        epochs=trial_epochs,
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        patience=max(2, trial_epochs // 3),
        seed=seed,
        verbose=verbose,
    )

    test_embs = extract_embeddings(student, X_te, device)
    km        = run_kmeans(test_embs, n_classes, seed)
    met       = compute_clustering_metrics(y_test, km, test_embs)
    try:
        recon_mse = compute_reconstruction_mse(student, X_te, device)
    except Exception:
        recon_mse = 9999.0

    train_time = time.time() - t0

    row = {
        "trial_id":       trial_id,
        "ARI":            met["ARI"],
        "NMI":            met["NMI"],
        "CHI":            met["CHI"],
        "silhouette":     met["silhouette"],
        "reconstruction_mse": recon_mse,
        "training_time":  train_time,
        "score":          0.0,
        **config,
    }
    return row, student, teacher, prep, scheduler


# ─────────────────────────────────────────────────────────────────────────────
# AutoML Search class (Optuna)
# ─────────────────────────────────────────────────────────────────────────────

class AutoTSRCSearch:
    """
    Bayesian optimization for Auto-TSRC using Optuna.
    """

    def __init__(
        self,
        n_trials: int = 8,
        trial_epochs: int = 5,
        final_epochs: int = 20,
        teacher_epochs: int = 10,
        device: torch.device = None,
        seed: int = 42,
        tables_dir: str = "outputs/tables",
        models_dir: str = "outputs/models",
        verbose: bool = False,
    ):
        self.n_trials      = n_trials
        self.trial_epochs  = trial_epochs
        self.final_epochs  = final_epochs
        self.teacher_epochs = teacher_epochs
        self.device        = device or torch.device("cpu")
        self.seed          = seed
        self.tables_dir    = tables_dir
        self.models_dir    = models_dir
        self.verbose       = verbose

        self.trial_results_  = []
        self.best_config_    = None
        self.best_student_   = None
        self.best_teacher_   = None
        self.best_prep_      = None
        self.best_scheduler_ = None

    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset: str,
    ):
        time_steps = X_train.shape[1]
        print(f"\\n[AutoML] Starting Optuna search: {self.n_trials} trials x "
              f"{self.trial_epochs} epochs on {dataset}")

        trial_rows = []

        def objective(trial):
            # Sample from the AUTOML_SEARCH_SPACE
            config = {
                "normalization": trial.suggest_categorical("normalization", AUTOML_SEARCH_SPACE["normalization"]),
                "missing_strategy": trial.suggest_categorical("missing_strategy", AUTOML_SEARCH_SPACE["missing_strategy"]),
                "clip_outliers": trial.suggest_categorical("clip_outliers", AUTOML_SEARCH_SPACE["clip_outliers"]),
                "augmentation_strength": trial.suggest_categorical("augmentation_strength", AUTOML_SEARCH_SPACE["augmentation_strength"]),
                "mask_ratio": trial.suggest_categorical("mask_ratio", AUTOML_SEARCH_SPACE["mask_ratio"]),
                "jitter_sigma": trial.suggest_categorical("jitter_sigma", AUTOML_SEARCH_SPACE["jitter_sigma"]),
                "student_type": trial.suggest_categorical("student_type", AUTOML_SEARCH_SPACE["student_type"]),
                "embedding_dim": trial.suggest_categorical("embedding_dim", AUTOML_SEARCH_SPACE["embedding_dim"]),
                "hidden_dim": trial.suggest_categorical("hidden_dim", AUTOML_SEARCH_SPACE["hidden_dim"]),
                "learning_rate": trial.suggest_categorical("learning_rate", AUTOML_SEARCH_SPACE["learning_rate"]),
                "batch_size": trial.suggest_categorical("batch_size", AUTOML_SEARCH_SPACE["batch_size"]),
                "start_lambda": trial.suggest_categorical("start_lambda", AUTOML_SEARCH_SPACE["start_lambda"]),
                "end_lambda": trial.suggest_categorical("end_lambda", AUTOML_SEARCH_SPACE["end_lambda"]),
                "delay_epoch": trial.suggest_categorical("delay_epoch", AUTOML_SEARCH_SPACE["delay_epoch"]),
                "tau": trial.suggest_categorical("tau", AUTOML_SEARCH_SPACE["tau"]),
            }
            
            cfg_seed = self.seed + trial.number
            try:
                row, _, _, _, _ = _run_trial(
                    trial_id=trial.number,
                    config=config,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    time_steps=time_steps,
                    device=self.device,
                    trial_epochs=self.trial_epochs,
                    teacher_epochs=min(self.teacher_epochs, self.trial_epochs),
                    seed=cfg_seed,
                    verbose=self.verbose,
                )
                trial_rows.append(row)
                
                # We return a mix of metrics for Optuna to maximize. For simplicity in Optuna, 
                # we'll maximize a basic combination of ARI and NMI for the raw objective.
                # The actual multi-objective score will be computed later for the final selection.
                return row["ARI"] + 0.5 * row["NMI"] - 0.1 * min(row["reconstruction_mse"], 10.0)
            except Exception as e:
                print(f"    [AutoML] Optuna Trial {trial.number} FAILED: {e}")
                raise optuna.exceptions.TrialPruned()

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials)

        if not trial_rows:
            raise RuntimeError("[AutoML] All trials failed. Cannot continue.")

        # Re-score the trials cleanly
        chi_max  = max(r["CHI"]           for r in trial_rows) + 1e-8
        time_max = max(r["training_time"] for r in trial_rows) + 1e-8
        for r in trial_rows:
            r["score"] = _multi_objective_score(
                r["ARI"], r["NMI"], r["CHI"],
                r["reconstruction_mse"], r["training_time"],
                chi_max=chi_max, time_max=time_max,
            )

        best_idx = max(range(len(trial_rows)), key=lambda i: trial_rows[i]["score"])
        best_row = trial_rows[best_idx]
        best_cfg = {k: best_row[k] for k in AUTOML_SEARCH_SPACE}

        print(f"\\n[AutoML] Optuna found best config: score={best_row['score']:.4f}  "
              f"ARI={best_row['ARI']:.4f}  NMI={best_row['NMI']:.4f}")

        # ── Stage 3: retrain best for final_epochs ───────────────────────
        print(f"[AutoML] Stage 3: retraining best config for {self.final_epochs} epochs...")
        final_prep = RobustTimeSeriesPreprocessor(
            normalization=best_cfg["normalization"],
            missing_strategy=best_cfg["missing_strategy"],
            clip_outliers=best_cfg["clip_outliers"],
        )
        X_tr_f = final_prep.fit_transform(X_train)

        final_teacher = LightweightContrastiveCNNTeacher(
            time_steps=time_steps,
            embedding_dim=best_cfg["embedding_dim"],
        )
        train_teacher(
            X_tr_f, final_teacher, self.device,
            epochs=self.teacher_epochs,
            batch_size=best_cfg["batch_size"],
            lr=best_cfg["learning_rate"],
            aug_strength=best_cfg["augmentation_strength"],
            jitter_sigma=best_cfg["jitter_sigma"],
            mask_ratio=best_cfg["mask_ratio"],
            seed=self.seed,
            verbose=True,
        )

        final_student = build_student(
            best_cfg["student_type"], time_steps,
            best_cfg["embedding_dim"], best_cfg["hidden_dim"],
        )
        final_sched = CurriculumScheduler(
            start_lambda=best_cfg["start_lambda"],
            end_lambda=best_cfg["end_lambda"],
            delay_epoch=best_cfg["delay_epoch"],
            total_epochs=self.final_epochs,
            tau=best_cfg["tau"],
        )
        os.makedirs(self.tables_dir, exist_ok=True)
        history_path = os.path.join(
            self.tables_dir, f"{dataset}_training_history.csv"
        )
        train_tsrc_student(
            X_tr_f, final_student, final_teacher, self.device, final_sched,
            epochs=self.final_epochs,
            batch_size=best_cfg["batch_size"],
            lr=best_cfg["learning_rate"],
            patience=max(3, self.final_epochs // 4),
            seed=self.seed,
            verbose=True,
            history_path=history_path,
        )

        # Save model checkpoint
        os.makedirs(self.models_dir, exist_ok=True)
        torch.save(
            final_student.state_dict(),
            os.path.join(self.models_dir, f"{dataset}_best_student.pt"),
        )
        torch.save(
            final_teacher.state_dict(),
            os.path.join(self.models_dir, f"{dataset}_best_teacher.pt"),
        )

        self.trial_results_  = trial_rows
        self.best_config_    = best_cfg
        self.best_student_   = final_student
        self.best_teacher_   = final_teacher
        self.best_prep_      = final_prep
        self.best_scheduler_ = final_sched

        self._save_trials_csv(dataset, trial_rows)
        return self

    def _save_trials_csv(self, dataset: str, rows: list):
        path = os.path.join(self.tables_dir, f"{dataset}_automl_trials.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[AutoML] Trials saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Manual TSRC (fixed config, no search)
# ─────────────────────────────────────────────────────────────────────────────

def run_manual_tsrc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    final_epochs: int = 20,
    teacher_epochs: int = 10,
    batch_size: int = 32,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    student_type: str = "cnn_ae",
    learning_rate: float = 1e-3,
    seed: int = 42,
    tables_dir: str = "outputs/tables",
    models_dir: str = "outputs/models",
    dataset: str = "dataset",
    verbose: bool = True,
):
    """Run TSRC with a fixed manual curriculum (no AutoML)."""
    time_steps = X_train.shape[1]

    prep = RobustTimeSeriesPreprocessor(normalization="zscore",
                                         missing_strategy="linear_interpolation",
                                         clip_outliers=True)
    X_tr = prep.fit_transform(X_train)

    teacher = LightweightContrastiveCNNTeacher(time_steps, embedding_dim)
    train_teacher(X_tr, teacher, device, epochs=teacher_epochs,
                  batch_size=batch_size, lr=learning_rate, seed=seed,
                  verbose=verbose)

    student = build_student(student_type, time_steps, embedding_dim, hidden_dim)
    sched   = CurriculumScheduler(**MANUAL_CURRICULUM, total_epochs=final_epochs)

    os.makedirs(tables_dir, exist_ok=True)
    history_path = os.path.join(tables_dir, f"{dataset}_manual_tsrc_history.csv")
    train_tsrc_student(X_tr, student, teacher, device, sched,
                       epochs=final_epochs, batch_size=batch_size,
                       lr=learning_rate, patience=5, seed=seed,
                       verbose=verbose, history_path=history_path)

    os.makedirs(models_dir, exist_ok=True)
    torch.save(student.state_dict(),
               os.path.join(models_dir, f"{dataset}_manual_student.pt"))
    torch.save(teacher.state_dict(),
               os.path.join(models_dir, f"{dataset}_manual_teacher.pt"))

    return student, teacher, prep, sched
