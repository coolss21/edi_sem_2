"""
main.py - Auto-TSRC command-line entry point.

Usage examples:
  python main.py --download-data --datasets ECG200 GunPoint Coffee --trials 8 --trial-epochs 5 --final-epochs 20
  python main.py --download-data --datasets ECG200 --trials 3 --trial-epochs 2 --final-epochs 5 --quick
"""

import os
import sys
import csv
import time
import argparse
import json
import numpy as np
import torch

# Ensure stdout can handle all characters on Windows (cp1252 would crash on
# Unicode box-drawing chars and arrows).  Reconfigure to UTF-8 when possible.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Make src importable whether run from project root or elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    ensure_output_dirs, DEFAULT_DATASETS,
    TABLES_DIR, PLOTS_DIR, REPORTS_DIR, MODELS_DIR, EMBEDDINGS_DIR,
)
from src.utils import set_seed, get_device, print_banner, print_section, save_json
from src.data_downloader import download_ucr_archive, verify_dataset
from src.data_loader import load_ucr_dataset
from src.preprocessing import RobustTimeSeriesPreprocessor
from src.teacher_contrastive import LightweightContrastiveCNNTeacher, train_teacher
from src.student_autoencoder import build_student
from src.curriculum import CurriculumScheduler, plot_curriculum_schedule
from src.trainer import train_tsrc_student, train_student_autoencoder_only
from src.evaluation import (
    extract_embeddings, run_kmeans,
    compute_clustering_metrics, compute_reconstruction_mse,
    svm_probe, evaluate_model,
)
from src.automl_search import AutoTSRCSearch, run_manual_tsrc
from src.robustness import evaluate_robustness, save_robustness_csv
from src.explainability import run_all_explainability
from src.report_generator import generate_research_report


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-TSRC: AutoML-Guided Robust Curriculum Distillation"
    )
    parser.add_argument("--download-data", action="store_true",
                        help="Download UCR Archive if not already present.")
    parser.add_argument("--datasets", nargs="+", default=["ECG200"],
                        help="UCR dataset name(s) to process.")
    parser.add_argument("--data-dir", default="data/ucr",
                        help="Directory to store UCR data.")
    parser.add_argument("--outputs-dir", default="outputs",
                        help="Root directory for all outputs.")
    parser.add_argument("--trials", type=int, default=8,
                        help="Number of AutoML trials.")
    parser.add_argument("--trial-epochs", type=int, default=5,
                        help="Epochs per AutoML trial.")
    parser.add_argument("--final-epochs", type=int, default=20,
                        help="Epochs for final best-config training.")
    parser.add_argument("--teacher-epochs", type=int, default=10,
                        help="Epochs for teacher pre-training.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (optional).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: reduce epochs for fast CPU testing.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(
    dataset: str,
    args,
    device: torch.device,
    tables_dir: str,
    plots_dir: str,
    reports_dir: str,
    models_dir: str,
    embeddings_dir: str,
) -> dict:
    """Run full Auto-TSRC pipeline for one dataset."""
    print_section(f"Processing dataset: {dataset}")

    # ── Load data ───────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = load_ucr_dataset(dataset, args.data_dir)
    time_steps = X_train.shape[1]
    n_classes  = len(np.unique(y_test))

    # Quick mode overrides
    teacher_epochs = 3  if args.quick else args.teacher_epochs
    final_epochs   = args.final_epochs
    trial_epochs   = args.trial_epochs
    n_trials       = args.trials
    batch_size_ovr = args.batch_size

    # ── Resolve output paths ─────────────────────────────────────────────
    def t_path(name): return os.path.join(tables_dir, f"{dataset}_{name}.csv")
    def p_path(name): return os.path.join(plots_dir,  f"{dataset}_{name}.png")
    def r_path(name): return os.path.join(reports_dir, f"{dataset}_{name}")

    # ── Shared default preprocessor (for baselines) ──────────────────────
    def make_default_prep():
        return RobustTimeSeriesPreprocessor(
            normalization="zscore",
            missing_strategy="linear_interpolation",
            clip_outliers=True,
        )

    comparison_rows = []
    t_total_start   = time.time()

    # ═══════════════════════════════════════════════════════════════════════
    # BASELINE 1 – Student Autoencoder Only
    # ═══════════════════════════════════════════════════════════════════════
    print_section("Baseline 1: Student Autoencoder Only")
    t0 = time.time()
    prep_so = make_default_prep()
    X_tr_so = prep_so.fit_transform(X_train)
    X_te_so = prep_so.transform(X_test)

    student_only = build_student("cnn_ae", time_steps, 64, 64)
    bs = batch_size_ovr or 32
    train_student_autoencoder_only(
        X_tr_so, student_only, device,
        epochs=final_epochs, batch_size=bs,
        lr=1e-3, patience=5, seed=args.seed, verbose=True,
    )
    embs_so = extract_embeddings(student_only, X_te_so, device)
    km_so   = run_kmeans(embs_so, n_classes, args.seed)
    met_so  = compute_clustering_metrics(y_test, km_so, embs_so)
    mse_so  = compute_reconstruction_mse(student_only, X_te_so, device)
    svm_so  = svm_probe(
        extract_embeddings(student_only, X_tr_so, device), y_train,
        embs_so, y_test,
    )
    comparison_rows.append({
        "dataset": dataset, "model_name": "Student Autoencoder Only",
        **met_so, "reconstruction_mse": mse_so,
        "svm_accuracy": svm_so, "training_time_seconds": time.time() - t0,
    })
    print(f"  ARI={met_so['ARI']:.4f}  NMI={met_so['NMI']:.4f}  MSE={mse_so:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # BASELINE 2 – Teacher Contrastive Only
    # ═══════════════════════════════════════════════════════════════════════
    print_section("Baseline 2: Teacher Contrastive Only")
    t0 = time.time()
    prep_to = make_default_prep()
    X_tr_to = prep_to.fit_transform(X_train)
    X_te_to = prep_to.transform(X_test)

    teacher_only = LightweightContrastiveCNNTeacher(time_steps, 64)
    train_teacher(
        X_tr_to, teacher_only, device,
        epochs=teacher_epochs, batch_size=bs, lr=1e-3, seed=args.seed, verbose=True,
    )
    embs_to = extract_embeddings(teacher_only, X_te_to, device)
    km_to   = run_kmeans(embs_to, n_classes, args.seed)
    met_to  = compute_clustering_metrics(y_test, km_to, embs_to)
    svm_to  = svm_probe(
        extract_embeddings(teacher_only, X_tr_to, device), y_train,
        embs_to, y_test,
    )
    comparison_rows.append({
        "dataset": dataset, "model_name": "Teacher Contrastive Only",
        **met_to, "reconstruction_mse": None,
        "svm_accuracy": svm_to, "training_time_seconds": time.time() - t0,
    })
    print(f"  ARI={met_to['ARI']:.4f}  NMI={met_to['NMI']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # BASELINE 3 – Manual TSRC
    # ═══════════════════════════════════════════════════════════════════════
    print_section("Baseline 3: Manual TSRC")
    t0 = time.time()
    manual_student, manual_teacher, manual_prep, manual_sched = run_manual_tsrc(
        X_train, y_train, X_test, y_test,
        device=device,
        final_epochs=final_epochs,
        teacher_epochs=teacher_epochs,
        batch_size=bs,
        embedding_dim=64, hidden_dim=64,
        student_type="cnn_ae",
        learning_rate=1e-3,
        seed=args.seed,
        tables_dir=tables_dir,
        models_dir=models_dir,
        dataset=dataset,
        verbose=True,
    )
    X_te_man = manual_prep.transform(X_test)
    X_tr_man = manual_prep.transform(X_train)
    embs_man = extract_embeddings(manual_student, X_te_man, device)
    km_man   = run_kmeans(embs_man, n_classes, args.seed)
    met_man  = compute_clustering_metrics(y_test, km_man, embs_man)
    mse_man  = compute_reconstruction_mse(manual_student, X_te_man, device)
    svm_man  = svm_probe(
        extract_embeddings(manual_student, X_tr_man, device), y_train,
        embs_man, y_test,
    )
    comparison_rows.append({
        "dataset": dataset, "model_name": "Manual TSRC",
        **met_man, "reconstruction_mse": mse_man,
        "svm_accuracy": svm_man, "training_time_seconds": time.time() - t0,
    })
    print(f"  ARI={met_man['ARI']:.4f}  NMI={met_man['NMI']:.4f}  MSE={mse_man:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # AUTO-TSRC – AutoML Search
    # ═══════════════════════════════════════════════════════════════════════
    print_section("Auto-TSRC: AutoML Search")
    t0 = time.time()
    searcher = AutoTSRCSearch(
        n_trials=n_trials,
        trial_epochs=trial_epochs,
        final_epochs=final_epochs,
        teacher_epochs=teacher_epochs,
        device=device,
        seed=args.seed,
        tables_dir=tables_dir,
        models_dir=models_dir,
        verbose=False,
    )
    searcher.search(X_train, y_train, X_test, y_test, dataset=dataset)

    best_student  = searcher.best_student_
    best_teacher  = searcher.best_teacher_
    best_prep     = searcher.best_prep_
    best_sched    = searcher.best_scheduler_
    best_config   = searcher.best_config_

    X_te_auto = best_prep.transform(X_test)
    X_tr_auto = best_prep.transform(X_train)
    embs_auto = extract_embeddings(best_student, X_te_auto, device)
    km_auto   = run_kmeans(embs_auto, n_classes, args.seed)
    met_auto  = compute_clustering_metrics(y_test, km_auto, embs_auto)
    mse_auto  = compute_reconstruction_mse(best_student, X_te_auto, device)
    svm_auto  = svm_probe(
        extract_embeddings(best_student, X_tr_auto, device), y_train,
        embs_auto, y_test,
    )
    comparison_rows.append({
        "dataset": dataset, "model_name": "Auto-TSRC",
        **met_auto, "reconstruction_mse": mse_auto,
        "svm_accuracy": svm_auto, "training_time_seconds": time.time() - t0,
    })
    print(f"  ARI={met_auto['ARI']:.4f}  NMI={met_auto['NMI']:.4f}  MSE={mse_auto:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Save comparison CSV
    # ═══════════════════════════════════════════════════════════════════════
    _save_comparison_csv(comparison_rows, t_path("comparison_metrics"))

    # ── Save best params ─────────────────────────────────────────────────
    best_params = {
        "dataset":        dataset,
        "automl_config":  best_config,
        "preprocessing":  best_prep.get_params(),
        "curriculum":     best_sched.to_dict(),
        "final_epochs":   final_epochs,
        "teacher_epochs": teacher_epochs,
    }
    bp_path = os.path.join(reports_dir, f"{dataset}_best_params.json")
    save_json(best_params, bp_path)
    print(f"[main] Best params saved -> {bp_path}")

    # ── Save embeddings ──────────────────────────────────────────────────
    np.save(os.path.join(embeddings_dir, f"{dataset}_auto_tsrc_embeddings.npy"),
            embs_auto)

    # ═══════════════════════════════════════════════════════════════════════
    # ROBUSTNESS EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    print_section("Robustness Evaluation")
    robust_rows = []

    for model_name, model, prep_obj in [
        ("Manual TSRC",  manual_student, manual_prep),
        ("Auto-TSRC",    best_student,   best_prep),
    ]:
        rows = evaluate_robustness(
            model=model,
            model_name=model_name,
            X_test=X_test,
            y_test=y_test,
            preprocessor=prep_obj,
            device=device,
            dataset=dataset,
            seed=args.seed,
        )
        robust_rows.extend(rows)
        for r in rows:
            print(f"  {model_name:20s}  {r['corruption_type']:15s}  "
                  f"ARI={r['ARI']:.4f}  drop={r['ari_drop_from_clean']:.4f}")

    save_robustness_csv(robust_rows, t_path("robustness_metrics"))

    # ═══════════════════════════════════════════════════════════════════════
    # EXPLAINABILITY PLOTS
    # ═══════════════════════════════════════════════════════════════════════
    print_section("Generating Explainability Plots")
    run_all_explainability(
        student=best_student,
        X_test=X_te_auto,
        y_test=y_test,
        embeddings=embs_auto,
        cluster_labels=km_auto,
        device=device,
        dataset=dataset,
        plots_dir=plots_dir,
    )

    # ── Curriculum schedule plot ─────────────────────────────────────────
    plot_curriculum_schedule(best_sched, dataset, plots_dir, label="Auto-TSRC Best")

    # ═══════════════════════════════════════════════════════════════════════
    # RESEARCH REPORT
    # ═══════════════════════════════════════════════════════════════════════
    print_section("Generating Research Report")
    generate_research_report(
        dataset=dataset,
        best_params=best_params,
        comparison_rows=comparison_rows,
        robustness_rows=robust_rows,
        plots_dir=plots_dir,
        reports_dir=reports_dir,
        automl_trials=searcher.trial_results_,
        training_time=time.time() - t_total_start,
    )

    return {
        "dataset":        dataset,
        "comparison":     comparison_rows,
        "best_config":    best_config,
        "robustness_rows": robust_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_comparison_csv(rows: list, path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["dataset", "model_name", "ARI", "NMI", "CHI", "silhouette",
            "reconstruction_mse", "svm_accuracy", "training_time_seconds"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[main] Comparison metrics saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(all_results: list):
    print_banner("AUTO-TSRC -- FINAL SUMMARY")
    for res in all_results:
        ds = res["dataset"]
        print(f"\n  Dataset: {ds}")
        print(f"  {'Model':<28}  {'ARI':>8}  {'NMI':>8}  {'SVM':>8}")
        print(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*8}")
        for row in res["comparison"]:
            ari = f"{row['ARI']:.4f}" if row.get("ARI") is not None else "  N/A  "
            nmi = f"{row['NMI']:.4f}" if row.get("NMI") is not None else "  N/A  "
            svm = f"{row['svm_accuracy']:.4f}" if row.get("svm_accuracy") else "  N/A  "
            print(f"  {row['model_name']:<28}  {ari:>8}  {nmi:>8}  {svm:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    print_banner(
        "Auto-TSRC: AutoML-Guided Robust Curriculum Distillation\n"
        "  for Interpretable Time-Series Representation Learning"
    )
    print(f"  Device : {device}")
    print(f"  Seed   : {args.seed}")
    print(f"  Trials : {args.trials} x {args.trial_epochs} epochs -> final {args.final_epochs}")
    print(f"  Datasets: {args.datasets}")

    # Resolve output dirs
    out_root    = args.outputs_dir
    tables_dir  = os.path.join(out_root, "tables")
    plots_dir   = os.path.join(out_root, "plots")
    reports_dir = os.path.join(out_root, "reports")
    models_dir  = os.path.join(out_root, "models")
    emb_dir     = os.path.join(out_root, "embeddings")
    for d in [tables_dir, plots_dir, reports_dir, models_dir, emb_dir]:
        os.makedirs(d, exist_ok=True)

    # Download data
    if args.download_data:
        print_section("Downloading UCR Archive")
        ok = download_ucr_archive(args.data_dir)
        if not ok:
            print("[main] WARNING: Data download failed. "
                  "Continuing -- will error if dataset files missing.")

    # Process each dataset
    all_results = []
    for dataset in args.datasets:
        if not verify_dataset(dataset, args.data_dir):
            print(f"[main] WARNING: {dataset} not found - skipping.")
            continue
        try:
            res = run_dataset(
                dataset=dataset,
                args=args,
                device=device,
                tables_dir=tables_dir,
                plots_dir=plots_dir,
                reports_dir=reports_dir,
                models_dir=models_dir,
                embeddings_dir=emb_dir,
            )
            all_results.append(res)
        except Exception as e:
            import traceback
            print(f"\n[main] ERROR processing {dataset}: {e}")
            traceback.print_exc()

    _print_summary(all_results)
    print_banner("Auto-TSRC run complete. Check outputs/ directory.")


if __name__ == "__main__":
    main()
