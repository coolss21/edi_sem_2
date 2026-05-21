"""
report_generator.py - Generate Markdown research report for Auto-TSRC.
"""

import os
import json
import datetime


def generate_research_report(
    dataset: str,
    best_params: dict,
    comparison_rows: list,
    robustness_rows: list,
    plots_dir: str,
    reports_dir: str,
    automl_trials: list = None,
    training_time: float = None,
) -> str:
    """
    Generate a Markdown research report and save it to reports_dir.
    Returns path to saved report.
    """
    os.makedirs(reports_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Helper: format comparison table ─────────────────────────────────
    def _comparison_table(rows):
        if not rows:
            return "_No results available._\n"
        headers = ["Model", "ARI", "NMI", "CHI", "Silhouette", "Recon MSE", "SVM Acc", "Time (s)"]
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join(["---"] * len(headers)) + "|"]
        for r in rows:
            def fmt(v, decimals=4):
                if v is None:
                    return "—"
                try:
                    return f"{float(v):.{decimals}f}"
                except Exception:
                    return str(v)
            lines.append(
                f"| {r.get('model_name','?')} "
                f"| {fmt(r.get('ARI'))} "
                f"| {fmt(r.get('NMI'))} "
                f"| {fmt(r.get('CHI'),1)} "
                f"| {fmt(r.get('silhouette'))} "
                f"| {fmt(r.get('reconstruction_mse'))} "
                f"| {fmt(r.get('svm_accuracy'))} "
                f"| {fmt(r.get('training_time_seconds'),1)} |"
            )
        return "\n".join(lines) + "\n"

    # ── Helper: robustness table ─────────────────────────────────────────
    def _robustness_table(rows):
        if not rows:
            return "_No robustness results available._\n"
        headers = ["Model", "Corruption", "ARI", "NMI", "CHI", "Recon MSE", "ARI Drop"]
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join(["---"] * len(headers)) + "|"]
        for r in rows:
            def fmt(v, d=4):
                if v is None: return "—"
                try: return f"{float(v):.{d}f}"
                except: return str(v)
            lines.append(
                f"| {r.get('model','?')} "
                f"| {r.get('corruption_type','?')} "
                f"| {fmt(r.get('ARI'))} "
                f"| {fmt(r.get('NMI'))} "
                f"| {fmt(r.get('CHI'),1)} "
                f"| {fmt(r.get('reconstruction_mse'))} "
                f"| {fmt(r.get('ari_drop_from_clean'))} |"
            )
        return "\n".join(lines) + "\n"

    # ── Best params block ────────────────────────────────────────────────
    params_block = "```json\n" + json.dumps(best_params, indent=2, default=str) + "\n```\n"

    # ── Auto-TSRC row for results ────────────────────────────────────────
    auto_row = next(
        (r for r in comparison_rows if "Auto-TSRC" in str(r.get("model_name", ""))),
        None,
    )
    abstract_ari = f"{auto_row['ARI']:.4f}" if auto_row and auto_row.get('ARI') is not None else "N/A"
    abstract_nmi = f"{auto_row['NMI']:.4f}" if auto_row and auto_row.get('NMI') is not None else "N/A"

    report = f"""# Auto-TSRC: AutoML-Guided Robust Curriculum Distillation for Interpretable Time-Series Representation Learning

> **Prototype Research Report** | Dataset: `{dataset}` | Generated: {now}

---

## 1. Abstract

We present **Auto-TSRC**, a prototype extension of the Time-Series Representation via
Curriculum Distillation (TSRC) framework. TSRC combines a contrastive teacher model with a
reconstruction-based student model to produce interpretable time-series embeddings. In this
prototype, we introduce three new research contributions:

1. **AutoML-guided search** over preprocessing strategies, student architectures,
   curriculum schedules, augmentation policies, and training hyperparameters.
2. **Robust preprocessing** that explicitly handles missing values, spike anomalies, outliers,
   and different normalization strategies.
3. **Standardised explainability outputs** including reconstruction overlays, error heatmaps,
   PCA scatter plots, and cluster prototype visualisations.

On the `{dataset}` benchmark, the best Auto-TSRC configuration achieves
ARI = **{abstract_ari}** and NMI = **{abstract_nmi}** compared to a fixed-curriculum Manual TSRC baseline.

---

## 2. Problem Statement

Time-series representation learning is a foundational task in many domains including healthcare,
finance, and industrial monitoring. Obtaining labelled data is expensive, motivating
self-supervised approaches. However, most self-supervised methods:

- Require manual hyperparameter tuning.
- Fail gracefully under real-world data quality issues (missing values, noise, spikes).
- Produce "black-box" embeddings with limited interpretability.

---

## 3. Gap in Original TSRC

The original TSRC paper defines a compelling curriculum distillation framework but leaves open:

- **No automated hyperparameter selection** — the curriculum schedule, student architecture,
  and preprocessing strategy must be chosen manually.
- **No robustness analysis** — performance under corrupted inputs is not characterised.
- **No standardised explainability** — visualisations such as reconstruction overlays and
  PCA plots are not produced systematically.

---

## 4. Proposed Contributions

| Contribution | Description |
|---|---|
| AutoML Search | Budgeted random search with successive halving over 16 hyperparameters |
| Robust Preprocessing | Missing value imputation, outlier clipping, z-score / robust normalisation |
| Robustness Testing | 6 corruption types: missing-10%, missing-20%, spike, gaussian×2, burst |
| Explainability Outputs | Reconstruction overlay, error heatmap, PCA scatter, cluster prototypes |
| Multi-objective Scoring | ARI/NMI/CHI/reconstruction quality/robustness/time composite |

---

## 5. Dataset Used

**Dataset:** `{dataset}` (UCR Time Series Archive 2018)

| Property | Value |
|---|---|
| Source | UCR Time Series Archive 2018 |
| Format | TSV, first column = label |
| Task | Time-series classification / clustering |

---

## 6. Methodology

### Teacher Model

The teacher is a **LightweightContrastiveCNNTeacher** — a compact CNN encoder trained with
SimCLR-style NT-Xent contrastive loss on two augmented views of each time series.

> **Note:** This teacher is *inspired by* TS2Vec-style contrastive representation learning
> but is **not** a full reproduction of TS2Vec. It uses a two-layer CNN with adaptive pooling
> rather than hierarchical temporal contrasting.

### Student Models

Two autoencoder architectures are available:

- **CNNAutoencoderStudent** — CNN encoder + interpolation decoder.
- **LSTMAutoencoderStudent** — LSTM encoder + LSTM decoder.

The student is trained on:

```
Total Loss = (1 - λ) × ReconLoss + λ × HintLoss
```

where `ReconLoss = MSE(original, reconstructed)` and `HintLoss = MSE(student_emb, teacher_emb)`.

### Curriculum

`λ(epoch)` follows a power-law schedule:

```
λ(epoch) = start_λ + (end_λ - start_λ) × progress^τ
```

where `progress = (epoch - delay_epoch) / (total_epochs - delay_epoch)`.

---

## 7. AutoML Search Space

| Hyperparameter | Options |
|---|---|
| normalization | zscore, robust |
| missing_strategy | linear_interpolation, forward_fill, mean_fill |
| clip_outliers | True, False |
| augmentation_strength | weak, medium, strong |
| mask_ratio | 0.05, 0.10, 0.20 |
| jitter_sigma | 0.01, 0.03, 0.05 |
| student_type | cnn_ae, lstm_ae |
| embedding_dim | 32, 64, 128 |
| hidden_dim | 32, 64, 128 |
| learning_rate | 1e-3, 5e-4, 1e-4 |
| batch_size | 16, 32, 64 |
| start_lambda | 0.0, 0.1, 0.2, 0.25 |
| end_lambda | 0.5, 0.7, 0.9, 1.0 |
| delay_epoch | 0, 3, 5 |
| tau | 0.5, 1.0, 1.5, 2.0 |

---

## 8. Robust Preprocessing Module

`RobustTimeSeriesPreprocessor` handles:

1. **Missing values** — linear interpolation → forward fill → mean fill fallback.
2. **Entire-NaN samples** — replaced with zeros with a warning.
3. **Outlier clipping** — winsorize using training-set quantiles.
4. **Normalisation** — z-score (mean/std) or robust (median/IQR).

All parameters are fitted on training data only, preventing data leakage.

---

## 9. Teacher-Student Distillation Loss

```
Total Loss = (1 - λ) × MSE(x, x̂) + λ × MSE(z_student, z_teacher)
```

- `λ = 0` at the start → pure reconstruction learning.
- `λ → end_lambda` at the end → strong knowledge distillation.
- Teacher is **frozen** during student training.

---

## 10. Curriculum Scheduler

```
λ(epoch) = 0                                                if epoch < delay_epoch
           start_λ + (end_λ - start_λ) × progress^τ        otherwise
```

| τ value | Shape |
|---|---|
| < 1.0 | Fast start (concave) — hint loss increases quickly |
| = 1.0 | Linear schedule |
| > 1.0 | Slow start (convex) — reconstruction dominates early |

---

## 11. Experimental Setup

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Device | CPU (CUDA if available) |
| Seed | 42 |
| Validation split | 15% of training data |
| Early stopping patience | Configurable (default 5) |
| AutoML scoring | Multi-objective composite |

---

## 12. Results Table

{_comparison_table(comparison_rows)}

> **Rows:**
> - *Student Autoencoder Only* — reconstruction MSE only, no teacher.
> - *Teacher Contrastive Only* — teacher encoder evaluated directly.
> - *Manual TSRC* — fixed curriculum (start=0.0, end=0.9, delay=0, τ=1.0).
> - *Auto-TSRC* — best AutoML configuration after successive halving.

---

## 13. Robustness Evaluation

{_robustness_table(robustness_rows)}

> ARI Drop from Clean measures degradation from clean test performance.
> Lower drop = more robust model.

---

## 14. Explainability Outputs

The following plots are generated automatically:

| Plot | Description |
|---|---|
| Reconstruction Overlay | Original vs. reconstructed for 3 test samples |
| Error Heatmap | Absolute reconstruction error across all test samples |
| Embedding PCA | 2-D PCA projection coloured by true label |
| Cluster Prototypes | Nearest sample to each KMeans centroid |
| Curriculum Schedule | λ vs. epoch for the best configuration |

---

## 15. Best Parameters

{params_block}

---

## 16. Limitations

1. **Not full TS2Vec** — the teacher uses a simplified two-layer CNN without hierarchical
   temporal contrasting. Full TS2Vec integration is left as future work.
2. **Random search only** — Bayesian / Optuna-based search could find better configurations
   more efficiently.
3. **Single dataset per run** — multi-dataset pretraining is not implemented.
4. **CPU-oriented** — training is lightweight but slower than GPU-optimised implementations.
5. **No statistical significance testing** — results should be validated with Friedman /
   Nemenyi tests across multiple datasets.

---

## 17. Future Work

For an IEEE/Springer journal extension:

- [ ] **Full TS2Vec integration** — hierarchical temporal contrasting with timestamp-level loss.
- [ ] **Broader UCR benchmark** — all 128 UCR datasets with proper train/test splits.
- [ ] **Optuna / Bayesian search** — more sample-efficient hyperparameter optimisation.
- [ ] **Statistical tests** — Friedman rank test + Nemenyi post-hoc across all datasets.
- [ ] **Multi-dataset pretraining** — learn a shared representation across multiple time-series.
- [ ] **Multivariate UEA datasets** — extend to multi-channel time series.
- [ ] **Transformer student** — replace LSTM/CNN decoder with a lightweight Transformer.
- [ ] **Concept drift robustness** — evaluate on datasets with distribution shift.

---

*This report was automatically generated by Auto-TSRC v1.0 (prototype).*
*For reproducibility, see `best_params.json` and `automl_trials.csv` in the outputs directory.*
"""

    save_path = os.path.join(reports_dir, f"{dataset}_research_report.md")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[report] Research report saved -> {save_path}")
    return save_path
