# Auto-TSRC: AutoML-Guided Robust Curriculum Distillation for Interpretable Time-Series Representation Learning

> A prototype research implementation extending the TSRC framework with AutoML-guided hyperparameter search, robust preprocessing, robustness testing, and standardised explainability outputs.

---

## What This Project Does

**Auto-TSRC** trains a teacher-student time-series representation learning system where:

- A **LightweightContrastiveCNNTeacher** (inspired by TS2Vec-style contrastive learning) is pre-trained using SimCLR NT-Xent loss on augmented views of time series.
- A **student autoencoder** (CNN or LSTM) learns compact, interpretable embeddings by jointly minimising reconstruction loss and knowledge distillation (hint) loss from the frozen teacher.
- A **curriculum scheduler** controls the balance between reconstruction and distillation loss using a configurable power-law schedule.

---

## Why This Is an Extension of TSRC

The original TSRC paper defines curriculum distillation for time-series but leaves open:

| Gap | Auto-TSRC Solution |
|---|---|
| Manual hyperparameter tuning | AutoML random search with successive halving |
| No robustness analysis | 6 corruption types evaluated systematically |
| No explainability outputs | 5 standardised visualisations generated automatically |
| Fixed preprocessing | Search over missing-value strategy, normalization, clipping |

---

## What Is New

### 1. AutoML Search
- Searches over 15 hyperparameters: preprocessing, augmentation, student architecture, embedding dimension, learning rate, batch size, curriculum schedule.
- Successive-halving style: Stage 1 (many trials, few epochs) → Stage 2 (top-k, more epochs) → Stage 3 (best config, full training).
- Multi-objective scoring: 0.35×ARI + 0.20×NMI + 0.15×CHI + 0.15×ReconQuality + 0.10×Robustness − 0.05×Time.

### 2. Robust Preprocessing
- Missing value handling: linear interpolation → forward fill → mean fill fallback.
- Outlier clipping via training-set quantiles.
- Z-score or robust (median/IQR) normalisation.
- All parameters fitted on training data only.

### 3. Robustness Testing
- 6 corruption types: `missing_10`, `missing_20`, `spike_noise`, `gaussian_0.05`, `gaussian_0.1`, `clean`.
- ARI/NMI drop from clean baseline measured for Manual TSRC vs Auto-TSRC.

### 4. Explainability Outputs
- Reconstruction overlay (original vs reconstructed).
- Reconstruction error heatmap (time × samples).
- Embedding PCA 2-D scatter plot coloured by true label.
- Cluster prototypes (nearest sample to each KMeans centroid).
- Curriculum lambda schedule plot.

---

## Installation

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run Commands

### Quick demo (recommended first run)
```bash
python main.py --download-data --datasets ECG200 --trials 3 --trial-epochs 2 --final-epochs 5 --quick
```

### Standard demo with 3 datasets
```bash
python main.py --download-data --datasets ECG200 GunPoint Coffee --trials 8 --trial-epochs 5 --final-epochs 20
```

### Single dataset, more search
```bash
python main.py --download-data --datasets ECG200 --trials 8 --trial-epochs 5 --final-epochs 20
```

### Using the convenience script
```bash
python run_quick_demo.py
```

### All available arguments
```
--download-data        Download UCR Archive 2018 automatically
--datasets             One or more UCR dataset names (default: ECG200)
--data-dir             Path to store UCR data (default: data/ucr)
--outputs-dir          Root output directory (default: outputs)
--trials               Number of AutoML trials (default: 8)
--trial-epochs         Epochs per trial (default: 5)
--final-epochs         Epochs for best-config final training (default: 20)
--teacher-epochs       Epochs for teacher pre-training (default: 10)
--batch-size           Override batch size
--seed                 Random seed (default: 42)
--device               auto | cpu | cuda (default: auto)
--quick                Reduce epochs for fast CPU test
```

---

## Output Files

After a successful run on `ECG200`, you will find:

| File | Description |
|---|---|
| `outputs/reports/ECG200_best_params.json` | Best AutoML hyperparameters |
| `outputs/tables/ECG200_automl_trials.csv` | All trial results with scores |
| `outputs/tables/ECG200_comparison_metrics.csv` | Baseline vs Auto-TSRC comparison |
| `outputs/tables/ECG200_robustness_metrics.csv` | Robustness under corruptions |
| `outputs/tables/ECG200_training_history.csv` | Per-epoch training losses |
| `outputs/plots/ECG200_reconstruction_overlay.png` | Original vs reconstructed |
| `outputs/plots/ECG200_error_heatmap.png` | Reconstruction error heatmap |
| `outputs/plots/ECG200_embedding_pca.png` | Embedding PCA scatter |
| `outputs/plots/ECG200_cluster_prototypes.png` | Cluster prototype time series |
| `outputs/plots/ECG200_curriculum_schedule.png` | Lambda vs epoch schedule |
| `outputs/reports/ECG200_research_report.md` | Full Markdown research report |
| `outputs/embeddings/ECG200_auto_tsrc_embeddings.npy` | Numpy embeddings array |
| `outputs/models/ECG200_best_student.pt` | Best student model checkpoint |
| `outputs/models/ECG200_best_teacher.pt` | Best teacher model checkpoint |

---

## Known Limitations

1. **Not full TS2Vec** — the teacher is a simplified 2-layer CNN, not the full hierarchical temporal contrasting of TS2Vec.
2. **Random search only** — no Bayesian/Optuna-based search.
3. **CPU-oriented** — lightweight by design; GPU will speed things up significantly.
4. **No statistical significance tests** — results should be validated with Friedman/Nemenyi tests across multiple datasets.
5. **Single dataset per run** — multi-dataset pretraining not implemented.

---

## Future Work (IEEE/Springer Version)

- [ ] Full TS2Vec integration with hierarchical temporal contrasting
- [ ] All 128 UCR datasets evaluated
- [ ] Friedman/Nemenyi statistical significance tests
- [ ] Optuna/Bayesian hyperparameter search
- [ ] Multi-dataset pretraining
- [ ] Multivariate UEA datasets (NATOPS, ArticularyWordRecognition, etc.)
- [ ] Transformer-based student decoder
- [ ] Concept drift robustness scenarios

---

## Teacher Model Notice

The `LightweightContrastiveCNNTeacher` is **inspired by TS2Vec-style contrastive representation learning** but is **not** a full reproduction of the original TS2Vec paper. It uses a two-layer Conv1D encoder with SimCLR NT-Xent loss rather than hierarchical timestamp-level contrasting.

---

## Citation / Attribution

This is a prototype research implementation. If using as a basis for published work, please:
1. Cite the original TSRC paper.
2. Cite the TS2Vec paper (Yue et al., NeurIPS 2022) for the contrastive learning inspiration.
3. Note that this is a prototype extension, not a full reproduction.
