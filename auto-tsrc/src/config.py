"""
config.py - Central configuration and default hyperparameters for Auto-TSRC.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ucr")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
EMBEDDINGS_DIR = os.path.join(OUTPUTS_DIR, "embeddings")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")

UCR_ARCHIVE_NAME = "UCRArchive_2018"
UCR_ZIP_NAME = "UCRArchive_2018.zip"
UCR_DOWNLOAD_URL = (
    "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
)

# Default demo datasets
DEFAULT_DATASETS = [
    "ECG200",
    "GunPoint",
    "Coffee",
    "ItalyPowerDemand",
    "TwoLeadECG",
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Teacher training defaults
# ---------------------------------------------------------------------------
TEACHER_DEFAULTS = {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "embedding_dim": 64,
    "temperature": 0.07,
    "channels": [32, 64],
}

# ---------------------------------------------------------------------------
# Student training defaults
# ---------------------------------------------------------------------------
STUDENT_DEFAULTS = {
    "student_type": "cnn_ae",
    "embedding_dim": 64,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "batch_size": 32,
}

# ---------------------------------------------------------------------------
# Curriculum defaults (manual TSRC)
# ---------------------------------------------------------------------------
MANUAL_CURRICULUM = {
    "start_lambda": 0.0,
    "end_lambda": 0.9,
    "delay_epoch": 0,
    "tau": 1.0,
}

# ---------------------------------------------------------------------------
# AutoML search space
# ---------------------------------------------------------------------------
AUTOML_SEARCH_SPACE = {
    "normalization": ["zscore", "robust"],
    "missing_strategy": ["linear_interpolation", "forward_fill", "mean_fill"],
    "clip_outliers": [True, False],
    "augmentation_strength": ["weak", "medium", "strong"],
    "mask_ratio": [0.05, 0.1, 0.2],
    "jitter_sigma": [0.01, 0.03, 0.05],
    "student_type": ["cnn_ae", "lstm_ae"],
    "embedding_dim": [32, 64, 128],
    "hidden_dim": [32, 64, 128],
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size": [16, 32, 64],
    "start_lambda": [0.0, 0.1, 0.2, 0.25],
    "end_lambda": [0.5, 0.7, 0.9, 1.0],
    "delay_epoch": [0, 3, 5],
    "tau": [0.5, 1.0, 1.5, 2.0],
}

# ---------------------------------------------------------------------------
# Robustness corruption levels
# ---------------------------------------------------------------------------
ROBUSTNESS_CORRUPTIONS = [
    {"type": "clean", "level": "none"},
    {"type": "missing_random", "level": "10pct"},
    {"type": "missing_random", "level": "20pct"},
    {"type": "spike_noise", "level": "5pct"},
    {"type": "gaussian_noise", "level": "0.05"},
    {"type": "gaussian_noise", "level": "0.1"},
]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EARLY_STOPPING_PATIENCE = 5

def ensure_output_dirs():
    """Create all output directories if they do not exist."""
    for d in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR, EMBEDDINGS_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)
