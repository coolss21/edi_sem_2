"""
utils.py - Utility functions: seeding, device selection, timing, logging.
"""

import os
import random
import time
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Return the best available torch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array to float32 tensor on device."""
    return torch.tensor(x, dtype=torch.float32, device=device)


def numpy_to_loader(X: np.ndarray, y=None, batch_size: int = 32,
                    shuffle: bool = True, seed: int = 42) -> torch.utils.data.DataLoader:
    """Wrap numpy arrays into a DataLoader."""
    from torch.utils.data import TensorDataset, DataLoader
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y_t = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(X_t, y_t)
    else:
        ds = TensorDataset(X_t)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                      drop_last=False)


def save_json(data: dict, path: str):
    """Save dict as JSON."""
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> dict:
    """Load JSON file."""
    import json
    with open(path, "r") as f:
        return json.load(f)


def print_banner(text: str):
    width = max(60, len(text) + 4)
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_section(text: str):
    print(f"\n{'-'*60}")
    print(f"  {text}")
    print(f"{'-'*60}")
