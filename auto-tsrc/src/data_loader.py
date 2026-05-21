"""
data_loader.py - Load UCR Time Series datasets in standard format.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


UCR_ARCHIVE_NAME = "UCRArchive_2018"


def load_ucr_dataset(
    dataset_name: str,
    data_dir: str = "data/ucr",
):
    """
    Load a UCR 2018 dataset.

    File format:
        UCRArchive_2018/<DatasetName>/<DatasetName>_TRAIN.tsv
        UCRArchive_2018/<DatasetName>/<DatasetName>_TEST.tsv

    First column = label, remaining columns = time-series values.

    Returns
    -------
    X_train : np.ndarray  shape (N_train, T, 1)  float32
    y_train : np.ndarray  shape (N_train,)        int64
    X_test  : np.ndarray  shape (N_test, T, 1)   float32
    y_test  : np.ndarray  shape (N_test,)         int64
    """
    archive_dir = os.path.join(data_dir, UCR_ARCHIVE_NAME)
    train_path = os.path.join(archive_dir, dataset_name, f"{dataset_name}_TRAIN.tsv")
    test_path  = os.path.join(archive_dir, dataset_name, f"{dataset_name}_TEST.tsv")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"Train file not found: {train_path}\n"
            "Run with --download-data first, or manually place UCRArchive_2018.zip "
            f"in {data_dir}."
        )
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path, sep="\t", header=None)
    test_df  = pd.read_csv(test_path,  sep="\t", header=None)

    # First column = labels
    y_train_raw = train_df.iloc[:, 0].values
    y_test_raw  = test_df.iloc[:, 0].values

    X_train_raw = train_df.iloc[:, 1:].values.astype(np.float32)
    X_test_raw  = test_df.iloc[:, 1:].values.astype(np.float32)

    # Encode labels consistently
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = le.transform(y_train_raw).astype(np.int64)
    y_test  = le.transform(y_test_raw).astype(np.int64)

    # Shape: (N, T, 1)
    X_train = X_train_raw[:, :, np.newaxis]  # (N_train, T, 1)
    X_test  = X_test_raw[:, :, np.newaxis]   # (N_test, T, 1)

    n_classes   = len(le.classes_)
    time_steps  = X_train.shape[1]

    print(
        f"\n[data_loader] Dataset        : {dataset_name}\n"
        f"              Train samples  : {X_train.shape[0]}\n"
        f"              Test samples   : {X_test.shape[0]}\n"
        f"              Series length  : {time_steps}\n"
        f"              Num classes    : {n_classes}\n"
        f"              Label classes  : {list(le.classes_)}\n"
    )

    return X_train, y_train, X_test, y_test
