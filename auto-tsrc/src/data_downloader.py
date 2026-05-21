"""
data_downloader.py - Download and extract the UCR Archive 2018.
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

UCR_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
UCR_ZIP_NAME = "UCRArchive_2018.zip"
UCR_ARCHIVE_NAME = "UCRArchive_2018"


def download_ucr_archive(data_dir: str = "data/ucr") -> bool:
    """
    Download the UCR Time Series Archive 2018 if not already present.

    Returns True if successful, False otherwise.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, UCR_ZIP_NAME)
    extract_dir = os.path.join(data_dir, UCR_ARCHIVE_NAME)

    # Already extracted
    if os.path.isdir(extract_dir):
        print(f"[data_downloader] UCR archive already extracted at: {extract_dir}")
        return True

    # Already downloaded but not extracted
    if os.path.isfile(zip_path):
        print(f"[data_downloader] Zip found at {zip_path}. Extracting...")
        return _extract(zip_path, data_dir)

    # Download
    print(f"[data_downloader] Downloading UCR Archive from:\n  {UCR_URL}")
    print("[data_downloader] This is ~60 MB. Please wait...")

    try:
        response = requests.get(UCR_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="UCR Download",
            ncols=80,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    except requests.exceptions.RequestException as e:
        print(f"\n[data_downloader] ERROR: Download failed.\n  Reason: {e}")
        _print_manual_instructions(data_dir, zip_path)
        return False

    return _extract(zip_path, data_dir)


def _extract(zip_path: str, extract_to: str) -> bool:
    """Extract zip archive."""
    try:
        print(f"[data_downloader] Extracting {zip_path} -> {extract_to} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        print("[data_downloader] Extraction complete.")
        return True
    except Exception as e:
        print(f"[data_downloader] ERROR: Extraction failed.\n  Reason: {e}")
        _print_manual_instructions(extract_to, zip_path)
        return False


def _print_manual_instructions(data_dir: str, zip_path: str):
    print(
        "\n"
        "========================================================\n"
        "  MANUAL DOWNLOAD REQUIRED\n"
        "========================================================\n"
        f"  1. Go to: {UCR_URL}\n"
        f"  2. Download UCRArchive_2018.zip\n"
        f"  3. Place it here: {zip_path}\n"
        f"  4. Re-run this script.\n"
        "========================================================\n"
    )


def verify_dataset(dataset_name: str, data_dir: str = "data/ucr") -> bool:
    """Check if a specific UCR dataset is available."""
    archive_dir = os.path.join(data_dir, UCR_ARCHIVE_NAME)
    train_file = os.path.join(archive_dir, dataset_name, f"{dataset_name}_TRAIN.tsv")
    test_file = os.path.join(archive_dir, dataset_name, f"{dataset_name}_TEST.tsv")

    if not os.path.isfile(train_file):
        print(f"[data_downloader] WARNING: {train_file} not found.")
        return False
    if not os.path.isfile(test_file):
        print(f"[data_downloader] WARNING: {test_file} not found.")
        return False
    return True
