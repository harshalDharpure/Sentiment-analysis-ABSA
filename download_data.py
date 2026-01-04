"""
Download script for DimABSA 2026 datasets.
Downloads the official datasets from the GitHub repository.
"""

import os
import requests
from pathlib import Path
import argparse

# Base URL for the DimABSA2026 repository
BASE_URL = "https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/subtask_1/eng/"

# Default files to download (English, Track A, Subtask 1)
DEFAULT_FILES = {
    "eng_laptop_dev_task1.jsonl": "eng_laptop_dev_task1.jsonl",
    "eng_laptop_train_alltasks.jsonl": "eng_laptop_train_alltasks.jsonl",
    "eng_restaurant_dev_task1.jsonl": "eng_restaurant_dev_task1.jsonl",
    "eng_restaurant_train_alltasks.jsonl": "eng_restaurant_train_alltasks.jsonl",
}


def download_file(url: str, save_path: Path) -> bool:
    """Download a file from URL and save to path."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def main():
    """Download all default datasets."""
    parser = argparse.ArgumentParser(description="Download DimABSA 2026 datasets")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()
    
    # Determine project root (parent of this script's directory)
    script_dir = Path(__file__).parent
    if args.output_dir:
        data_dir = Path(args.output_dir) / "raw"
    else:
        data_dir = script_dir / "data" / "raw"
    
    print("="*60)
    print("Downloading DimABSA 2026 datasets...")
    print(f"Save directory: {data_dir}")
    print("="*60)
    
    success_count = 0
    total_files = len(DEFAULT_FILES)
    
    # Download default English files
    for filename, local_name in DEFAULT_FILES.items():
        url = BASE_URL + filename
        save_path = data_dir / local_name
        print(f"\n[{success_count + 1}/{total_files}] Downloading {filename}...")
        if download_file(url, save_path):
            file_size = save_path.stat().st_size / (1024 * 1024)  # MB
            print(f"[OK] Saved to {save_path} ({file_size:.2f} MB)")
            success_count += 1
        else:
            print(f"[FAILED] Failed to download {filename}")
    
    print("\n" + "="*60)
    print(f"Download Summary: {success_count}/{total_files} files")
    if success_count == total_files:
        print("[SUCCESS] All default files downloaded successfully!")
        print("\nNext steps:")
        print("  1. Check data format: head -n 1 data/raw/eng_laptop_train_alltasks.jsonl")
        print("  2. Start training: python -m src.train --train-file data/raw/eng_laptop_train_alltasks.jsonl ...")
    else:
        print("[WARNING] Some files failed to download. Check your internet connection.")
        print("  You can manually download from:")
        print(f"  {BASE_URL}")
    print("="*60)


if __name__ == "__main__":
    main()
