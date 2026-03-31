"""
Download ALL planned Ayurvedic datasets to chatbot/dataset/ folder.
Run: python chatbot/download_all_datasets.py
"""
import subprocess
import sys
import os
import json

# Ensure datasets library is installed
try:
    import datasets
except ImportError:
    print("Installing 'datasets' library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub", "--quiet"])
    import datasets

from datasets import load_dataset

# ── Config ──
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.makedirs(DATASET_DIR, exist_ok=True)


def save_json(data, filename):
    path = os.path.join(DATASET_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  -> Saved: {filename} ({len(data)} items, {size_mb:.1f} MB)")


def save_csv(dataset_obj, filename):
    path = os.path.join(DATASET_DIR, filename)
    import pandas as pd
    df = dataset_obj.to_pandas()
    df.to_csv(path, index=False, encoding="utf-8")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  -> Saved: {filename} ({len(df)} rows, {size_mb:.1f} MB)")


# ──────────────────────────────────────
# 1. Ayurveda Text Q&A (OPEN - no gating)
# ──────────────────────────────────────
print("\n" + "="*60)
print("[1/3] Downloading: Macromrit/ayurveda-text-based-qanda")
print("="*60)
try:
    ds = load_dataset("Macromrit/ayurveda-text-based-qanda", token=HF_TOKEN)
    all_data = []
    for split in ds.keys():
        rows = [dict(r) for r in ds[split]]
        print(f"  Split '{split}': {len(rows)} samples")
        all_data.extend(rows)
    save_json(all_data, "ayurveda_qa.json")
except Exception as e:
    print(f"  ERROR: {e}")


# ──────────────────────────────────────
# 2. BhashaBench-Ayur (GATED)
# ──────────────────────────────────────
print("\n" + "="*60)
print("[2/3] Downloading: bharatgenai/BhashaBench-Ayur")
print("="*60)
for lang in ["Hindi", "English"]:
    try:
        print(f"  Downloading {lang} split...")
        ds = load_dataset("bharatgenai/BhashaBench-Ayur", data_dir=lang, split="test", token=HF_TOKEN)
        data = [dict(r) for r in ds]
        save_json(data, f"bhashbench_ayur_{lang.lower()}.json")
    except Exception as e:
        print(f"  ERROR ({lang}): {e}")
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print("  -> This is a gated dataset. Request access at:")
            print("     https://huggingface.co/datasets/bharatgenai/BhashaBench-Ayur")


# ──────────────────────────────────────
# 3. HiMed - Traditional Indian Medicine (OPEN)
# ──────────────────────────────────────
print("\n" + "="*60)
print("[3/3] Downloading: FreedomIntelligence/HiMed (Traditional subsets)")
print("="*60)
for config in ["himed_trad_corpus", "himed_trad_bench"]:
    try:
        print(f"  Downloading {config}...")
        ds = load_dataset("FreedomIntelligence/HiMed", config, token=HF_TOKEN)
        all_data = []
        for split in ds.keys():
            rows = [dict(r) for r in ds[split]]
            print(f"    Split '{split}': {len(rows)} samples")
            all_data.extend(rows)
        save_json(all_data, f"{config}.json")
    except Exception as e:
        print(f"  ERROR ({config}): {e}")


# ── Summary ──
print("\n" + "="*60)
print("DOWNLOAD COMPLETE!")
print("="*60)
print(f"\nFiles in {DATASET_DIR}:")
for f in sorted(os.listdir(DATASET_DIR)):
    fpath = os.path.join(DATASET_DIR, f)
    size = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  {f:40s} {size:8.1f} MB")
print(f"\nTotal files: {len(os.listdir(DATASET_DIR))}")
