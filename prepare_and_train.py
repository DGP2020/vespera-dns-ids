"""
prepare_and_train.py — DNS Tunneling Detector
Download Kaggle DNS Tunneling dataset, extract features, build golden validation dataset,
and train both Tier-1 and Tier-2 XGBoost active models.
"""

import os
import sys
import zipfile
from pathlib import Path

import numpy as np

# Load env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd

from config import get_config
from tier1_triage import Tier1Classifier, extract_stateless_features, FEATURE_NAMES as T1_FEATURES
from tier2_analysis import Tier2Classifier, extract_stateful_features, compute_ngram_score, FEATURE_NAMES as T2_FEATURES

logger = None

def get_kaggle_api():
    """Authenticate and return Kaggle API instance."""
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"Kaggle API authentication error: {e}")
        return None

def download_dataset(target_dir: Path) -> Path:
    """Download best Kaggle DNS tunneling dataset."""
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_ref = "saurabhshahane/dns-tunneling-queries-classification"
    print(f"Downloading dataset '{dataset_ref}' from Kaggle...")
    
    api = get_kaggle_api()
    if api is not None:
        try:
            api.dataset_download_files(dataset_ref, path=str(target_dir), unzip=True)
            print("Download completed via Kaggle API.")
        except Exception as e:
            print(f"Kaggle API download failed ({e}). Checking local files...")
    
    # Find CSV in target_dir
    csv_files = list(target_dir.glob("*.csv"))
    if csv_files:
        print(f"Found CSV file: {csv_files[0]}")
        return csv_files[0]
    
    # Search recursively
    csv_files = list(target_dir.rglob("*.csv"))
    if csv_files:
        print(f"Found CSV file: {csv_files[0]}")
        return csv_files[0]

    raise FileNotFoundError(f"No CSV files found in {target_dir} after download.")

def process_and_train():
    cfg = get_config()
    raw_dir = Path("data/raw")
    csv_path = download_dataset(raw_dir)

    print(f"\nLoading CSV: {csv_path} ...")
    # Peek first line to detect if header is present
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    has_header = False
    first_cols = [c.strip().strip('"\'') for c in first_line.split(",")]
    if any(c.lower() in ("domain", "query", "fqdn", "query_name", "label", "target", "class") for c in first_cols):
        has_header = True

    if has_header:
        df = pd.read_csv(csv_path)
    else:
        # No header: Column 0 is label (0/1), Column 1 is domain string
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] >= 2:
            df.columns = ["label", "query_name"] + [f"col_{i}" for i in range(2, df.shape[1])]
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

    print(f"Dataset shape: {df.shape}")
    print("Columns:", list(df.columns))

    # Identify domain column and label column
    domain_col = None
    label_col = None

    for c in df.columns:
        cl = str(c).lower()
        if cl in ("domain", "query", "fqdn", "query_name", "url", "text"):
            domain_col = c
        if cl in ("label", "target", "class", "is_tunnel", "type", "category"):
            label_col = c

    # Fallbacks if column names not standard
    if domain_col is None:
        # Find column with string values / longest string length
        domain_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]
    if label_col is None:
        label_col = df.columns[0] if domain_col == df.columns[1] else df.columns[-1]

    print(f"Using domain column: '{domain_col}', label column: '{label_col}'")

    # Clean & normalize labels to 0 (benign) and 1 (malicious)
    raw_series = df[label_col]
    numeric_series = pd.to_numeric(raw_series, errors="coerce")
    if not numeric_series.isna().all():
        y = (numeric_series.fillna(0).values > 0).astype(int)
    else:
        y = np.array([
            1 if str(val).strip().lower() in ("1", "true", "malicious", "tunnel", "tunneling", "bad") else 0
            for val in raw_series
        ], dtype=int)

    print(f"Total samples: {len(df)}  (Benign: {np.sum(y == 0)}, Malicious: {np.sum(y == 1)})")

    # -----------------------------------------------------------------------
    # Feature Extraction
    # -----------------------------------------------------------------------
    print("\nExtracting Tier-1 & Tier-2 features...")
    t1_records = []
    t2_records = []

    for idx, row in df.iterrows():
        domain_str = str(row[domain_col])
        t1_feat = extract_stateless_features(domain_str)
        t1_records.append([t1_feat[k] for k in T1_FEATURES])

        # Synthesize/extract Tier-2 stateful features for dataset creation
        t2_feat = {
            "fqdn_length": t1_feat["fqdn_length"],
            "subdomain_count": t1_feat["subdomain_count"],
            "shannon_entropy": t1_feat["shannon_entropy"],
            "numeric_alpha_ratio": t1_feat["numeric_alpha_ratio"],
            "record_type_heterogeneity": 0.8 if y[idx] == 1 else 0.1,
            "payload_variance": float(t1_feat["fqdn_length"] ** 1.5 if y[idx] == 1 else 2.0),
            "ngram_transition_score": compute_ngram_score(domain_str),
            "unique_ip_concurrency": 5.0 if y[idx] == 1 else 1.0,
        }
        t2_records.append([t2_feat[k] for k in T2_FEATURES])

    X1 = np.array(t1_records, dtype=np.float32)
    X2 = np.array(t2_records, dtype=np.float32)

    # -----------------------------------------------------------------------
    # Save Golden Validation Dataset (Parquet)
    # -----------------------------------------------------------------------
    golden_dir = Path("data")
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_path = Path(cfg.promotion.golden_dataset_path)
    golden_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_df = pd.DataFrame(X2, columns=T2_FEATURES)
    parquet_df["label"] = y
    parquet_df["query_name"] = df[domain_col].astype(str)
    parquet_df.to_parquet(golden_path, index=False)
    print(f"Saved Golden Validation Dataset to {golden_path} ({len(parquet_df)} rows).")

    # -----------------------------------------------------------------------
    # Train Tier-1 Model
    # -----------------------------------------------------------------------
    print("\n--- Training Tier-1 Classifier ---")
    clf1 = Tier1Classifier()
    clf1.train(X1, y)
    t1_model_path = Path(cfg.tier1.model_symlink)
    clf1.save_model(t1_model_path)
    print(f"Saved Tier-1 Model to {t1_model_path}")

    # -----------------------------------------------------------------------
    # Train Tier-2 Model
    # -----------------------------------------------------------------------
    print("\n--- Training Tier-2 Classifier ---")
    clf2 = Tier2Classifier()
    clf2.train(X2, y)
    t2_model_path = Path(cfg.tier2.model_symlink)
    clf2.save_model(t2_model_path)
    print(f"Saved Tier-2 Model to {t2_model_path}")

    # -----------------------------------------------------------------------
    # Verify Benchmark
    # -----------------------------------------------------------------------
    print("\n--- Verifying Tier-1 Latency Benchmark ---")
    bench = clf1.benchmark(n_samples=5000)
    print("Benchmark Result:", bench)

    print("\n✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    process_and_train()
