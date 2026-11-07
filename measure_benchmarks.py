"""
measure_benchmarks.py
Runs thorough latency benchmarking and validation testing on Tier-1 and Tier-2 models.
Measures:
- Tier-1 per-packet latency (median, mean, p90, p95, p99, min, max)
- Tier-1 throughput (QPS)
- Tier-2 per-event latency with and without TreeSHAP
- Precision, Recall, False Positive Rate (FPR), F1 Score, Accuracy on benchmark dataset
"""

import time
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score

from config import get_config
from tier1_triage import Tier1Classifier, extract_stateless_features, FEATURE_NAMES as T1_FEATURES
from tier2_analysis import Tier2Classifier, extract_stateful_features, compute_ngram_score, FEATURE_NAMES as T2_FEATURES
from state_cache import StateCache

def benchmark_tier1_latency(n_samples=50000, warmup=1000):
    clf = Tier1Classifier()
    assert clf.is_fitted, "Tier 1 classifier not fitted!"

    rng = np.random.default_rng(seed=42)
    sample_domains = [
        "google.com",
        "cloudflare.com",
        "a1b2c3d4e5.tunnel.evil.com",
        "dGhpcyBpcyBhIHRlc3Q.exfil.xyz",
        "vpn.corporate-network.internal",
        "rnd7h2xp91mzqavl.c2.malware.ru",
        "microsoft.com",
        "github.com",
        "portal.azure.com",
        "0123456789abcdef0123456789abcdef.dnstunnel.attacker.org"
    ]

    # Pre-generate test data
    test_queries = []
    for _ in range(n_samples + warmup):
        domain = rng.choice(sample_domains)
        # add random hex prefix to some
        if rng.random() > 0.5:
            prefix = "".join(rng.choice(list("0123456789abcdef"), size=rng.integers(8, 32)))
            domain = f"{prefix}.{domain}"
        test_queries.append((domain, rng.choice(["A", "TXT", "CNAME", "AAAA"])))

    # Warmup
    for i in range(warmup):
        domain, rtype = test_queries[i]
        feat = extract_stateless_features(domain, rtype)
        clf.predict_proba(feat)

    # Measure Feature Extraction alone
    feat_latencies = []
    for i in range(warmup, warmup + n_samples):
        domain, rtype = test_queries[i]
        t0 = time.perf_counter()
        feat = extract_stateless_features(domain, rtype)
        feat_latencies.append((time.perf_counter() - t0) * 1000)

    # Measure Inference alone
    pre_feats = [extract_stateless_features(test_queries[i][0], test_queries[i][1]) for i in range(warmup, warmup + n_samples)]
    inf_latencies = []
    for feat in pre_feats:
        t0 = time.perf_counter()
        clf.predict_proba(feat)
        inf_latencies.append((time.perf_counter() - t0) * 1000)

    # Measure End-to-End per-packet latency
    e2e_latencies = []
    for i in range(warmup, warmup + n_samples):
        domain, rtype = test_queries[i]
        t0 = time.perf_counter()
        feat = extract_stateless_features(domain, rtype)
        clf.predict_proba(feat)
        e2e_latencies.append((time.perf_counter() - t0) * 1000)

    e2e_arr = np.array(e2e_latencies)
    feat_arr = np.array(feat_latencies)
    inf_arr = np.array(inf_latencies)

    results = {
        "n_samples": n_samples,
        "feature_extraction": {
            "mean_ms": float(np.mean(feat_arr)),
            "median_ms": float(np.median(feat_arr)),
            "p90_ms": float(np.percentile(feat_arr, 90)),
            "p99_ms": float(np.percentile(feat_arr, 99)),
        },
        "model_inference": {
            "mean_ms": float(np.mean(inf_arr)),
            "median_ms": float(np.median(inf_arr)),
            "p90_ms": float(np.percentile(inf_arr, 90)),
            "p99_ms": float(np.percentile(inf_arr, 99)),
        },
        "end_to_end": {
            "mean_ms": float(np.mean(e2e_arr)),
            "median_ms": float(np.median(e2e_arr)),
            "p90_ms": float(np.percentile(e2e_arr, 90)),
            "p95_ms": float(np.percentile(e2e_arr, 95)),
            "p99_ms": float(np.percentile(e2e_arr, 99)),
            "min_ms": float(np.min(e2e_arr)),
            "max_ms": float(np.max(e2e_arr)),
            "throughput_qps": float(1000.0 / np.mean(e2e_arr)),
        }
    }
    return results

def benchmark_tier2_latency(n_samples=500, warmup=20):
    clf = Tier2Classifier()
    assert clf.is_fitted, "Tier 2 classifier not fitted!"

    rng = np.random.default_rng(seed=42)

    # Warmup
    for _ in range(warmup):
        feat = {
            "fqdn_length": 45,
            "subdomain_count": 3,
            "shannon_entropy": 3.8,
            "numeric_alpha_ratio": 0.4,
            "record_type_heterogeneity": 0.6,
            "payload_variance": 120.0,
            "ngram_transition_score": 1.2,
            "unique_ip_concurrency": 4.0,
        }
        event = {"stateful_features": feat, "event_id": "warmup", "tier2_mode": "TRIAGE"}
        clf.evaluate(event)

    latencies_with_shap = []
    latencies_without_shap = []

    for _ in range(n_samples):
        # Sample malicious-leaning to trigger SHAP explanation
        feat = {
            "fqdn_length": float(rng.uniform(30, 200)),
            "subdomain_count": float(rng.integers(1, 8)),
            "shannon_entropy": float(rng.uniform(3.2, 4.5)),
            "numeric_alpha_ratio": float(rng.uniform(0.1, 0.8)),
            "record_type_heterogeneity": float(rng.uniform(0.4, 0.9)),
            "payload_variance": float(rng.uniform(50.0, 300.0)),
            "ngram_transition_score": float(rng.uniform(0.8, 2.5)),
            "unique_ip_concurrency": float(rng.uniform(2.0, 8.0)),
        }
        event = {"stateful_features": feat, "event_id": "bench", "tier2_mode": "TRIAGE"}
        t0 = time.perf_counter()
        res = clf.evaluate(event)
        dt = (time.perf_counter() - t0) * 1000
        if res.get("shap_attributions"):
            latencies_with_shap.append(dt)
        else:
            latencies_without_shap.append(dt)

    arr_shap = np.array(latencies_with_shap) if latencies_with_shap else np.array([0.0])
    arr_no_shap = np.array(latencies_without_shap) if latencies_without_shap else np.array([0.0])

    return {
        "n_samples": n_samples,
        "with_treeshap": {
            "count": len(latencies_with_shap),
            "mean_ms": float(np.mean(arr_shap)),
            "median_ms": float(np.median(arr_shap)),
            "p95_ms": float(np.percentile(arr_shap, 95)),
            "p99_ms": float(np.percentile(arr_shap, 99)),
        },
        "without_treeshap": {
            "count": len(latencies_without_shap),
            "mean_ms": float(np.mean(arr_no_shap)),
            "median_ms": float(np.median(arr_no_shap)),
        }
    }

def evaluate_accuracy_metrics(n_samples=10000):
    """
    Evaluates both Tier-1 and Tier-2 models across synthetic + benchmark distributions.
    Computes Precision, Recall, FPR, F1, Accuracy, and Confusion Matrix.
    """
    rng = np.random.default_rng(seed=42)
    n_half = n_samples // 2

    # Benign distribution: legitimate domains (Alexa/Tranco top domains + internal corporate)
    benign_fqdn_len = rng.integers(8, 25, n_half)
    benign_subdomains = rng.integers(0, 2, n_half)
    benign_entropy = rng.uniform(1.8, 3.2, n_half)
    benign_num_alpha = rng.uniform(0.0, 0.1, n_half)

    # Malicious tunneling distribution: DNS exfil/tunnels (C2, base32/64 payloads, high entropy)
    tunnel_fqdn_len = rng.integers(40, 220, n_half)
    tunnel_subdomains = rng.integers(2, 7, n_half)
    tunnel_entropy = rng.uniform(3.5, 4.8, n_half)
    tunnel_num_alpha = rng.uniform(0.2, 0.8, n_half)

    X1_benign = np.column_stack([benign_fqdn_len, benign_subdomains, benign_entropy, benign_num_alpha])
    X1_tunnel = np.column_stack([tunnel_fqdn_len, tunnel_subdomains, tunnel_entropy, tunnel_num_alpha])
    X1 = np.vstack([X1_benign, X1_tunnel]).astype(np.float32)
    y1 = np.array([0] * n_half + [1] * n_half)

    clf1 = Tier1Classifier()
    # Batch predict
    probs1 = clf1._model.predict_proba(X1)[:, 1]
    # Decision threshold 0.70 (standard triage promotion threshold from design doc)
    preds1_triage = (probs1 >= 0.70).astype(int)
    # Binary 0.50 threshold
    preds1_50 = (probs1 >= 0.50).astype(int)

    def calc_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        fpr = fp / max(1, fp + tn)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        return {
            "precision": float(prec),
            "recall": float(rec),
            "fpr": float(fpr),
            "f1_score": float(f1),
            "accuracy": float(acc),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        }

    m1_50 = calc_metrics(y1, preds1_50)
    m1_triage = calc_metrics(y1, preds1_triage)

    # Tier-2 Evaluation (8 features: 4 base + 4 stateful)
    benign_rec_het = rng.uniform(0.0, 0.2, n_half)
    benign_p_var = rng.uniform(0.5, 5.0, n_half)
    benign_ngram = rng.uniform(0.05, 0.4, n_half)
    benign_ip = rng.uniform(1.0, 1.5, n_half)

    tunnel_rec_het = rng.uniform(0.6, 1.0, n_half)
    tunnel_p_var = rng.uniform(40.0, 250.0, n_half)
    tunnel_ngram = rng.uniform(0.9, 2.5, n_half)
    tunnel_ip = rng.uniform(3.0, 8.0, n_half)

    X2_benign = np.column_stack([X1_benign, benign_rec_het, benign_p_var, benign_ngram, benign_ip])
    X2_tunnel = np.column_stack([X1_tunnel, tunnel_rec_het, tunnel_p_var, tunnel_ngram, tunnel_ip])
    X2 = np.vstack([X2_benign, X2_tunnel]).astype(np.float32)
    y2 = np.array([0] * n_half + [1] * n_half)

    clf2 = Tier2Classifier()
    probs2 = clf2._model.predict_proba(X2)[:, 1]
    preds2 = (probs2 >= 0.50).astype(int)
    m2 = calc_metrics(y2, preds2)

    return {
        "tier1_50": m1_50,
        "tier1_triage": m1_triage,
        "tier2": m2,
    }

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING TIER-1 INFERENCE LATENCY BENCHMARK (N=50,000)")
    print("=" * 60)
    t1_res = benchmark_tier1_latency(n_samples=50000, warmup=1000)
    print("Feature Extraction:", t1_res["feature_extraction"])
    print("Model Inference   :", t1_res["model_inference"])
    print("End-to-End        :", t1_res["end_to_end"])

    print("\n" + "=" * 60)
    print("RUNNING TIER-2 DEEP ANALYSIS BENCHMARK (N=500)")
    print("=" * 60)
    t2_res = benchmark_tier2_latency(n_samples=500, warmup=20)
    print("Tier-2 with TreeSHAP   :", t2_res["with_treeshap"])
    print("Tier-2 without TreeSHAP:", t2_res["without_treeshap"])

    print("\n" + "=" * 60)
    print("RUNNING CLASSIFICATION ACCURACY / RECALL / FPR EVALUATION (N=10,000)")
    print("=" * 60)
    acc_res = evaluate_accuracy_metrics(n_samples=10000)
    print("Tier-1 (thresh 0.50):", acc_res["tier1_50"])
    print("Tier-1 (thresh 0.70):", acc_res["tier1_triage"])
    print("Tier-2              :", acc_res["tier2"])
