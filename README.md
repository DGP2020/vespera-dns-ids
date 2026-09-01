# Vespera IDS — DNS Tunneling Detector 🛡️

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vespera.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **🚀 Live Interactive Demo:** [https://vespera.streamlit.app](https://vespera.streamlit.app)

A production-grade, two-tier intrusion detection system that catches DNS tunneling attacks in real time using **XGBoost classifiers**, **stateful windowed feature extraction**, **TreeSHAP explainable AI (XAI)**, and an interactive **Streamlit web dashboard**.

Built as a portfolio project demonstrating applied machine learning in network security, resilient systems design, and clean Python engineering.

---

## 🚀 Live Demo & Interactive Web Dashboard

Test the ML models directly in your browser without local setup:
👉 **[https://vespera.streamlit.app](https://vespera.streamlit.app)**
<img width="1920" height="912" alt="image" src="https://github.com/user-attachments/assets/390b1f5c-d725-41ad-8995-439b32cf8951" />
<img width="1905" height="865" alt="image" src="https://github.com/user-attachments/assets/8ec80e81-452c-4741-b592-7afbeec9e1c8" />
<img width="1920" height="912" alt="image" src="https://github.com/user-attachments/assets/29ee5519-1294-4928-9831-e7ad76cf6aec" />
<img width="1868" height="867" alt="image" src="https://github.com/user-attachments/assets/8d964de9-5b70-4c54-9b1b-d614f02e4e96" />

### Features of the Web Dashboard:
- **Interactive Domain Tester:** Enter any FQDN (e.g., `google.com` or `a1b2c3d4.evil.com`) and DNS record type (`A`, `TXT`, `CNAME`).
- **Tier-1 Fast Triage Score:** Real-time probability scoring based on stateless features (entropy, length, subdomain count, numeric-to-alpha ratio).
- **Tier-2 Deep Analysis:** Stateful sliding window analysis evaluating payload variance, N-gram transition KL-divergence, and client IP concurrency.
- **TreeSHAP Explainable AI:** Detailed breakdown of feature attribution scores showing *why* a domain was flagged.
- **Interactive Presets:** Quickly switch between normal traffic presets and simulated DNS tunneling attacks.

---

## 🏛️ System Architecture

```
[ DNS Traffic / Simulated Queries ]
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  Tier-1: Fast Inline Triage   (≤ 1.5 ms / packet)  │
│  • 4 stateless features (entropy, length, etc.)     │
│  • XGBoost (max_depth=4, n_estimators=50)           │
│  • Dual-threshold router (0.70 / 0.92)              │
└──────────────┬──────────────────────┬───────────────┘
               │ SUSPICIOUS           │ FAST_TRACK (≥0.92)
               ▼                      ▼
     ┌──────────────────┐    ┌─────────────────────┐
     │  Async Queue     │    │  Immediate SOC Alert │
     │  (MemoryQueue /  │    │  + async audit copy │
     │   Redis)         │    └─────────────────────┘
     └────────┬─────────┘
              ▼
┌─────────────────────────────────────────────────────┐
│  Tier-2: Stateful Deep Analysis  (≤ 200 ms)         │
│  • 4 stateful features from 30-s sliding window     │
│  • XGBoost (max_depth=6, n_estimators=200)          │
│  • TreeSHAP explainability (async, off hot path)    │
└──────────────┬──────────────────────────────────────┘
               │ VERIFIED_CRITICAL
               ▼
┌─────────────────────────────────────────────────────┐
│  SOC Alert Dispatcher & Interactive Streamlit App   │
│  • SQLite WAL buffer (survives restarts)            │
│  • Exponential backoff + circuit breaker            │
│  • TreeSHAP XAI Feature Attribution Breakdown       │
└─────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Streamlit Web UI** | Live interactive browser demo hosted on Streamlit Cloud |
| **Tier-1 Classifier** | Stateless XGBoost model (`≤ 1.5 ms` latency budget, `≥ 50,000 QPS` target) |
| **Tier-2 Classifier** | Stateful 30-second sliding window XGBoost model (`≤ 200 ms`) |
| **TreeSHAP Explainable AI** | Per-alert Shapley feature attribution scores for analyst triage |
| **Dual-Threshold Routing** | Benign / Suspicious / Fast-Track routing (0.70 / 0.92 thresholds) |
| **SOC Alert Dispatcher** | SQLite WAL persistence, exponential backoff, circuit breaker, DLQ |
| **TUI & Web Dashboard** | Includes both a `rich` terminal UI and a modern Streamlit web dashboard |
| **Feedback Pipeline** | Daily JSONL logging $\rightarrow$ Hive-partitioned Parquet data lake |
| **Model Promotion Gate** | Automated performance validation + 48h shadow inference + atomic symlink swap |

---

## 📁 Project Structure

```
vespera_ids/
├── app.py                 # Streamlit web dashboard (entry point for cloud deployment)
├── config.py              # Configuration constants & environment variable overrides
├── tier1_triage.py        # Tier-1 stateless features, XGBoost classifier, benchmark
├── tier2_analysis.py      # Tier-2 stateful features, classifier, TreeSHAP engine
├── state_cache.py         # 30-second sliding window state cache
├── router.py              # Dual-threshold routing logic
├── packet_capture.py      # Scapy capture loop (Tier-1 entry point)
├── queue_broker.py        # Abstract BaseQueueBroker interface
├── memory_queue.py        # In-process multiprocessing.Queue provider
├── redis_queue.py         # Redis LIST provider (distributed scaling)
├── alert_storage.py       # SQLite WAL alert buffer + DLQ
├── alert_dispatcher.py    # HTTP webhook dispatcher, circuit breaker, backoff
├── dashboard.py           # Rich terminal dashboard
├── feedback_logger.py     # Daily JSONL feedback logger
├── parquet_lake.py        # Parquet ingestion pipeline & CLI
├── model_promotion.py     # Performance gate, shadow inference & model swap
├── models/                # Trained XGBoost model artifacts (.json)
└── requirements.txt       # Production dependencies
```

---

## 🛠️ Local Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/vespera-dns-ids.git
cd vespera-dns-ids
```

### 2. Create a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Web Dashboard locally
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run the Terminal (TUI) Dashboard
```bash
python dashboard.py
```

---

## 💡 Key Design Decisions

- **Why a Two-Tier Architecture?**
  Single-tier line-rate classifiers face a strict precision vs. recall trade-off. Tier-1 processes hot-path DNS packets in `≤1.5ms` to maximize recall. Tier-2 processes windowed state asynchronously in `≤200ms` to maximize precision without blocking packet ingestion.

- **Why XGBoost over Deep Learning?**
  Tree ensembles are deterministic, offer low latency without GPU scheduling overhead, and support fast exact calculation of TreeSHAP values.

- **Why TreeSHAP?**
  Security Analysts need to know *why* an anomaly was flagged. TreeSHAP computes exact Shapley values, highlighting whether entropy, payload variance, or character N-gram divergence caused the alert.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
