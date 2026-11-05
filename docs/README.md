# Vespera IDS Documentation

Welcome to the technical architecture and decision records documentation for **Vespera IDS** (DNS Tunneling Intrusion Detection System).

## Documentation Contents

### 1. [System Architecture Specification](ARCHITECTURE.md)
The central architectural reference document describing the end-to-end design, data flows, subsystem mechanics, performance budgets, and security posture of Vespera IDS:
- **High-Level Architecture**: Two-tier inline/async processing pipeline and end-to-end Mermaid topology.
- **Subsystems**:
  - Packet Capture & Ingestion Engine (`packet_capture.py`, `main.py`)
  - Tier-1 Fast Stateless Inline Triage (`tier1_triage.py`)
  - Dual-Threshold Routing Layer (`router.py`)
  - Decoupled Queue Broker Infrastructure (`queue_broker.py`, `memory_queue.py`, `redis_queue.py`)
  - Tier-2 Stateful Sliding-Window Analysis (`tier2_analysis.py`, `state_cache.py`, `queue_consumer.py`)
  - Explainable AI Engine (`TreeSHAP`, `shap.TreeExplainer`)
  - Resilient SOC Alerting & Dispatch Pipeline (`alert_storage.py`, `alert_dispatcher.py`)
  - Continual Learning & Data Lake Pipeline (`feedback_logger.py`, `parquet_lake.py`)
  - Model Governance, Gating & Promotion Pipeline (`model_promotion.py`, `prepare_and_train.py`)
  - User Interfaces & Visualization (`app.py`, `dashboard.py`, `index.html`)
- **Performance Budgets & SLAs**: Detailed per-component latency and throughput metrics.
- **Defensive Hardening**: SSRF protection, HMAC/Bearer signing, circuit breakers, and queue backpressure.
- **Configuration Engine**: Environment variable dictionary and subsystem parameters (`config.py`).

### 2. [Architecture Decision Records (ADR)](adr/README.md)
Formal decision logs capturing the motivation, context, trade-offs, and consequences of significant technical choices made in this project:
- [ADR-0001: Two-Tier Hybrid Detection Architecture](adr/0001-two-tier-hybrid-detection-architecture.md)
- [ADR-0002: XGBoost Decision Trees over Deep Learning for Detection](adr/0002-xgboost-decision-trees-over-deep-learning.md)
- [ADR-0003: Dual-Threshold Routing with Non-Blocking Fast-Track](adr/0003-dual-threshold-routing-with-fast-track.md)
- [ADR-0004: Asynchronous TreeSHAP for Explainable AI Alert Enrichment](adr/0004-treeshap-explainable-ai-for-soc-triage.md)
- [ADR-0005: Pluggable Abstract Queue Broker (In-Memory & Redis)](adr/0005-abstract-queue-broker-memory-redis.md)
- [ADR-0006: SQLite WAL Alert Staging with Circuit Breaker and Exponential Backoff](adr/0006-sqlite-wal-buffering-and-circuit-breaker.md)
- [ADR-0007: Automated Model Promotion Gate with Shadow Inference and Symlink Swapping](adr/0007-shadow-inference-model-promotion-gate.md)
- [ADR-0008: Hive-Partitioned Parquet Data Lake for Feedback and Retraining](adr/0008-hive-partitioned-parquet-feedback-lake.md)
- [ADR Template](adr/template.md): Blank template for recording future architectural decisions.
