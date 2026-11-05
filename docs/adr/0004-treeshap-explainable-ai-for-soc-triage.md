# ADR-0004: Asynchronous TreeSHAP for Explainable AI Alert Enrichment

**Date**: 2026-11-05
**Status**: accepted
**Deciders**: Vespera IDS Engineering Team

## Context

A primary cause of failure in ML-based Network Intrusion Detection Systems (NIDS) is "alert opacity." When an analyst in a Security Operations Center (SOC) receives an alert stating only `Domain: xk9a...net, Confidence: 0.94`, the analyst cannot immediately distinguish between:
1. Malicious command-and-control payload exfiltration.
2. An enterprise antivirus signature lookup.
3. An unusual Content Delivery Network (CDN) hashed hostname.

Manually analyzing PCAP dumps or searching DNS logs to understand *why* the model triggered takes 15–30 minutes per alert, leading to alert fatigue or premature suppression. However, calculating model explainability (such as sampling-based SHAP or LIME) is notoriously CPU-intensive and cannot run on real-time packet processing pipelines.

## Decision

We integrate **TreeSHAP** (`shap.TreeExplainer`) into the asynchronous Tier-2 deep analysis pipeline:
1. TreeSHAP executes strictly off the Tier-1 inline path.
2. It computes exact Shapley attribution values across all 8 features (both Tier-1 stateless and Tier-2 stateful) for every `VERIFIED_CRITICAL` alert.
3. Feature attributions (signed floats representing positive or negative pushes toward the anomaly verdict) are embedded directly into the alert JSON payload and visualized in the interactive Streamlit dashboard.
4. If TreeSHAP computation fails or encounters missing dependencies, the engine gracefully degrades to reporting raw feature values without dropping the alert.

## Alternatives Considered

### Alternative 1: Model-Agnostic Sampling (KernelSHAP or LIME)
- **Pros**: Works across any arbitrary model architecture.
- **Cons**: Requires exponential or high-sample evaluation ($O(2^{|F|})$ or thousands of sample evaluations), taking hundreds of milliseconds to seconds per query.
- **Why not**: Far too slow for Tier-2's $\le 200\text{ ms}$ budget; introduces non-deterministic sampling variance.

### Alternative 2: Global Feature Importance Only (XGBoost Feature Weights / Gain)
- **Pros**: Zero per-alert runtime cost; static metadata.
- **Cons**: Tells the analyst only which features matter across the entire training set, not which specific feature triggered *this specific domain query*.
- **Why not**: Global importance does not help an analyst triage an individual incident.

## Consequences

### Positive
- Actionable local explainability: analysts immediately see whether Shannon entropy, character N-gram KL-divergence, or query payload variance drove the alert.
- High computational efficiency: TreeSHAP exploits tree structure to compute exact Shapley values in polynomial time ($O(T L D^2)$, typically $<5\text{ ms}$ for our tree depths).
- Enables automated SOC triage playbooks (e.g., auto-escalate if `payload_variance` is dominant; route to network team if `unique_ip_concurrency` suggests internal misconfiguration).

### Negative
- Adds `shap` library dependency to Tier-2 worker environments.
- Increases alert JSON payload size slightly to carry the attribution dictionary.

### Risks
- CPU overhead during sustained attack bursts in Tier-2: mitigated by running explainers only for confirmed alerts (`VERIFIED_CRITICAL`), skipping benign or filtered events.
