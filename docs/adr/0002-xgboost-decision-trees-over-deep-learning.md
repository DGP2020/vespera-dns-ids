# ADR-0002: XGBoost Decision Trees over Deep Learning for Detection

**Date**: 2026-11-05
**Status**: accepted
**Deciders**: Vespera IDS Engineering Team

## Context

The intrusion detection system requires machine learning engines for both Tier-1 (stateless, sub-millisecond) and Tier-2 (stateful, complex feature interaction) classification. The chosen model family must satisfy strict operational criteria:
1. **Deterministic Latency**: Inference must execute consistently on CPU cores without kernel jitter or garbage-collection spikes.
2. **Tabular Performance**: Network IDS features (packet lengths, character entropy, sub-domain counts, variance, KL-divergence) are structured tabular features.
3. **Hardware Footprint**: The IDS must run on standard commodity Linux network appliances without requiring dedicated GPU or NPU accelerators.
4. **Explainability**: Security Operations Center (SOC) analysts must receive transparent, per-feature attribution for every generated alert.

## Decision

We select **XGBoost (Extreme Gradient Boosting)** with the histogram-based tree method (`tree_method='hist'`) as the ML classifier for both detection tiers:
- **Tier-1 Model**: Shallow gradient-boosted forest (`max_depth=4`, `n_estimators=50`) optimized for minimal execution time and maximum recall.
- **Tier-2 Model**: Deeper gradient-boosted forest (`max_depth=6`, `n_estimators=200`) trained on the full 8-dimensional feature vector for high precision.

## Alternatives Considered

### Alternative 1: Deep Learning Sequence Models (LSTM / 1D-CNN / Transformer)
- **Pros**: Can learn representation directly from raw character sequences of domain names without explicit feature engineering.
- **Cons**: High CPU inference latency ($>10\text{ ms}$ per sample without GPU batching); batching is incompatible with single-packet inline streaming; memory-intensive runtime dependencies (PyTorch/TensorFlow/ONNX); non-trivial exact explainability.
- **Why not**: DL models fail the Tier-1 1.5 ms latency budget on CPU hardware and add massive deployment overhead for modest gains on tabular features.

### Alternative 2: Classical Linear Models / Naive Bayes / Random Forest (scikit-learn)
- **Pros**: Minimal computational cost; simple to interpret coefficients in linear models.
- **Cons**: Linear models fail to capture non-linear feature interactions (e.g., high entropy is normal on common CDNs unless paired with abnormal subdomain count and record-type heterogeneity). Scikit-learn Random Forests lack the optimized C++ prediction runtime and compact serialization of XGBoost.
- **Why not**: Gradient boosted trees provide significantly superior ROC-AUC and precision-recall trade-offs on tabular network data.

## Consequences

### Positive
- Highly deterministic, microsecond-level CPU inference via native C++ prediction kernels.
- Native, direct integration with `shap.TreeExplainer`, enabling exact polynomial-time Shapley feature attribution ($O(T L D^2)$).
- Compact JSON-based model serialization (`.json`), enabling lightweight version control, atomic file swapping, and auditability.
- No heavy GPU/CUDA driver dependencies.

### Negative
- Requires explicit, domain-specific feature engineering upfront rather than automated representation learning.
- Susceptible to tabular feature drift if attack tools change encoding styles (addressed via the automated retraining and promotion pipeline).

### Risks
- Model drift when new legitimate high-entropy top-level domains emerge: mitigated by continuous data lake logging and the 14-day rolling performance gate.
