# ADR-0007: Automated Model Promotion Gate with Shadow Inference and Symlink Swapping

**Date**: 2026-11-05
**Status**: accepted
**Deciders**: Vespera IDS Engineering Team

## Context

In security machine learning systems, deploying retrained models directly to production carries substantial risk:
1. **Recall Regression**: A retrained model might experience a 0.5% drop in recall that allows a specific DNS tunneling tool to pass undetected.
2. **False Positive Surge**: Subtle data distribution shifts can cause massive false positive spikes on benign enterprise DNS traffic.
3. **Latency Regression**: Increased tree depth or suboptimal splits can breach the strict $\le 1.5\text{ ms}$ Tier-1 latency budget, causing packet drops.
4. **Deployment Downtime**: In-place model file replacement can cause race conditions or partial file reads while worker processes are actively classifying queries.

## Decision

We implement an automated, multi-stage model promotion pipeline (`model_promotion.py`):
1. **Automated Performance Gate**: Candidate models must be evaluated against a frozen Golden Validation Dataset and a 14-day rolling evaluation dataset. The model must satisfy hard performance bounds:
   - Tier-1: $\text{Recall} \ge 99.5\%$ with $0\%$ latency budget violation.
   - Tier-2: $\text{Precision} \ge 98.0\%$, False Positive Rate degradation $< 0.1\%$.
   - Latency Regression: Any regression exceeding $5\%$ over the latency budget triggers an immediate build rejection.
2. **Shadow Inference**: Before active promotion, the candidate runs silently alongside the live model on replayed or live traffic for 48 hours (`DNS_PROMO_SHADOW_HOURS`), logging hypothetical verdicts without triggering SOC alerts.
3. **Atomic Symlink Promotion**: Active models are accessed via symlinks (`models/xgboost_tier1_active.json`). Promotion atomically points the symlink to the new timestamped model file via `os.replace` / atomic rename, enabling zero-downtime hot-swapping and instantaneous rollback.

## Alternatives Considered

### Alternative 1: Direct File Overwrite
- **Pros**: Simple file copying.
- **Cons**: Race conditions where active processes read partially written JSON files; no instantaneous rollback mechanism.
- **Why not**: Causes crashes or indeterminate inference during deployments.

### Alternative 2: Manual Analyst Approval Only (No Automated Gates)
- **Pros**: Human oversight for all promotions.
- **Cons**: Subjective, slow, prone to human error, and inconsistent across different analysts.
- **Why not**: Does not mathematically guarantee SLA compliance or recall floors.

## Consequences

### Positive
- Mathematically impossible for a model with degraded recall ($<99.5\%$) or excessive latency ($>1.5\text{ ms}$) to enter active enforcement.
- Zero-downtime model updates: running processes switch seamlessly on file pointer updates.
- Instantaneous rollback: if an anomaly is identified post-promotion, reverting the symlink to the previous model artifact takes $<1\text{ ms}$.

### Negative
- Slower release velocity: candidate models must undergo evaluation against dual datasets and the 48-hour shadow inference window.
- Storage overhead: requires archiving historical model JSON artifacts and golden parquet datasets.

### Risks
- Outdated golden validation dataset: mitigated by periodically refreshing the golden dataset with analyst-verified ground-truth events from the Parquet feedback lake.
