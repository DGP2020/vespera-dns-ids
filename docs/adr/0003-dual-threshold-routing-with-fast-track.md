# ADR-0003: Dual-Threshold Routing with Non-Blocking Fast-Track

**Date**: 2026-11-05
**Status**: accepted
**Deciders**: Vespera IDS Engineering Team

## Context

In high-throughput network monitoring, Tier-1 produces a continuous anomaly probability score $P \in [0.0, 1.0]$. A single binary threshold creates two failure modes:
1. If the threshold is set high (e.g., $P \ge 0.90$), subtle or stealthy tunneling attempts ($P \in [0.70, 0.90]$) escape detection completely.
2. If the threshold is set low (e.g., $P \ge 0.65$), massive volumes of ambiguous enterprise queries are treated as confirmed incidents, overwhelming SOC analysts and exhausting downstream webhook channels.

Furthermore, when a high-confidence, blatant attack occurs (e.g., high-volume base64 data exfiltration over TXT queries), delaying notification while the query waits in a Tier-2 processing queue increases attacker dwell time.

## Decision

We implement a **Dual-Threshold Routing System** with three distinct operational regimes:
- **$P < 0.70$ (BENIGN)**: The query is dismissed immediately. No queueing, no logging, zero downstream overhead.
- **$0.70 \le P < 0.92$ (SUSPICIOUS)**: The query is enqueued to the asynchronous Tier-2 queue for stateful multi-query sliding-window verification.
- **$P \ge 0.92$ (FAST_TRACK)**: An immediate high-severity alert is dispatched directly to the SOC buffer *before* enqueuing an asynchronous copy to Tier-2. Tier-2 analysis enriches the existing incident view for auditing, but never gates or delays the alert.

## Alternatives Considered

### Alternative 1: Single Binary Threshold ($P \ge \theta$)
- **Pros**: Minimal routing logic; binary classification output.
- **Cons**: Cannot differentiate between ambiguous anomalies requiring multi-query context and blatant attacks requiring zero-second mitigation.
- **Why not**: Sacrifices either detection speed on catastrophic attacks or precision on border-line cases.

### Alternative 2: Synchronous Tier-2 Escalation for All Anomalies
- **Pros**: All alerts benefit from full stateful verification before firing.
- **Cons**: Queuing delays under high load directly delay SOC notification of active exfiltration.
- **Why not**: A blatant attack should never wait behind an asynchronous queue backlog.

## Consequences

### Positive
- Zero delay for unambiguous attacks: fast-track alerts fire in milliseconds from Tier-1.
- Significant reduction of Tier-2 queue volume: only ambiguous traffic ($0.70 \le P < 0.92$) requires deep sliding-window state accumulation.
- Clear separation of concerns between fast response (Tier-1) and forensic verification (Tier-2).

### Negative
- Requires maintaining two thresholds ($P_{\text{benign\_max}} = 0.70$ and $P_{\text{fasttrack\_min}} = 0.92$) that must be tuned and validated together during model promotion.

### Risks
- False positives on the fast-track path: mitigated by setting the default threshold conservatively high ($0.92$) and validating that candidate models meet a $\ge 99.5\%$ recall and zero-false-positive-spike criteria during automated promotion gating.
