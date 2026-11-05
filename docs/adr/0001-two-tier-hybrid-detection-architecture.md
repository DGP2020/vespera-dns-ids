# ADR-0001: Two-Tier Hybrid Detection Architecture

**Date**: 2026-11-05
**Status**: accepted
**Deciders**: Vespera IDS Engineering Team

## Context

DNS tunneling detection operates under severe operational constraints. Enterprise DNS recursors and network tap interfaces handle tens of thousands of queries per second (QPS). Inline inspection must satisfy a sub-2 millisecond per-packet latency budget ($\le 1.5\text{ ms}$) to prevent packet buffer overflow and network degradation.

At the same time, sophisticated DNS tunneling utilities (e.g., `dnscat2`, `iodine`, `Cobalt Strike` DNS beacons) employ high-entropy payloads, character substitution, and low-and-slow query cadences. Accurately detecting these stealthy attacks requires rich stateful context: evaluating query-frequency histograms, temporal variance in payload sizes, character transition distributions, and cross-client query concurrency over sliding time windows. 

Single-tier architectures force an untenable trade-off:
1. Lightweight stateless inspection operates at line speed but generates high false-positive rates on normal enterprise traffic (e.g., CDN hashes, anti-virus updates).
2. Deep stateful inspection provides high precision but incurs computational latency ($\ge 50\text{ ms}$) and memory contention that cannot keep pace with 50,000 QPS wire-speed traffic.

## Decision

We adopt a decoupled, two-tier hybrid detection architecture:
1. **Tier-1 (Fast Inline Triage)**: A lightweight, stateless classifier that executes synchronously in $\le 1.5\text{ ms}$ on pinned CPU cores. It classifies incoming queries into Benign, Suspicious, or Fast-Track categories based on four basic packet-level metrics.
2. **Tier-2 (Asynchronous Stateful Deep Analysis)**: An off-hot-path stateful classifier operating on a 30-second sliding window cache. Tier-2 processes only events flagged as Suspicious (and background audits for Fast-Track events), evaluating multi-query temporal metrics within a $\le 200\text{ ms}$ budget without blocking network ingestion.

## Alternatives Considered

### Alternative 1: Single-Tier In-Line Deep Stateful Classifier
- **Pros**: Uniform model pipeline, immediate final verdict per query, eliminates inter-process queueing.
- **Cons**: Exceeds the 1.5 ms packet budget by an order of magnitude; under sudden traffic bursts, backpressure forces packet dropping or recursor timeouts.
- **Why not**: Wire-speed DNS processing cannot tolerate variable multi-millisecond state-aggregation latency on every single packet.

### Alternative 2: Pure Batch / Post-Processing SIEM Ingestion
- **Pros**: Zero impact on network packet forwarding; queries can be processed offline with arbitrary complexity.
- **Cons**: High detection latency (minutes to hours); fails to disrupt active data exfiltration or active Command & Control (C2) channels in real time.
- **Why not**: DNS tunneling can exfiltrate sensitive credentials in seconds; real-time threat response requires immediate sub-second alerting.

## Consequences

### Positive
- Strict latency guarantees on the inline hot path ($\le 1.5\text{ ms}$, $\ge 50,000\text{ QPS}$ capability).
- 95%+ of normal traffic is dismissed at Tier-1, drastically reducing the load on complex stateful feature extractors.
- Tier-2 can allocate rich computational resources (N-gram KL-divergence, TreeSHAP explainability) without risking network bottlenecks.
- Independent horizontal scaling: Tier-1 capture nodes and Tier-2 worker processes scale according to their specific resource bottlenecks.

### Negative
- Increased architectural complexity: requires IPC queue brokers, process management, and synchronization.
- Minor latency window ($\approx 50\text{–}200\text{ ms}$) between packet capture and Tier-2 verified alert emission for suspicious queries.

### Risks
- Queue saturation during sustained anomalous traffic spikes: mitigated by configurable queue capacity caps, overflow disk logging, and priority fast-tracking.
