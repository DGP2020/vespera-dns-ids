# ADR-0006: SQLite WAL Alert Staging with Circuit Breaker and Exponential Backoff

**Date**: 2026-11-05
**Status**: accepted
**Deciders**: Vespera IDS Engineering Team

## Context

Security alert delivery to external endpoints (SOAR webhooks, SIEM APIs, PagerDuty, Slack) is subject to transient failures: network partitions, DNS lookup timeouts, endpoint rate-limiting (HTTP 429), and upstream server maintenance (HTTP 502/503). 

If an IDS attempts synchronous delivery on detection threads, external HTTP latency degrades internal detection throughput. If an IDS buffers alerts solely in memory, process restarts or crashes permanently lose unacknowledged alerts. Furthermore, aggressively retrying against a failing endpoint can create a "thundering herd" that prolongs downstream outages.

Finally, an intrusion detection system must never become a vector for Server-Side Request Forgery (SSRF) when configuring alert destinations.

## Decision

We decouple alert generation from network delivery using an asynchronous, resilient dispatch pipeline (`AlertStorage` and `AlertDispatcher`):
1. **Persistent SQLite WAL Buffer**: Every alert payload is written to a local SQLite database in Write-Ahead Logging (`WAL`) mode before HTTP delivery is attempted. Alerts survive daemon restarts.
2. **Exponential Backoff with Jitter**: Retries follow $\text{delay} = \min(\text{retry\_max}, \text{retry\_base} \cdot 2^{\text{retry\_count}}) \pm \text{jitter}$, up to 5 retries.
3. **Three-State Circuit Breaker**: Tracks consecutive failures per endpoint. After 3 consecutive errors, the circuit opens for 60 seconds, skipping outbound network calls. It then enters `HALF_OPEN` to permit a single probe request before resuming normal delivery.
4. **Dead-Letter Queue (DLQ)**: Alerts exceeding maximum retries are promoted to `FAILED_DLQ` status, written to a dead-letter JSONL file, and tracked via an incremented counter.
5. **SSRF Mitigation**: Webhook URLs are validated against private, loopback, and link-local IP ranges (RFC 1918, RFC 3927) unless explicitly allowed for local testing.

## Alternatives Considered

### Alternative 1: Pure In-Memory Async Queue with Python `queue.Queue`
- **Pros**: Fast, simple, no disk I/O.
- **Cons**: Unsent alerts are lost if the process is terminated, upgraded, or encounters an Out-Of-Memory condition.
- **Why not**: Loss of critical security breach alerts during restarts is unacceptable.

### Alternative 2: Synchronous Blocking Webhook Calls
- **Pros**: Immediate notification status.
- **Cons**: Webhook timeouts (typically 5–30 seconds) block detection pipelines, causing packet capture drops.
- **Why not**: Violates isolation between packet ingestion and downstream network services.

## Consequences

### Positive
- Guaranteed alert persistence across process restarts and power cycles.
- Zero latency impact on Tier-1 packet capture or Tier-2 analysis loops.
- Downstream systems are protected from retry storms via circuit breaking and jittered backoff.
- Secure by default against SSRF risks in multi-tenant or enterprise environments.

### Negative
- Local disk I/O for SQLite commits on each alert.
- Requires occasional database cleanup or vacuuming if alert volume is extraordinarily high.

### Risks
- Local disk exhaustion if the external webhook is down indefinitely: mitigated by SQLite storage caps, DLQ offloading, and disk space health monitoring.
