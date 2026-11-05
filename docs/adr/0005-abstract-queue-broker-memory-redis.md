# ADR-0005: Pluggable Abstract Queue Broker (In-Memory & Redis)

**Date**: 2026-11-05
**Status**: accepted
**Deciders**: Vespera IDS Engineering Team

## Context

Vespera IDS must support two distinct operational environments:
1. **Phase 1 / Edge Appliance / Local Testing**: A single host running packet capture, triage, and deep analysis without requiring third-party daemon installation or external network services.
2. **Phase 2 / Enterprise Distributed Cluster**: Multiple network sensor nodes capturing DNS traffic across different subnets, feeding a shared distributed pool of Tier-2 worker processes.

Hardcoding a specific IPC mechanism (such as Python's `multiprocessing.Queue` or a Redis client) would either make distributed scale impossible or force a heavy multi-service infrastructure footprint for simple edge deployments.

## Decision

We establish an abstract queue broker interface (`BaseQueueBroker`) with standardized enqueue, dequeue, capacity inspection, and health-check methods. We implement two interchangeable providers:
1. **`MemoryQueueProvider`**: Uses Python's native `multiprocessing.Queue` for zero-dependency, high-throughput, in-process/multi-process communication on a single machine.
2. **`RedisQueueProvider`**: Uses Redis `LPUSH` / `BRPOP` primitives with optional SSL authentication, supporting distributed multi-sensor clusters and independent Tier-2 scaling.

Both providers share a unified disk-overflow strategy: if the queue exceeds its configured capacity (`DNS_QUEUE_MAX_SIZE`), excess items are safely appended to an overflow log file (`queue_overflow.log`) rather than blocking the Tier-1 capture loop.

## Alternatives Considered

### Alternative 1: Hardcoding Python `multiprocessing.Queue`
- **Pros**: Zero third-party dependencies, fast in-memory IPC.
- **Cons**: Cannot scale horizontally across multiple machines or capture probes.
- **Why not**: Prevents deployment in enterprise multi-sensor network taps.

### Alternative 2: Hardcoding Enterprise Message Brokers (Kafka / RabbitMQ)
- **Pros**: High durability, persistent partition logs, clustering support.
- **Cons**: High operational overhead, JVM or Erlang runtime requirements, unsuitable for standalone network sensor appliances or local testing.
- **Why not**: Violates our goal of a lightweight, portable security sensor.

## Consequences

### Positive
- Zero friction for local development, CI/CD testing, and lightweight edge appliance deployment.
- Seamless upgrade path to multi-node distributed capture by setting `DNS_REDIS_HOST` in configuration.
- Unified backpressure handling: disk spillover protects Tier-1 from blocking regardless of which queue provider is active.

### Negative
- Abstraction restricts queue semantics to common denominator operations (simple FIFO queueing; no complex routing keys or pub/sub fanout within the queue broker itself).

### Risks
- Redis network partition in distributed mode: mitigated by bounded timeouts and automated spillover to local disk logging when Redis is unreachable.
