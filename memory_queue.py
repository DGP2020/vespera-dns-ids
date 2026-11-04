"""
memory_queue.py — DNS Tunneling Detector
MemoryQueueProvider — Monolithic MVP queue broker.

Design doc reference: dnstds §6 (MemoryQueueProvider)
  - Wraps multiprocessing.Queue with a hard cap of QUEUE_MAX_SIZE elements.
  - On overflow: drop-and-log.  The raw event is written to the disk
    ring-buffer log; the TIER2_QUEUE_OVERFLOW_DROP metric counter is
    incremented.  The Tier-1 inline loop is NEVER blocked.

This provider is dependency-free and is the target for Phase 1 (monolithic
MVP). The distributed RedisQueueProvider (redis_queue.py) swaps in later
without touching any Tier-1 or Tier-2 code.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue
import threading
import time
import zlib
from pathlib import Path
from typing import Optional

from config import get_config, QueueConfig
from queue_broker import BaseQueueBroker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric counter (in-process, thread-safe)
# ---------------------------------------------------------------------------

_overflow_lock = threading.Lock()
_overflow_counter: int = 0  # TIER2_QUEUE_OVERFLOW_DROP


def get_overflow_count() -> int:
    """Return total overflow-drop count since process start."""
    with _overflow_lock:
        return _overflow_counter


def _increment_overflow() -> None:
    global _overflow_counter
    with _overflow_lock:
        _overflow_counter += 1


# ---------------------------------------------------------------------------
# Disk ring-buffer writer
# ---------------------------------------------------------------------------

def _write_overflow_log(event: dict, log_path: Path) -> None:
    """
    Append a single dropped event to the disk ring-buffer log in JSONL
    format.  Creates parent directories if needed.  Errors here must not
    propagate — disk logging is best-effort and must never interrupt the
    Tier-1 capture loop.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({
            "ts": time.time(),
            "type": "TIER2_QUEUE_OVERFLOW_DROP",
            **event,
        })
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to write overflow event to disk log %s — "
            "the drop is still counted in the metric counter.",
            log_path,
        )


# ---------------------------------------------------------------------------
# MemoryQueuePartition — dedicated single-worker queue view
# ---------------------------------------------------------------------------

class MemoryQueuePartition(BaseQueueBroker):
    """
    Dedicated partition queue view wrapping a single multiprocessing.Queue.
    Passed to an individual Tier-2 worker process for consistent-hash partition consumption.
    Pickle-safe for multiprocessing start methods (spawn/fork).
    """

    def __init__(
        self,
        q: multiprocessing.Queue,
        partition_id: int = 0,
        overflow_log_path: Optional[Path] = None,
    ) -> None:
        self._q = q
        self._partition_id = partition_id
        self._overflow_log_path = overflow_log_path or Path("data/logs/queue_overflow.log")

    @property
    def partition_id(self) -> int:
        return self._partition_id

    def put(self, event: dict) -> bool:
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            self.on_overflow(event)
            return False

    def get(self, timeout: float = 0.1) -> Optional[dict]:
        try:
            return self._q.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except NotImplementedError:
            return -1

    def close(self) -> None:
        try:
            self._q.close()
            self._q.join_thread()
        except Exception:  # noqa: BLE001
            logger.exception("Error while closing MemoryQueuePartition %d", self._partition_id)

    def on_overflow(self, event: dict) -> None:
        _increment_overflow()
        _write_overflow_log(event, Path(self._overflow_log_path))
        super().on_overflow(event)

    def __repr__(self) -> str:  # pragma: no cover
        return f"MemoryQueuePartition(id={self._partition_id}, qsize≈{self.qsize()})"


# ---------------------------------------------------------------------------
# MemoryQueueProvider
# ---------------------------------------------------------------------------

class MemoryQueueProvider(BaseQueueBroker):
    """
    Wraps multiprocessing.Queue for single-process (monolithic MVP) use,
    with built-in consistent-hash partitioning across Tier-2 workers.

    put() is strictly non-blocking:
      - Hashes parent_domain (via CRC32) to deterministically select target partition.
      - Tries a non-blocking enqueue into that partition.
      - On queue.Full: calls on_overflow() then returns False immediately.
        The Tier-1 capture loop never waits.

    get() uses a configurable timeout so the Tier-2 consumer worker can
    sleep efficiently without busy-looping.
    """

    def __init__(
        self,
        cfg: Optional[QueueConfig] = None,
        num_partitions: int = 1,
    ) -> None:
        self._cfg = cfg or get_config().queue
        self._num_partitions = max(1, num_partitions)
        self._partitions: list[multiprocessing.Queue] = [
            multiprocessing.Queue(maxsize=self._cfg.max_size)
            for _ in range(self._num_partitions)
        ]
        self._partition_wrappers: list[MemoryQueuePartition] = [
            MemoryQueuePartition(
                q=self._partitions[i],
                partition_id=i,
                overflow_log_path=Path(self._cfg.overflow_log_path),
            )
            for i in range(self._num_partitions)
        ]
        # Primary queue reference for backwards compatibility (e.g. single queue access)
        self._q: multiprocessing.Queue = self._partitions[0]
        logger.info(
            "MemoryQueueProvider initialised (partitions=%d, maxsize_per_partition=%d, overflow_log=%s)",
            self._num_partitions,
            self._cfg.max_size,
            self._cfg.overflow_log_path,
        )

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def get_partition(self, partition_id: int) -> MemoryQueuePartition:
        """
        Return the dedicated partition view for worker partition_id.
        """
        if partition_id < 0 or partition_id >= self._num_partitions:
            raise IndexError(
                f"Partition index {partition_id} out of range [0, {self._num_partitions})"
            )
        return self._partition_wrappers[partition_id]

    def _get_partition_index(self, event: dict) -> int:
        if self._num_partitions <= 1:
            return 0
        domain = event.get("parent_domain")
        if not domain:
            qname = event.get("query_name", "")
            parts = qname.rstrip(".").split(".")
            domain = ".".join(parts[-2:]) if len(parts) >= 2 else qname
        if not domain:
            return 0
        return (zlib.crc32(domain.encode("utf-8")) & 0xFFFFFFFF) % self._num_partitions

    # ------------------------------------------------------------------
    # BaseQueueBroker implementation
    # ------------------------------------------------------------------

    def put(self, event: dict) -> bool:
        """
        Non-blocking enqueue partitioned deterministically by parent_domain hash.
        Returns True on success, False on drop.
        Caller must treat False as a successful no-op; the dropped event
        has already been persisted to the disk ring-buffer and the metric
        counter incremented.
        """
        idx = self._get_partition_index(event)
        target_q = self._partitions[idx]
        try:
            target_q.put_nowait(event)
            return True
        except queue.Full:
            self.on_overflow(event)
            return False

    def get(self, timeout: float = 0.1) -> Optional[dict]:
        """
        Block for up to `timeout` seconds waiting for an event on partition 0.
        In multi-worker setups, each worker consumes from its own partition via get_partition().
        Returns None if the queue is empty at the end of the timeout.
        """
        try:
            return self._partitions[0].get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        """
        Approximate current total depth across all partitions.
        """
        total = 0
        for q in self._partitions:
            try:
                sz = q.qsize()
                if sz >= 0:
                    total += sz
            except NotImplementedError:
                return -1
        return total

    def close(self) -> None:
        """Close and join all internal partition queues."""
        for q in self._partitions:
            try:
                q.close()
                q.join_thread()
            except Exception:  # noqa: BLE001
                logger.exception("Error while closing partition queue")

    # ------------------------------------------------------------------
    # Overflow hook
    # ------------------------------------------------------------------

    def on_overflow(self, event: dict) -> None:
        """
        Log the overflow to disk and increment the metric counter.
        Never raises — must not interrupt the Tier-1 capture loop.
        """
        _increment_overflow()
        _write_overflow_log(event, Path(self._cfg.overflow_log_path))
        # Let the base class emit the WARNING log line as well.
        super().on_overflow(event)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MemoryQueueProvider("
            f"partitions={self._num_partitions}, "
            f"maxsize_per_partition={self._cfg.max_size}, "
            f"qsize≈{self.qsize()}, "
            f"overflows={get_overflow_count()})"
        )
