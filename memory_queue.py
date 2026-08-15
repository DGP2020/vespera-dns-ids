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
# MemoryQueueProvider
# ---------------------------------------------------------------------------

class MemoryQueueProvider(BaseQueueBroker):
    """
    Wraps multiprocessing.Queue for single-process (monolithic MVP) use.

    put() is strictly non-blocking:
      - Tries a non-blocking enqueue.
      - On queue.Full: calls on_overflow() then returns False immediately.
        The Tier-1 capture loop never waits.

    get() uses a configurable timeout so the Tier-2 consumer worker can
    sleep efficiently without busy-looping.
    """

    def __init__(self, cfg: Optional[QueueConfig] = None) -> None:
        self._cfg = cfg or get_config().queue
        self._q: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=self._cfg.max_size
        )
        logger.info(
            "MemoryQueueProvider initialised (maxsize=%d, overflow_log=%s)",
            self._cfg.max_size,
            self._cfg.overflow_log_path,
        )

    # ------------------------------------------------------------------
    # BaseQueueBroker implementation
    # ------------------------------------------------------------------

    def put(self, event: dict) -> bool:
        """
        Non-blocking enqueue.  Returns True on success, False on drop.
        Caller must treat False as a successful no-op; the dropped event
        has already been persisted to the disk ring-buffer and the metric
        counter incremented.
        """
        try:
            self._q.put_nowait(event)
            return True
        except queue.Full:
            self.on_overflow(event)
            return False

    def get(self, timeout: float = 0.1) -> Optional[dict]:
        """
        Block for up to `timeout` seconds waiting for an event.
        Returns None if the queue is empty at the end of the timeout.
        """
        try:
            return self._q.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        """
        Approximate current depth.  Not perfectly accurate for
        multiprocessing.Queue across process boundaries, but sufficient
        for dashboard metrics.
        """
        try:
            return self._q.qsize()
        except NotImplementedError:
            # macOS/some Unix platforms don't support qsize on mp.Queue
            return -1

    def close(self) -> None:
        """Close and join the internal queue."""
        try:
            self._q.close()
            self._q.join_thread()
        except Exception:  # noqa: BLE001
            logger.exception("Error while closing MemoryQueueProvider")

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
            f"maxsize={self._cfg.max_size}, "
            f"qsize≈{self.qsize()}, "
            f"overflows={get_overflow_count()})"
        )
