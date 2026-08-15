"""
queue_broker.py — DNS Tunneling Detector
Abstract base interface for queue providers.

Design doc reference: dnstds §6 — Queue / Backpressure
The BaseQueueBroker abstraction ensures that core Tier-1 producer and
Tier-2 consumer code never changes when swapping between the monolithic
MemoryQueueProvider and the distributed RedisQueueProvider.

Both providers expose identical put() / get() signatures.
"""

from __future__ import annotations

import abc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BaseQueueBroker(abc.ABC):
    """
    Abstract queue broker.

    Concrete implementations:
        - MemoryQueueProvider  (memory_queue.py) — Monolithic MVP
        - RedisQueueProvider   (redis_queue.py)  — Distributed Phase 2
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def put(self, event: dict) -> bool:
        """
        Enqueue a triage event payload.

        Args:
            event: Serialisable dict containing at minimum the keys
                   produced by tier1_triage.py (base features + metadata).

        Returns:
            True  — event was successfully enqueued.
            False — event was dropped (queue full / OOM). Caller must
                    already have written the drop to the disk fallback log.
        """

    @abc.abstractmethod
    def get(self, timeout: float = 0.1) -> Optional[dict]:
        """
        Dequeue the next triage event.

        Args:
            timeout: Seconds to block waiting for an item.

        Returns:
            dict  — next event payload.
            None  — queue was empty for the full timeout window.
        """

    @abc.abstractmethod
    def qsize(self) -> int:
        """Return an approximate current queue depth."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release any underlying resources (threads, connections)."""

    # ------------------------------------------------------------------
    # Optional hook — providers may override for richer logging
    # ------------------------------------------------------------------

    def on_overflow(self, event: dict) -> None:
        """
        Called by the provider when an enqueue attempt is dropped due
        to capacity limits.  Default implementation just logs at WARNING.
        Providers should call super().on_overflow(event) after their own
        disk-logging logic so the log line is always emitted.
        """
        logger.warning(
            "Queue overflow: event dropped. "
            "event_id=%s  source_ip=%s  query=%s",
            event.get("event_id", "<unknown>"),
            event.get("source_ip", "<unknown>"),
            event.get("query_name", "<unknown>"),
        )
