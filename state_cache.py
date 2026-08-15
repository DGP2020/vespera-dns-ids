"""
state_cache.py — DNS Tunneling Detector
30-second sliding window state cache for Tier-2 stateful feature extraction.

Design doc reference: dnstds §4 — Tier-2 Deep Analysis

Context window: 30-second sliding in-memory key-value cache, keyed by
parent domain / client IP. Entries older than STATE_WINDOW_SECONDS are
pruned automatically on every update.

Thread-safety: all public methods hold a per-key RLock so concurrent
Tier-2 worker processes can safely share one StateCache instance (if
multiprocessing.Manager is used), or more typically each worker process
has its own StateCache that it updates exclusively.

Tracked per (parent_domain, client_ip) key:
  - record_type_counts : dict[str, int]      TXT/CNAME/AAAA/A/… hit counts
  - payload_lengths    : list[float]          ordered payload byte lengths
  - unique_client_ips  : set[str]             distinct clients querying zone
  - timestamps         : list[float]          event Unix timestamps (for pruning)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from config import get_config, Tier2Config


# ---------------------------------------------------------------------------
# Per-key window entry
# ---------------------------------------------------------------------------

@dataclass
class WindowEntry:
    """
    All state accumulated for one (parent_domain, client_ip) key within
    the current 30-second sliding window.
    """
    # Event-level timestamps — used for pruning old records
    timestamps: List[float] = field(default_factory=list)

    # Parallel list of record types aligned with timestamps — used to
    # rebuild record_type_counts accurately after pruning expired events.
    record_type_log: List[str] = field(default_factory=list)

    # Record type hit counts (A, TXT, CNAME, AAAA, MX, …)
    # Kept as a derived aggregate; rebuilt from record_type_log on prune.
    record_type_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Ordered list of payload/query byte lengths per successive query
    payload_lengths: List[float] = field(default_factory=list)

    # All distinct client IPs that have queried this parent domain zone
    # (keyed by zone, not per-client — for unique_ip_concurrency metric)
    unique_client_ips: Set[str] = field(default_factory=set)

    def total_queries(self) -> int:
        return len(self.timestamps)


# ---------------------------------------------------------------------------
# StateCache
# ---------------------------------------------------------------------------

class StateCache:
    """
    Thread-safe 30-second sliding window state accumulator.

    Key: (parent_domain: str, client_ip: str)
    Value: WindowEntry — mutable, pruned on each update.

    Usage:
        cache = StateCache()

        # On each Tier-2 event:
        cache.update(
            parent_domain="evil.com",
            client_ip="10.0.0.5",
            record_type="TXT",
            payload_length=220.0,
            ts=time.time(),
        )

        # Before Tier-2 feature extraction:
        snapshot = cache.snapshot("evil.com", "10.0.0.5")
    """

    def __init__(self, cfg: Optional[Tier2Config] = None) -> None:
        self._cfg = cfg or get_config().tier2
        self._window_sec: int = self._cfg.state_window_seconds  # 30s

        # Primary store: (parent_domain, client_ip) → WindowEntry
        self._store: dict[tuple[str, str], WindowEntry] = {}
        # Per-parent-domain store: parent_domain → set of all unique client IPs
        # (used for unique_ip_concurrency across ALL clients of a zone)
        self._zone_clients: dict[str, set[str]] = defaultdict(set)

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def update(
        self,
        parent_domain: str,
        client_ip: str,
        record_type: str,
        payload_length: float,
        ts: Optional[float] = None,
    ) -> None:
        """
        Record a single DNS query event into the sliding window.

        Args:
            parent_domain: Second-level + TLD portion of the FQDN
                           (e.g. "evil.com" from "data.evil.com").
            client_ip:     Source IP of the DNS client.
            record_type:   DNS QTYPE string ("A", "TXT", "CNAME", …).
            payload_length: Total byte length of the DNS query wire payload.
            ts:            Unix timestamp of the event (defaults to now).
        """
        now = ts or time.time()
        key = (parent_domain, client_ip)

        with self._lock:
            if key not in self._store:
                self._store[key] = WindowEntry()

            entry = self._store[key]

            # Append new event
            entry.timestamps.append(now)
            entry.record_type_log.append(record_type.upper())
            entry.record_type_counts[record_type.upper()] += 1
            entry.payload_lengths.append(payload_length)
            entry.unique_client_ips.add(client_ip)

            # Track this client in the zone-level registry
            self._zone_clients[parent_domain].add(client_ip)

            # Prune stale events from this entry
            self._prune_entry(entry, now)

    def evict_stale_keys(self) -> int:
        """
        Remove entries whose entire window has expired.
        Should be called periodically (e.g., every 5 seconds) from a
        background maintenance thread to prevent unbounded memory growth.

        Returns:
            Number of keys evicted.
        """
        now = time.time()
        cutoff = now - self._window_sec
        removed = 0

        with self._lock:
            stale_keys = [
                key for key, entry in self._store.items()
                if not entry.timestamps or entry.timestamps[-1] < cutoff
            ]
            for key in stale_keys:
                del self._store[key]
                removed += 1

        return removed

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def snapshot(
        self,
        parent_domain: str,
        client_ip: str,
        ts: Optional[float] = None,
    ) -> Optional[WindowEntry]:
        """
        Return a read-only snapshot of the current window entry for this key.

        The snapshot is a *copy* of the live entry post-prune — safe to read
        outside the lock without racing against concurrent updates.

        Returns None if no data has been accumulated for this key.
        """
        now = ts or time.time()
        key = (parent_domain, client_ip)

        with self._lock:
            if key not in self._store:
                return None

            entry = self._store[key]
            self._prune_entry(entry, now)

            if not entry.timestamps:
                return None

            # Return a shallow copy so the caller cannot mutate live state
            snap = WindowEntry(
                timestamps=list(entry.timestamps),
                record_type_log=list(entry.record_type_log),
                record_type_counts=dict(entry.record_type_counts),
                payload_lengths=list(entry.payload_lengths),
                unique_client_ips=set(entry.unique_client_ips),
            )
            return snap

    def zone_unique_ip_count(self, parent_domain: str) -> int:
        """
        Return the number of distinct client IPs that have queried
        `parent_domain` within any recorded window entry.
        This implements the 'unique client IP concurrency against the
        same anomalous zone' feature from dnstds §4.
        """
        with self._lock:
            return len(self._zone_clients.get(parent_domain, set()))

    def key_count(self) -> int:
        """Current number of live (not yet evicted) cache keys."""
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_entry(self, entry: WindowEntry, now: float) -> None:
        """
        Remove all events older than STATE_WINDOW_SECONDS from entry.
        Mutates entry in-place; caller must hold self._lock.

        Rebuilds record_type_counts from the surviving record_type_log slice
        so that stale counts from expired events never leak into the
        record_type_heterogeneity feature (bug fix).
        """
        cutoff = now - self._window_sec
        # Find the first index still within the window
        idx = 0
        while idx < len(entry.timestamps) and entry.timestamps[idx] < cutoff:
            idx += 1

        if idx == 0:
            return  # nothing to prune

        # Slice away expired events from all aligned lists
        entry.timestamps = entry.timestamps[idx:]
        entry.record_type_log = entry.record_type_log[idx:]
        entry.payload_lengths = entry.payload_lengths[idx:]

        # Rebuild record_type_counts from the surviving record_type_log so
        # counts reflect only the in-window events (core correctness fix).
        new_counts: Dict[str, int] = defaultdict(int)
        for rtype in entry.record_type_log:
            new_counts[rtype] += 1
        entry.record_type_counts = new_counts

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary dict for dashboard / health-check display."""
        with self._lock:
            return {
                "live_keys": len(self._store),
                "tracked_zones": len(self._zone_clients),
                "window_seconds": self._window_sec,
            }
