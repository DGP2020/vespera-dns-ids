"""
redis_queue.py — DNS Tunneling Detector
Redis-backed distributed queue provider (Phase 2 topology).

Design doc reference: dnstds §6 — Queue / Backpressure
                       dnstds §8 — Distributed scaling (Phase 2)

Implements BaseQueueBroker over a Redis LIST:
  - RPUSH  (put)      — producer side
  - BLPOP  (get)      — consumer side, blocking with timeout
  - OOM interception  — catches ResponseError on RPUSH and falls back to
                         the same disk ring-buffer log used by MemoryQueueProvider

Configuration (environment variables):
  DNS_REDIS_HOST           default: 127.0.0.1
  DNS_REDIS_PORT           default: 6379
  DNS_REDIS_QUEUE_NAME     default: dns_triage_queue
  DNS_REDIS_MAX_MEMORY_MB  default: 512  (applied via CONFIG SET maxmemory)

Local testing (no Redis installed):
  The provider detects that redis-py is not installed and raises a clear
  ImportError with install instructions.

  To run Redis locally:
    Windows (WSL2):   wsl redis-server
    Windows (native): https://github.com/microsoftarchive/redis/releases
    macOS:            brew install redis && redis-server
    Docker:           docker run -p 6379:6379 redis:7-alpine
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from config import get_config, QueueConfig
from queue_broker import BaseQueueBroker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful redis import
# ---------------------------------------------------------------------------

try:
    import redis as redis_lib  # type: ignore[import-untyped]
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    redis_lib = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# RedisQueueProvider
# ---------------------------------------------------------------------------

class RedisQueueProvider(BaseQueueBroker):
    """
    Redis LIST-backed queue provider (drop-in replacement for MemoryQueueProvider).

    Implements the same BaseQueueBroker interface so no producer or consumer
    code needs to change when swapping from MemoryQueueProvider.

    Queue layout:
        One Redis LIST key per queue name (default: "dns_triage_queue").
        Producers use RPUSH (append to tail).
        Consumers use BLPOP (blocking pop from head, with timeout).

    OOM / memory cap:
        On startup, sets Redis maxmemory to DNS_REDIS_MAX_MEMORY_MB and
        maxmemory-policy to noeviction.  When RPUSH gets a ResponseError
        (OOM), the event is written to the overflow disk log and dropped,
        matching the MemoryQueueProvider overflow behaviour.

    Reconnect:
        On connection error during get(), the provider sleeps briefly and
        returns None (consumer loop will retry on next poll).
    """

    OVERFLOW_METRIC = "TIER2_QUEUE_OVERFLOW_DROP"

    def __init__(self, cfg: Optional[QueueConfig] = None) -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "redis-py is not installed. Install it with:\n"
                "    pip install redis\n"
                "Then start a local Redis instance:\n"
                "    docker run -p 6379:6379 redis:7-alpine"
            )
        self._cfg = cfg or get_config().queue
        self._queue_name: str = self._cfg.redis_queue_name
        self._overflow_log: Path = Path(self._cfg.overflow_log_path)
        self._overflow_log.parent.mkdir(parents=True, exist_ok=True)
        self._overflow_count: int = 0

        # Build the Redis client with authentication & TLS support
        self._client = redis_lib.Redis(
            host=self._cfg.redis_host,
            port=self._cfg.redis_port,
            password=self._cfg.redis_password or None,
            ssl=self._cfg.redis_ssl,
            decode_responses=False,  # we handle JSON encoding ourselves
            socket_connect_timeout=5,
            socket_timeout=10,
            retry_on_timeout=True,
        )
        self._configure_redis()

    # ------------------------------------------------------------------
    # Startup configuration
    # ------------------------------------------------------------------

    def _configure_redis(self) -> None:
        """
        Attempt to verify connection and apply memory limits if allowed.
        Tolerates restricted CONFIG permissions and connection delays.
        """
        try:
            self._client.ping()
            max_mem = self._cfg.redis_max_memory_mb
            try:
                self._client.config_set("maxmemory", f"{max_mem}mb")
                self._client.config_set("maxmemory-policy", "noeviction")
            except redis_lib.exceptions.ResponseError as cfg_err:
                logger.info(
                    "Redis CONFIG SET not permitted or disabled by server policy (%s) — skipping runtime memory cap.",
                    cfg_err,
                )
            logger.info(
                "RedisQueueProvider connected to %s:%d (ssl=%s)  queue=%r",
                self._cfg.redis_host, self._cfg.redis_port,
                self._cfg.redis_ssl, self._queue_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RedisQueueProvider: could not connect to Redis at %s:%d — %s. "
                "Ensure Redis is running before starting workers.",
                self._cfg.redis_host, self._cfg.redis_port, exc,
            )

    # ------------------------------------------------------------------
    # BaseQueueBroker interface
    # ------------------------------------------------------------------

    def put(self, event: dict) -> bool:
        """
        Serialize `event` to JSON and RPUSH to the Redis queue.

        On OOM (ResponseError) or connection failure: drop-and-log to disk.

        Returns:
            True on success, False if the event was dropped.
        """
        payload = json.dumps(event, ensure_ascii=False).encode("utf-8")
        try:
            self._client.rpush(self._queue_name, payload)
            return True
        except redis_lib.exceptions.ResponseError as exc:
            # OOM — Redis refused the write
            logger.warning(
                "%s — Redis OOM: %s  event_id=%s",
                self.OVERFLOW_METRIC, exc, event.get("event_id"),
            )
            self.on_overflow(event)
            self._write_overflow_log(event)
            return False
        except redis_lib.exceptions.ConnectionError as exc:
            logger.error("RedisQueueProvider.put: connection error — %s", exc)
            self.on_overflow(event)
            self._write_overflow_log(event)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("RedisQueueProvider.put: unexpected error — %s", exc)
            self.on_overflow(event)
            self._write_overflow_log(event)
            return False

    def get(self, timeout: float = 0.1) -> Optional[dict]:
        """
        BLPOP from the Redis queue, blocking for up to `timeout` seconds.

        Returns:
            Deserialized event dict, or None on timeout / connection error.
        """
        timeout_int = max(1, int(timeout))  # BLPOP requires integer seconds
        try:
            result = self._client.blpop(self._queue_name, timeout=timeout_int)
            if result is None:
                return None  # timeout
            _key, raw = result
            return json.loads(raw.decode("utf-8"))
        except redis_lib.exceptions.ConnectionError as exc:
            logger.warning("RedisQueueProvider.get: connection error — %s  (will retry)", exc)
            time.sleep(0.5)
            return None
        except json.JSONDecodeError as exc:
            logger.warning("RedisQueueProvider.get: malformed JSON payload — %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("RedisQueueProvider.get: unexpected error — %s", exc)
            return None

    def qsize(self) -> int:
        """Return current queue depth (LLEN). Returns 0 on connection error."""
        try:
            return int(self._client.llen(self._queue_name))
        except Exception:  # noqa: BLE001
            return 0

    def close(self) -> None:
        """Close the Redis connection pool."""
        try:
            self._client.close()
            logger.info("RedisQueueProvider closed.")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Overflow / OOM disk fallback
    # ------------------------------------------------------------------

    def _write_overflow_log(self, event: dict) -> None:
        """
        Write a dropped event to the disk ring-buffer overflow log.
        Same format as MemoryQueueProvider to allow unified replay tooling.
        """
        self._overflow_count += 1
        entry = {
            "ts":              time.time(),
            "overflow_count":  self._overflow_count,
            "event_id":        event.get("event_id", ""),
            "query_name":      event.get("query_name", ""),
            "source_ip":       event.get("source_ip", ""),
            "tier1_score":     event.get("tier1_score", 0.0),
        }
        try:
            with open(self._overflow_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            logger.debug(
                "RedisQueueProvider: overflow log write failed (disk full?)", exc_info=True
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return health / diagnostic stats."""
        return {
            "provider":       "redis",
            "host":           self._cfg.redis_host,
            "port":           self._cfg.redis_port,
            "queue_name":     self._queue_name,
            "queue_depth":    self.qsize(),
            "overflow_drops": self._overflow_count,
        }


# ---------------------------------------------------------------------------
# Factory helper (used by main.py when Redis mode is configured)
# ---------------------------------------------------------------------------

def create_redis_provider(cfg: Optional[QueueConfig] = None) -> RedisQueueProvider:
    """
    Create a RedisQueueProvider, raising a clear error with install
    instructions if redis-py or a Redis server is unavailable.
    """
    if not _REDIS_AVAILABLE:
        raise ImportError(
            "redis-py is not installed.\n"
            "Install: pip install redis\n"
            "Start Redis locally:\n"
            "  Docker:  docker run -p 6379:6379 redis:7-alpine\n"
            "  WSL2:    wsl redis-server\n"
            "  macOS:   brew install redis && redis-server\n"
        )
    return RedisQueueProvider(cfg)
