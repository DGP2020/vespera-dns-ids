"""
alert_storage.py — DNS Tunneling Detector
SQLite WAL-mode persistent alert buffer and Dead-Letter Queue (DLQ).

Design doc reference: dnstds §11 — SOC Alert Dispatcher

Responsibilities:
  - Stage every outbound alert in a local SQLite database (WAL mode) before
    delivery. Survives process restarts.
  - Track retry count, auth type, and full JSON payload per alert.
  - Promote alerts to FAILED_DLQ status after MAX_RETRIES failed attempts
    and write them to the dead-letter JSONL log.
  - Increment alert_dispatch_dropped_total counter on DLQ promotion.

Schema:  outbound_alerts
  event_id         TEXT PRIMARY KEY
  created_at       REAL    (Unix timestamp)
  updated_at       REAL
  retry_count      INTEGER
  status           TEXT    PENDING | SENDING | DELIVERED | FAILED_DLQ
  auth_type        TEXT    bearer | hmac
  payload          TEXT    JSON string (includes shap_attributions)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator, List, Optional

from config import get_config, AlertConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_PENDING = "PENDING"
STATUS_SENDING = "SENDING"
STATUS_DELIVERED = "DELIVERED"
STATUS_FAILED_DLQ = "FAILED_DLQ"

# ---------------------------------------------------------------------------
# Thread-safe dropped counter
# ---------------------------------------------------------------------------

_drop_lock = threading.Lock()
_dropped_total: int = 0  # alert_dispatch_dropped_total


def get_dropped_total() -> int:
    """Return the cumulative DLQ promotion count since process start."""
    with _drop_lock:
        return _dropped_total


def _increment_dropped() -> int:
    global _dropped_total
    with _drop_lock:
        _dropped_total += 1
        return _dropped_total


# ---------------------------------------------------------------------------
# AlertStorage
# ---------------------------------------------------------------------------

class AlertStorage:
    """
    SQLite WAL-mode persistent buffer for outbound SOC alerts.

    Usage:
        storage = AlertStorage()
        storage.store_alert(payload_dict)

        pending = storage.fetch_pending_alerts(limit=50)
        for alert in pending:
            # attempt delivery...
            if success:
                storage.update_alert_status(alert["event_id"], STATUS_DELIVERED)
            else:
                storage.increment_retry(alert["event_id"])
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS outbound_alerts (
        event_id    TEXT    PRIMARY KEY,
        created_at  REAL    NOT NULL,
        updated_at  REAL    NOT NULL,
        retry_count INTEGER NOT NULL DEFAULT 0,
        status      TEXT    NOT NULL DEFAULT 'PENDING',
        auth_type   TEXT    NOT NULL DEFAULT 'bearer',
        payload     TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_status ON outbound_alerts (status, updated_at);
    """

    def __init__(self, cfg: Optional[AlertConfig] = None) -> None:
        self._cfg = cfg or get_config().alert
        self._db_path: Path = Path(self._cfg.sqlite_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()  # per-thread SQLite connections
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the database file and schema if they don't exist yet."""
        with self._connection() as conn:
            conn.executescript(self.CREATE_TABLE_SQL)
        logger.info("AlertStorage initialised at %s (WAL mode).", self._db_path)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Yield a per-thread SQLite connection in WAL mode.
        Commits on clean exit, rolls back on exception.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn

        conn = self._local.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def store_alert(self, payload: dict) -> bool:
        """
        Persist a new alert in PENDING state.

        Args:
            payload: Full alert dict (must contain 'event_id' and 'auth_type').

        Returns:
            True on success, False if event_id already exists (idempotent).
        """
        event_id: str = payload.get("event_id", "")
        if not event_id:
            logger.error("store_alert: payload missing 'event_id' — skipping.")
            return False

        auth_type: str = payload.get("auth_type", self._cfg.auth_type)
        now = time.time()
        payload_json = json.dumps(payload)

        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO outbound_alerts
                        (event_id, created_at, updated_at, retry_count,
                         status, auth_type, payload)
                    VALUES (?, ?, ?, 0, ?, ?, ?)
                    """,
                    (event_id, now, now, STATUS_PENDING, auth_type, payload_json),
                )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("store_alert failed for event_id=%s", event_id)
            return False

    def update_alert_status(self, event_id: str, status: str) -> bool:
        """Update the delivery status of an alert."""
        try:
            with self._connection() as conn:
                conn.execute(
                    "UPDATE outbound_alerts SET status=?, updated_at=? WHERE event_id=?",
                    (status, time.time(), event_id),
                )
            return True
        except Exception:  # noqa: BLE001
            logger.exception("update_alert_status failed for event_id=%s", event_id)
            return False

    def increment_retry(self, event_id: str) -> int:
        """
        Increment retry_count for an alert and promote to FAILED_DLQ if
        retry_count reaches MAX_RETRIES. Writes to dead-letter log on DLQ.

        Returns:
            The new retry_count.
        """
        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    UPDATE outbound_alerts
                    SET retry_count = retry_count + 1, updated_at = ?
                    WHERE event_id = ?
                    """,
                    (time.time(), event_id),
                )
                row = conn.execute(
                    "SELECT retry_count, payload FROM outbound_alerts WHERE event_id=?",
                    (event_id,),
                ).fetchone()

            if row is None:
                return 0

            retry_count: int = row["retry_count"]
            payload_json: str = row["payload"]

            if retry_count >= self._cfg.max_retries:
                self._promote_to_dlq(event_id, payload_json, retry_count)

            return retry_count

        except Exception:  # noqa: BLE001
            logger.exception("increment_retry failed for event_id=%s", event_id)
            return -1

    def _promote_to_dlq(
        self, event_id: str, payload_json: str, retry_count: int
    ) -> None:
        """
        Mark alert as FAILED_DLQ, write to dead-letter JSONL log, and
        increment the alert_dispatch_dropped_total counter.
        """
        self.update_alert_status(event_id, STATUS_FAILED_DLQ)
        dropped_total = _increment_dropped()

        dlq_path = Path(self._cfg.dead_letter_log_path)
        dlq_entry = {
            "ts": time.time(),
            "event_id": event_id,
            "retry_count": retry_count,
            "alert_dispatch_dropped_total": dropped_total,
            "payload": json.loads(payload_json),
        }

        try:
            dlq_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dlq_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dlq_entry) + "\n")
            logger.error(
                "Alert %s promoted to DLQ after %d retries. "
                "alert_dispatch_dropped_total=%d  dlq=%s",
                event_id, retry_count, dropped_total, dlq_path,
            )
        except OSError:
            logger.exception(
                "Failed to write DLQ entry for event_id=%s to %s",
                event_id, dlq_path,
            )

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def fetch_pending_alerts(self, limit: int = 100) -> List[dict]:
        """
        Return up to `limit` alerts in PENDING status, ordered by created_at.
        """
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    """
                    SELECT event_id, created_at, retry_count, auth_type, payload
                    FROM outbound_alerts
                    WHERE status = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (STATUS_PENDING, limit),
                ).fetchall()

            return [
                {
                    "event_id":    row["event_id"],
                    "created_at":  row["created_at"],
                    "retry_count": row["retry_count"],
                    "auth_type":   row["auth_type"],
                    "payload":     json.loads(row["payload"]),
                }
                for row in rows
            ]
        except Exception:  # noqa: BLE001
            logger.exception("fetch_pending_alerts failed.")
            return []

    def count_by_status(self) -> dict[str, int]:
        """Return a dict of status → count for monitoring / dashboard."""
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) AS n FROM outbound_alerts GROUP BY status"
                ).fetchall()
            return {row["status"]: row["n"] for row in rows}
        except Exception:  # noqa: BLE001
            return {}

    def close(self) -> None:
        """Close the per-thread connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
