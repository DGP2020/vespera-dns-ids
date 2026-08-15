"""
alert_dispatcher.py — DNS Tunneling Detector
Resilient SOC webhook alert dispatcher.

Design doc reference: dnstds §11 — SOC Alert Dispatcher
                       dnstds §11.5 — XAI payload inclusion

Responsibilities:
  - Accept alert payloads from Tier-1 (fast-track) and Tier-2 (verified critical).
  - Stage every payload in the SQLite WAL buffer BEFORE attempting HTTP delivery.
  - Sign requests with Bearer token or HMAC-SHA256 (X-Signature-SHA256).
  - Retry with exponential backoff + jitter:
      delay = min(retry_max_sec, retry_base_sec * 2^retry_count) ± jitter
  - Implement a per-endpoint circuit breaker that suspends delivery on
    repeated failures rather than stacking retries.
  - After MAX_RETRIES failures: promote to DLQ via AlertStorage.increment_retry().

The dispatcher runs in a background thread (dispatch loop) that periodically
re-attempts PENDING alerts from the SQLite buffer.  enqueue_alert() is the
non-blocking call used by Tier-1 and Tier-2 consumers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import random
import threading
import time
from enum import Enum
from typing import Optional

try:
    import requests  # type: ignore[import-untyped]
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    requests = None  # type: ignore[assignment]

from alert_storage import AlertStorage, STATUS_DELIVERED, STATUS_PENDING
from config import get_config, AlertConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    CLOSED = "CLOSED"    # Normal operation
    OPEN   = "OPEN"      # Suspended — too many failures
    HALF   = "HALF_OPEN" # Probe allowed — one trial delivery


class CircuitBreaker:
    """
    Per-endpoint circuit breaker.

    Transitions:
        CLOSED → OPEN  after `threshold` consecutive failures
        OPEN   → HALF  after `reset_timeout_sec` seconds
        HALF   → CLOSED  if next delivery succeeds
        HALF   → OPEN    if next delivery fails again

    When OPEN, all delivery attempts for this endpoint are skipped immediately.
    """

    def __init__(self, threshold: int = 3, reset_timeout_sec: int = 60) -> None:
        self.threshold = threshold
        self.reset_timeout_sec = reset_timeout_sec
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """Return True if a delivery attempt is allowed right now."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if (time.time() - (self._opened_at or 0)) >= self.reset_timeout_sec:
                    self._state = CircuitState.HALF
                    logger.info("Circuit breaker → HALF_OPEN (probe trial).")
                    return True
                return False
            # HALF_OPEN — allow exactly one probe
            return True

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF or self._failure_count >= self.threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                logger.warning(
                    "Circuit breaker → OPEN after %d failure(s). "
                    "Deliveries suspended for %ds.",
                    self._failure_count, self.reset_timeout_sec,
                )

    @property
    def state(self) -> CircuitState:
        return self._state


# ---------------------------------------------------------------------------
# Backoff helper
# ---------------------------------------------------------------------------

def compute_backoff(
    retry_count: int,
    base_sec: int = 2,
    max_sec: int = 300,
    jitter_range: float = 2.0,
) -> float:
    """
    Exponential backoff with uniform jitter (dnstds §11):
        delay = min(max_sec, base_sec * 2^retry_count) ± jitter

    Args:
        retry_count:  Number of failed attempts so far (0-indexed).
        base_sec:     Base delay in seconds (default 2).
        max_sec:      Maximum delay cap in seconds (default 300 = 5 min).
        jitter_range: Maximum ± jitter in seconds (default ±2s).

    Returns:
        Delay in seconds (float).
    """
    raw = base_sec * math.pow(2, retry_count)
    capped = min(max_sec, raw)
    jitter = random.uniform(-jitter_range, jitter_range)
    return max(0.0, capped + jitter)


# ---------------------------------------------------------------------------
# AlertDispatcher
# ---------------------------------------------------------------------------

class AlertDispatcher:
    """
    Resilient SOC webhook alert dispatcher.

    Lifecycle:
        1. Call enqueue_alert(payload) from Tier-1 or Tier-2 — non-blocking.
        2. The background dispatch_loop() reads PENDING alerts from SQLite
           and attempts HTTP POST delivery.
        3. On failure, increments retry_count (with backoff).
        4. On success, marks DELIVERED.
        5. After MAX_RETRIES failures, promotes to DLQ.

    Thread safety:
        enqueue_alert() is safe to call from any thread.
        dispatch_loop() should run in exactly one background thread.
    """

    DISPATCH_POLL_INTERVAL_SEC: float = 2.0  # how often the loop polls SQLite
    BATCH_SIZE: int = 50                      # alerts per poll cycle

    def __init__(self, cfg: Optional[AlertConfig] = None) -> None:
        self._cfg = cfg or get_config().alert
        self._storage = AlertStorage(self._cfg)
        self._circuit = CircuitBreaker(
            threshold=self._cfg.circuit_open_threshold,
            reset_timeout_sec=60,
        )
        self._stop_event = threading.Event()
        self._dispatch_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API (called by Tier-1 and Tier-2)
    # ------------------------------------------------------------------

    def enqueue_alert(self, payload: dict) -> bool:
        """
        Non-blocking enqueue: stage the alert payload in the SQLite buffer.
        The background thread handles actual HTTP delivery.

        Args:
            payload: Alert dict (must include 'event_id' and 'auth_type').

        Returns:
            True if successfully staged, False on error.
        """
        return self._storage.store_alert(payload)

    def enqueue_enrichment(self, enrichment: dict) -> bool:
        """
        Enqueue a Tier-2 AUDIT_ENRICHMENT payload.  These are flagged with
        alert_type='TIER2_ENRICHMENT' so the downstream consumer can append
        to the already-open incident rather than create a new one.
        """
        if "alert_type" not in enrichment:
            enrichment = {**enrichment, "alert_type": "TIER2_ENRICHMENT"}
        return self._storage.store_alert(enrichment)

    # ------------------------------------------------------------------
    # Background dispatch loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background dispatch thread."""
        if self._dispatch_thread and self._dispatch_thread.is_alive():
            logger.warning("AlertDispatcher.start() called but already running.")
            return

        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="alert-dispatcher",
            daemon=True,
        )
        self._dispatch_thread.start()
        logger.info("AlertDispatcher background thread started.")

    def stop(self) -> None:
        """Signal the dispatch loop to exit."""
        self._stop_event.set()
        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=10)

    def _dispatch_loop(self) -> None:
        """
        Background loop: poll SQLite for PENDING alerts and attempt delivery.
        Respects circuit breaker and per-alert backoff state.
        """
        logger.info("AlertDispatcher dispatch loop running.")
        while not self._stop_event.is_set():
            try:
                if not self._circuit.allow_request():
                    logger.debug(
                        "Circuit OPEN — skipping dispatch cycle. state=%s",
                        self._circuit.state,
                    )
                    time.sleep(self.DISPATCH_POLL_INTERVAL_SEC)
                    continue

                pending = self._storage.fetch_pending_alerts(limit=self.BATCH_SIZE)
                for alert in pending:
                    self._attempt_delivery(alert)

            except Exception:  # noqa: BLE001
                logger.exception("AlertDispatcher loop error — continuing.")

            time.sleep(self.DISPATCH_POLL_INTERVAL_SEC)

        logger.info("AlertDispatcher dispatch loop stopped.")

    # ------------------------------------------------------------------
    # HTTP delivery
    # ------------------------------------------------------------------

    def _attempt_delivery(self, alert: dict) -> None:
        """
        Attempt a single HTTP POST delivery of one alert.
        Updates retry count and circuit breaker state.
        """
        if not _REQUESTS_AVAILABLE:
            logger.error("requests library not installed — cannot deliver alerts.")
            return

        webhook_url = self._cfg.webhook_url
        if not webhook_url:
            logger.warning("DNS_ALERT_WEBHOOK_URL not configured — alert not sent.")
            return

        event_id: str = alert["event_id"]
        retry_count: int = alert["retry_count"]
        payload: dict = alert["payload"]

        # Honour backoff — skip if not enough time has elapsed since last attempt
        if retry_count > 0:
            wait = compute_backoff(
                retry_count - 1,
                base_sec=self._cfg.retry_base_sec,
                max_sec=self._cfg.retry_max_sec,
            )
            last_attempt = alert.get("updated_at", 0)
            elapsed = time.time() - last_attempt
            if elapsed < wait:
                logger.debug(
                    "event %s: backoff %.1fs remaining — skipping this cycle.",
                    event_id, wait - elapsed,
                )
                return

        headers = self._build_headers(payload)
        body = json.dumps(payload)

        try:
            resp = requests.post(
                webhook_url,
                data=body,
                headers=headers,
                timeout=10,
            )
            if resp.status_code < 400:
                self._storage.update_alert_status(event_id, STATUS_DELIVERED)
                self._circuit.record_success()
                logger.info(
                    "Alert delivered: event=%s  status=%d  retry=%d",
                    event_id, resp.status_code, retry_count,
                )
            else:
                raise requests.HTTPError(
                    f"HTTP {resp.status_code}", response=resp
                )

        except Exception as exc:  # noqa: BLE001
            self._circuit.record_failure()
            new_retry = self._storage.increment_retry(event_id)
            logger.warning(
                "Alert delivery failed: event=%s  retry=%d  error=%s",
                event_id, new_retry, exc,
            )

    # ------------------------------------------------------------------
    # Auth header builder (dnstds §11)
    # ------------------------------------------------------------------

    def _build_headers(self, payload: dict) -> dict[str, str]:
        """
        Build HTTP headers for the outbound alert request.

        Auth modes (dnstds §11):
          bearer → Authorization: Bearer <token>
          hmac   → X-Signature-SHA256: sha256=<hex>  (HMAC over JSON body)
        """
        body = json.dumps(payload).encode("utf-8")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }

        auth_type = self._cfg.auth_type.lower()

        if auth_type == "bearer":
            token = self._cfg.bearer_token
            if token:
                headers["Authorization"] = f"Bearer {token}"
            else:
                logger.warning("Bearer auth selected but DNS_ALERT_BEARER_TOKEN is empty.")

        elif auth_type == "hmac":
            secret = self._cfg.hmac_secret.encode("utf-8")
            if secret:
                sig = hmac.new(secret, body, hashlib.sha256).hexdigest()
                headers["X-Signature-SHA256"] = f"sha256={sig}"
            else:
                logger.warning("HMAC auth selected but DNS_ALERT_HMAC_SECRET is empty.")

        else:
            logger.warning("Unknown auth_type %r — sending unsigned request.", auth_type)

        return headers

    # ------------------------------------------------------------------
    # Health / diagnostics
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Return current dispatcher health state for dashboard display."""
        return {
            "circuit_state":         self._circuit.state.value,
            "alert_dispatch_dropped_total": __import__("alert_storage").get_dropped_total(),
            "queue_by_status":       self._storage.count_by_status(),
            "webhook_url_configured": bool(self._cfg.webhook_url),
        }
