"""
queue_consumer.py — DNS Tunneling Detector
Tier-2 async queue consumer worker.

Design doc reference: dnstds §4 — Tier-2 Deep Analysis

Responsibilities:
  1. Poll the BaseQueueBroker for incoming triage events from Tier-1.
  2. Update the 30-second StateCache with the event's DNS metadata.
  3. Extract stateful features (shared-base + incremental windowed).
  4. Score via Tier2Classifier within the ≤200ms latency budget.
  5. Compute TreeSHAP feature attributions (dnstds §11.5).
  6. Route outcome:
       VERIFIED_CRITICAL      → build enriched alert payload → alert_dispatcher
       FALSE_POSITIVE_ANOMALY → emit to feedback_logger, suppress alert
     FAST_TRACK AUDIT_ENRICHMENT copies → append SHAP report to open incident,
       never create a duplicate alert.

Latency budget: ≤200ms from queue placement to verdict (dnstds §4).

Runs as a daemon sub-process spawned by main.py. Multiple instances may
run in parallel (tier2_worker_count from config) — each gets its own
StateCache and Tier2Classifier.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from config import get_config, AppConfig
from queue_broker import BaseQueueBroker
from state_cache import StateCache
from tier2_analysis import (
    Tier2Classifier,
    extract_parent_domain,
    extract_stateful_features,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert / feedback callback type aliases
# ---------------------------------------------------------------------------

# Called with a complete enriched alert payload dict for VERIFIED_CRITICAL events
AlertDispatchCallback = Callable[[dict], None]

# Called with the full event + result dict for FALSE_POSITIVE_ANOMALY events
FeedbackCallback = Callable[[dict], None]


# ---------------------------------------------------------------------------
# Tier-2 consumer worker
# ---------------------------------------------------------------------------

class Tier2ConsumerWorker:
    """
    Continuously drains the triage queue and runs deep stateful analysis.

    Designed to run inside a multiprocessing.Process (see main.py).
    All state (cache, classifier) is local to this process — no shared
    memory needed between worker instances.

    Maintenance thread:
        A background thread periodically calls StateCache.evict_stale_keys()
        every EVICTION_INTERVAL_SEC seconds to prevent unbounded memory growth.
    """

    EVICTION_INTERVAL_SEC: int = 5  # pruning cadence for expired cache keys

    def __init__(
        self,
        queue_broker: BaseQueueBroker,
        worker_id: int = 0,
        alert_callback: Optional[AlertDispatchCallback] = None,
        feedback_callback: Optional[FeedbackCallback] = None,
        cfg: Optional[AppConfig] = None,
    ) -> None:
        self._queue = queue_broker
        self._worker_id = worker_id
        self._alert_callback = alert_callback or _default_alert_callback
        self._feedback_callback = feedback_callback or _default_feedback_callback
        self._cfg = cfg or get_config()

        # Per-worker stateful resources
        self._state_cache = StateCache(self._cfg.tier2)
        self._classifier = Tier2Classifier(self._cfg.tier2)

        self._stop_event = threading.Event()
        self._processed: int = 0
        self._errors: int = 0

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Blocking main loop. Exits when stop() is called or the process
        receives SIGTERM.

        Launches the cache eviction maintenance thread before entering
        the consume loop.
        """
        logger.info(
            "Tier-2 consumer worker %d starting (pid=%d).",
            self._worker_id, os.getpid(),
        )

        # Start background cache maintenance thread
        eviction_thread = threading.Thread(
            target=self._eviction_loop,
            name=f"t2-evict-{self._worker_id}",
            daemon=True,
        )
        eviction_thread.start()

        # Main consume loop
        poll_timeout = self._cfg.queue.consumer_poll_timeout_sec
        try:
            while not self._stop_event.is_set():
                event = self._queue.get(timeout=poll_timeout)
                if event is None:
                    continue  # timeout — loop back and check stop_event
                self._process_event(event)
        except KeyboardInterrupt:
            pass
        finally:
            logger.info(
                "Tier-2 worker %d stopped — processed=%d  errors=%d",
                self._worker_id, self._processed, self._errors,
            )

    def stop(self) -> None:
        """Signal the run loop to exit cleanly."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Event processing pipeline
    # ------------------------------------------------------------------

    def _process_event(self, event: dict) -> None:
        """
        Full Tier-2 pipeline for a single queued event:
          StateCache.update → extract_stateful_features →
          Tier2Classifier.evaluate (≤200ms) → route outcome
        """
        t_start = time.perf_counter()
        event_id = event.get("event_id", "<unknown>")
        tier2_mode = event.get("tier2_mode", "TRIAGE")

        try:
            self._processed += 1

            # Step 1 — Extract routing metadata from payload
            query_name: str = event.get("query_name", "")
            record_type: str = event.get("record_type", "A")
            source_ip: str = event.get("source_ip", "")
            payload_length: float = float(
                event.get("base_features", {}).get("fqdn_length", len(query_name))
            )
            parent_domain: str = extract_parent_domain(query_name)

            # Step 2 — Update sliding window state cache
            self._state_cache.update(
                parent_domain=parent_domain,
                client_ip=source_ip,
                record_type=record_type,
                payload_length=payload_length,
            )

            # Step 3 — Extract the full 8-feature stateful vector
            base_features: dict = event.get("base_features", {})
            stateful_features = extract_stateful_features(
                base_features=base_features,
                state_cache=self._state_cache,
                parent_domain=parent_domain,
                client_ip=source_ip,
            )

            # Attach stateful features to the event payload so the classifier
            # can access them in one dict without re-passing arguments.
            event["stateful_features"] = stateful_features
            event["parent_domain"] = parent_domain

            # Step 4 — Score + compute TreeSHAP (within ≤200ms budget)
            result = self._classifier.evaluate(event)

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            result["total_pipeline_ms"] = elapsed_ms

            # Step 5 — Route outcome
            self._route_outcome(event, result)

        except Exception:  # noqa: BLE001
            self._errors += 1
            logger.exception("Tier-2 worker %d: unhandled exception on event %s.",
                             self._worker_id, event_id)

    def _route_outcome(self, event: dict, result: dict) -> None:
        """
        Apply the Tier-2 outcome routing rules from dnstds §5:

        - VERIFIED_CRITICAL (TRIAGE mode):
            Single enriched alert with SHAP breakdown → alert_callback.
        - FAST_TRACK AUDIT_ENRICHMENT copy (tier2_mode=AUDIT_ENRICHMENT):
            Append SHAP report to already-open incident — never fire new alert.
        - FALSE_POSITIVE_ANOMALY:
            Route to feedback stream for retraining — no alert.
        """
        verdict = result["verdict"]
        tier2_mode = event.get("tier2_mode", "TRIAGE")
        event_id = result["event_id"]

        if verdict == "VERIFIED_CRITICAL" and tier2_mode == "TRIAGE":
            # Confirmed anomaly from the SUSPICIOUS queue path
            alert_payload = self._build_alert_payload(event, result)
            logger.info(
                "Tier-2 worker %d: VERIFIED_CRITICAL → dispatching alert  "
                "event=%s  score=%.4f",
                self._worker_id, event_id, result["tier2_score"],
            )
            try:
                self._alert_callback(alert_payload)
            except Exception:  # noqa: BLE001
                logger.exception("alert_callback raised for event %s.", event_id)

        elif tier2_mode == "AUDIT_ENRICHMENT":
            # Fast-track copy — enrich the already-open incident (audit trail only).
            # Never fire a new alert, regardless of verdict.
            logger.info(
                "Tier-2 worker %d: AUDIT_ENRICHMENT  event=%s  "
                "shap_features=%d  verdict=%s",
                self._worker_id, event_id,
                len(result.get("shap_attributions", [])),
                verdict,
            )
            enrichment_payload = {
                "event_id":          event_id,
                "audit_type":        "TIER2_ENRICHMENT",
                "tier2_verdict":     verdict,
                "tier2_score":       result["tier2_score"],
                "shap_attributions": result["shap_attributions"],
                "stateful_features": result["features"],
                "latency_ms":        result["latency_ms"],
            }
            # Append enrichment to open incident via callback
            # (alert_dispatcher handles the append-not-create logic)
            try:
                self._alert_callback(enrichment_payload)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "alert_callback raised during AUDIT_ENRICHMENT for event %s.",
                    event_id,
                )

        else:
            # FALSE_POSITIVE_ANOMALY — no alert; route to feedback log
            logger.info(
                "Tier-2 worker %d: FALSE_POSITIVE_ANOMALY  event=%s  "
                "score=%.4f → feedback logger",
                self._worker_id, event_id, result["tier2_score"],
            )
            feedback_record = {
                **event,
                "tier2_result": result,
                "label": "FALSE_POSITIVE_ANOMALY",
            }
            try:
                self._feedback_callback(feedback_record)
            except Exception:  # noqa: BLE001
                logger.exception("feedback_callback raised for event %s.", event_id)

    # ------------------------------------------------------------------
    # Alert payload builder
    # ------------------------------------------------------------------

    def _build_alert_payload(self, event: dict, result: dict) -> dict:
        """
        Build the enriched SOC alert payload for a VERIFIED_CRITICAL event.
        Matches the SOC Alert Dispatcher payload schema (dnstds §11).
        """
        return {
            # --- Identity ---
            "event_id":          result["event_id"],
            "dispatch_timestamp": time.time(),
            "alert_type":        "VERIFIED_CRITICAL",
            # --- Threat classification ---
            "tier1_score":       event.get("tier1_score", 0.0),
            "tier2_score":       result["tier2_score"],
            "tier2_verdict":     result["verdict"],
            # --- Network context ---
            "source_ip":         event.get("source_ip", ""),
            "dest_ip":           event.get("dest_ip", ""),
            "source_port":       event.get("source_port", 0),
            "dest_port":         event.get("dest_port", 53),
            "query_name":        event.get("query_name", ""),
            "record_type":       event.get("record_type", ""),
            "parent_domain":     event.get("parent_domain", ""),
            # --- XAI explanation (dnstds §11.5) ---
            "shap_attributions": result["shap_attributions"],
            # --- Feature vectors (for SOC analyst context) ---
            "base_features":     event.get("base_features", {}),
            "stateful_features": result["features"],
            # --- Timing ---
            "packet_ts":         event.get("packet_ts", 0.0),
            "tier2_latency_ms":  result["latency_ms"],
        }

    # ------------------------------------------------------------------
    # Cache eviction maintenance loop
    # ------------------------------------------------------------------

    def _eviction_loop(self) -> None:
        """
        Background thread: periodically prune stale state cache entries.
        Runs until the main stop_event fires.
        """
        while not self._stop_event.is_set():
            time.sleep(self.EVICTION_INTERVAL_SEC)
            evicted = self._state_cache.evict_stale_keys()
            if evicted:
                logger.debug(
                    "Tier-2 worker %d: evicted %d stale cache key(s).",
                    self._worker_id, evicted,
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def processed(self) -> int:
        return self._processed

    @property
    def errors(self) -> int:
        return self._errors


# ---------------------------------------------------------------------------
# Default no-op callbacks (replaced by real implementations at runtime)
# ---------------------------------------------------------------------------

def _default_alert_callback(payload: dict) -> None:
    """
    Placeholder alert callback — logs the alert at CRITICAL level.
    Replaced by alert_dispatcher.enqueue_alert() in Phase 4.
    """
    logger.critical(
        "ALERT [%s]  event=%s  score=%.4f  query=%s  shap_top=%s",
        payload.get("alert_type", "?"),
        payload.get("event_id", "?"),
        payload.get("tier2_score", 0.0),
        payload.get("query_name", "?"),
        payload.get("shap_attributions", [])[:1],
    )


def _default_feedback_callback(record: dict) -> None:
    """
    Default feedback callback — delegates to FeedbackLogger (Phase 6).
    Falls back to a plain JSONL stub if feedback_logger is unavailable.
    """
    try:
        from feedback_logger import FeedbackLogger, make_feedback_callback  # noqa: PLC0415
        _fl = FeedbackLogger()
        cb = make_feedback_callback(_fl)
        cb(record)
    except ImportError:
        # feedback_logger.py not available — write minimal stub entry
        log_path = Path("data/feedback/fp_stub.jsonl")
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "event_id":   record.get("event_id"),
                    "query_name": record.get("query_name"),
                    "label":      record.get("label"),
                    "ts":         time.time(),
                }) + "\n")
        except Exception:  # noqa: BLE001
            logger.debug("Default feedback callback stub: write failed.", exc_info=True)
