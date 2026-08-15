"""
feedback_logger.py — DNS Tunneling Detector
Analyst feedback logger for the retraining data pipeline.

Design doc reference: dnstds §9 — Retraining / Feedback Loop

Responsibilities:
  - Record analyst interactions (Confirm / Dismiss) to a daily JSONL file:
      {feedback_log_dir}/feedback_metrics_YYYY_MM_DD.jsonl
  - Each record stores the full event context needed for retraining:
      event_id, timestamp, label, query_name, record_type, source_ip,
      base_features (Tier-1), stateful_features (Tier-2), shap_attributions
  - Thread-safe: uses per-file locks so multiple workers can write concurrently.

Labels:
  CONFIRMED_CRITICAL  — analyst confirmed this was real DNS tunnelling
  DISMISSED_FP        — analyst dismissed as a false positive

These labels feed the retraining loop:
  Tier-1: DISMISSED_FP entries added as hard-negative benign samples
  Tier-2: CONFIRMED_CRITICAL entries added as fresh positive samples
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from config import get_config, FeedbackConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

LABEL_CONFIRMED = "CONFIRMED_CRITICAL"
LABEL_DISMISSED = "DISMISSED_FP"
VALID_LABELS = {LABEL_CONFIRMED, LABEL_DISMISSED}


# ---------------------------------------------------------------------------
# FeedbackLogger
# ---------------------------------------------------------------------------

class FeedbackLogger:
    """
    Append-mode daily JSONL feedback logger.

    One file is written per calendar day (UTC):
        {feedback_log_dir}/feedback_metrics_YYYY_MM_DD.jsonl

    Each line is a JSON object with the full event context for retraining.

    Usage:
        logger = FeedbackLogger()

        # From a SOC dashboard action:
        logger.log_feedback(
            event_id="abc-123",
            label=LABEL_DISMISSED,
            query_name="sub.cdn.example.com",
            record_type="TXT",
            source_ip="10.0.0.5",
            base_features={...},
            stateful_features={...},
            shap_attributions=[...],
        )
    """

    def __init__(self, cfg: Optional[FeedbackConfig] = None) -> None:
        self._cfg = cfg or get_config().feedback
        self._log_dir: Path = Path(self._cfg.feedback_log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Per-file write locks to prevent concurrent write corruption
        self._file_locks: Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()  # protects _file_locks dict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_feedback(
        self,
        event_id: str,
        label: str,
        query_name: str = "",
        record_type: str = "",
        source_ip: str = "",
        tier1_score: float = 0.0,
        tier2_score: float = 0.0,
        base_features: Optional[dict] = None,
        stateful_features: Optional[dict] = None,
        shap_attributions: Optional[List[dict]] = None,
        extra: Optional[dict] = None,
    ) -> bool:
        """
        Append one analyst feedback record to today's JSONL file.

        Args:
            event_id:           UUID of the original alert event.
            label:              CONFIRMED_CRITICAL or DISMISSED_FP.
            query_name:         Full FQDN of the DNS query.
            record_type:        DNS QTYPE string ("A", "TXT", …).
            source_ip:          Source IP of the DNS client.
            tier1_score:        Tier-1 probability score.
            tier2_score:        Tier-2 probability score.
            base_features:      Tier-1 stateless feature dict.
            stateful_features:  Tier-2 stateful feature dict.
            shap_attributions:  Top-N TreeSHAP attribution list.
            extra:              Optional dict of additional fields.

        Returns:
            True on success, False on error.
        """
        if label not in VALID_LABELS:
            logger.error(
                "log_feedback: invalid label %r — must be one of %s",
                label, VALID_LABELS,
            )
            return False

        record = {
            "event_id":          event_id,
            "ts":                time.time(),
            "date_utc":          time.strftime("%Y-%m-%d", time.gmtime()),
            "label":             label,
            # Network context
            "query_name":        query_name,
            "record_type":       record_type,
            "source_ip":         source_ip,
            # Model scores
            "tier1_score":       tier1_score,
            "tier2_score":       tier2_score,
            # Feature vectors (full — needed for retraining)
            "base_features":     base_features or {},
            "stateful_features": stateful_features or {},
            # XAI explanations
            "shap_attributions": shap_attributions or [],
        }
        if extra:
            record.update(extra)

        return self._write_record(record)

    def log_from_queue_event(
        self,
        event: dict,
        tier2_result: dict,
        label: str,
    ) -> bool:
        """
        Convenience wrapper: extract fields from a raw queue event + Tier-2
        result dict as produced by Tier2ConsumerWorker._route_outcome().

        This is the signature that replaces _default_feedback_callback in
        queue_consumer.py.
        """
        return self.log_feedback(
            event_id=event.get("event_id", ""),
            label=label,
            query_name=event.get("query_name", ""),
            record_type=event.get("record_type", ""),
            source_ip=event.get("source_ip", ""),
            tier1_score=float(event.get("tier1_score", 0.0)),
            tier2_score=float(tier2_result.get("tier2_score", 0.0)),
            base_features=event.get("base_features", {}),
            stateful_features=tier2_result.get("features", {}),
            shap_attributions=tier2_result.get("shap_attributions", []),
        )

    # ------------------------------------------------------------------
    # File path & per-file locking
    # ------------------------------------------------------------------

    def _today_path(self) -> Path:
        """Return the path for today's feedback JSONL file (UTC date)."""
        date_str = time.strftime("%Y_%m_%d", time.gmtime())
        return self._log_dir / f"feedback_metrics_{date_str}.jsonl"

    def _get_file_lock(self, path: Path) -> threading.Lock:
        """Return (and lazily create) the per-file write lock."""
        key = str(path)
        with self._meta_lock:
            if key not in self._file_locks:
                self._file_locks[key] = threading.Lock()
            return self._file_locks[key]

    def _write_record(self, record: dict) -> bool:
        """Append one JSON record to today's file. Thread-safe."""
        path = self._today_path()
        lock = self._get_file_lock(path)
        line = json.dumps(record, ensure_ascii=False) + "\n"

        try:
            with lock:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line)
            logger.debug(
                "Feedback logged: event_id=%s  label=%s  file=%s",
                record.get("event_id"), record.get("label"), path.name,
            )
            return True
        except OSError:
            logger.exception(
                "FeedbackLogger: failed to write to %s", path
            )
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def list_feedback_files(self) -> List[Path]:
        """Return all feedback JSONL files sorted by date (oldest first)."""
        return sorted(self._log_dir.glob("feedback_metrics_*.jsonl"))

    def count_records_in_file(self, path: Path) -> int:
        """Count the number of JSON records in one feedback file."""
        try:
            with open(path, encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        except OSError:
            return 0

    def stats(self) -> dict:
        """Return summary stats across all feedback files."""
        files = self.list_feedback_files()
        total = sum(self.count_records_in_file(f) for f in files)
        return {
            "feedback_files": len(files),
            "total_records":  total,
            "log_dir":        str(self._log_dir),
        }


# ---------------------------------------------------------------------------
# queue_consumer.py callback adapter
# ---------------------------------------------------------------------------

def make_feedback_callback(feedback_logger: FeedbackLogger):
    """
    Return a feedback callback compatible with Tier2ConsumerWorker's
    feedback_callback argument.  Replaces _default_feedback_callback.

    Args:
        feedback_logger: Live FeedbackLogger instance.

    Returns:
        Callable[[dict], None]
    """
    def _callback(record: dict) -> None:
        event = {k: v for k, v in record.items() if k != "tier2_result"}
        tier2_result = record.get("tier2_result", {})
        raw_label = record.get("label", "DISMISSED_FP")
        # Map queue_consumer labels to feedback schema labels
        label_map = {
            "FALSE_POSITIVE_ANOMALY": LABEL_DISMISSED,
            "CONFIRMED_CRITICAL":     LABEL_CONFIRMED,
            LABEL_DISMISSED:          LABEL_DISMISSED,
            LABEL_CONFIRMED:          LABEL_CONFIRMED,
        }
        label = label_map.get(raw_label, LABEL_DISMISSED)
        feedback_logger.log_from_queue_event(event, tier2_result, label)

    return _callback
