"""
router.py — DNS Tunneling Detector
Dual-threshold Tier-1 routing logic.

Design doc reference: dnstds §5 — Routing / Threshold Logic

Routing table:
  P < 0.70           → BENIGN        → dismiss, no further action
  0.70 ≤ P < 0.92   → SUSPICIOUS    → enqueue to Tier-2 async queue
  P ≥ 0.92           → FAST_TRACK    → fire alert immediately
                                        AND enqueue async copy to Tier-2
                                        (Tier-2 never gates this alert)

Important constraints from the design doc:
  - Tier-1 fast-track alert must be fired BEFORE the async queue enqueue.
  - Tier-2 enrichment on a fast-tracked event appends to the already-open
    incident view (audit trail); it never creates a duplicate alert.
  - The router itself does NOT call the alert dispatcher directly to keep
    the coupling loose. Instead it calls an optional alert_callback so the
    caller (packet_capture.py / main.py) injects the real dispatcher.
"""

from __future__ import annotations

import logging
import time
import uuid
from enum import Enum
from typing import Callable, Optional

from config import get_config, Tier1Config
from queue_broker import BaseQueueBroker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    """Tier-1 routing classification label."""
    BENIGN = "BENIGN"
    SUSPICIOUS = "SUSPICIOUS"
    FAST_TRACK = "FAST_TRACK"


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------

class RoutingResult:
    """
    Encapsulates the complete outcome of a single route_triage_result() call.

    Attributes:
        verdict:       One of BENIGN | SUSPICIOUS | FAST_TRACK.
        score:         Raw Tier-1 probability score.
        event_id:      UUID assigned to this event (stable across routing
                       and Tier-2 enrichment so duplicates are detectable).
        queued:        True if the event was successfully enqueued to Tier-2.
        fast_tracked:  True if an immediate alert was fired.
        ts:            Unix timestamp of routing.
    """

    def __init__(
        self,
        verdict: Verdict,
        score: float,
        event_id: str,
        queued: bool = False,
        fast_tracked: bool = False,
        ts: Optional[float] = None,
    ) -> None:
        self.verdict = verdict
        self.score = score
        self.event_id = event_id
        self.queued = queued
        self.fast_tracked = fast_tracked
        self.ts = ts or time.time()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RoutingResult("
            f"verdict={self.verdict.value}, "
            f"score={self.score:.4f}, "
            f"event_id={self.event_id}, "
            f"queued={self.queued}, "
            f"fast_tracked={self.fast_tracked})"
        )


# ---------------------------------------------------------------------------
# Alert callback type alias
# ---------------------------------------------------------------------------

# Signature: (event_payload: dict) -> None
# Injected by main.py / packet_capture.py to avoid circular imports.
AlertCallback = Callable[[dict], None]


# ---------------------------------------------------------------------------
# Core routing function
# ---------------------------------------------------------------------------

def route_triage_result(
    packet_metadata: dict,
    base_features: dict,
    score: float,
    queue_broker: BaseQueueBroker,
    alert_callback: Optional[AlertCallback] = None,
    cfg: Optional[Tier1Config] = None,
) -> RoutingResult:
    """
    Apply the dual-threshold routing table to a Tier-1 score and dispatch
    accordingly.

    Args:
        packet_metadata:  Raw packet fields: source_ip, dest_ip, port,
                          query_name, record_type, packet_ts.
        base_features:    Dict from tier1_triage.extract_stateless_features().
                          Packed into the queue payload so Tier-2 can skip
                          recomputing them (dnstds §4 shared-base pipeline).
        score:            Tier-1 probability score ∈ [0.0, 1.0].
        queue_broker:     Concrete BaseQueueBroker instance for enqueuing
                          Tier-2 work items.
        alert_callback:   Callable fired synchronously for FAST_TRACK events
                          BEFORE the async enqueue.  Pass None in unit tests.
        cfg:              Tier1Config override (uses global config if None).

    Returns:
        RoutingResult describing what was done.
    """
    _cfg = cfg or get_config().tier1
    event_id = str(uuid.uuid4())
    ts = time.time()

    verdict: Verdict
    queued = False
    fast_tracked = False

    if score < _cfg.p_benign_max:
        # ---------------------------------------------------------------
        # BENIGN — dismiss silently
        # ---------------------------------------------------------------
        verdict = Verdict.BENIGN
        logger.debug(
            "BENIGN  score=%.4f  event=%s  query=%s",
            score, event_id, packet_metadata.get("query_name"),
        )

    elif score < _cfg.p_fasttrack_min:
        # ---------------------------------------------------------------
        # SUSPICIOUS — enqueue for Tier-2 deep analysis (async)
        # ---------------------------------------------------------------
        verdict = Verdict.SUSPICIOUS
        event_payload = _build_event_payload(
            event_id, ts, verdict, score, packet_metadata, base_features
        )
        queued = queue_broker.put(event_payload)
        logger.debug(
            "SUSPICIOUS  score=%.4f  event=%s  query=%s  queued=%s",
            score, event_id, packet_metadata.get("query_name"), queued,
        )

    else:
        # ---------------------------------------------------------------
        # FAST_TRACK — fire alert immediately, THEN enqueue async copy
        # (dnstds §5: "Tier-2 never gates a fast-track alert")
        # ---------------------------------------------------------------
        verdict = Verdict.FAST_TRACK
        event_payload = _build_event_payload(
            event_id, ts, verdict, score, packet_metadata, base_features
        )

        # 1. Fire the alert first — alert_callback must be non-blocking /
        #    hand-off-to-dispatcher to preserve Tier-1 latency budget.
        if alert_callback is not None:
            try:
                alert_callback(event_payload)
                fast_tracked = True
            except Exception:  # noqa: BLE001
                logger.exception(
                    "alert_callback raised for FAST_TRACK event %s — "
                    "continuing with Tier-2 enqueue anyway.",
                    event_id,
                )
        else:
            logger.warning(
                "No alert_callback registered — FAST_TRACK event %s will "
                "not generate an immediate SOC alert.",
                event_id,
            )
            fast_tracked = False

        # 2. Enqueue async copy to Tier-2 for enrichment / audit trail.
        #    Mark payload so Tier-2 knows it is audit-only and must not
        #    send a duplicate alert.
        event_payload["tier2_mode"] = "AUDIT_ENRICHMENT"
        queued = queue_broker.put(event_payload)

        logger.info(
            "FAST_TRACK  score=%.4f  event=%s  query=%s  "
            "alert_fired=%s  tier2_queued=%s",
            score, event_id,
            packet_metadata.get("query_name"),
            fast_tracked, queued,
        )

    return RoutingResult(
        verdict=verdict,
        score=score,
        event_id=event_id,
        queued=queued,
        fast_tracked=fast_tracked,
        ts=ts,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_event_payload(
    event_id: str,
    ts: float,
    verdict: Verdict,
    score: float,
    packet_metadata: dict,
    base_features: dict,
) -> dict:
    """
    Construct the canonical queue payload dict.

    The payload packs Tier-1 base features alongside raw packet metadata
    so Tier-2 can skip recomputing the shared-base features (dnstds §4).
    """
    return {
        # --- Identity ---
        "event_id": event_id,
        "ts": ts,
        "verdict": verdict.value,
        "tier1_score": score,
        # --- Packet metadata ---
        **packet_metadata,
        # --- Tier-1 base features (shared-base pipeline) ---
        "base_features": base_features,
        # tier2_mode is set to "AUDIT_ENRICHMENT" for FAST_TRACK copies;
        # left absent (or "TRIAGE") for SUSPICIOUS events.
        "tier2_mode": "TRIAGE",
    }
