"""
packet_capture.py — DNS Tunneling Detector
Inline DNS packet capture loop — Tier-1 entry point.

Design doc reference: dnstds §3 — Tier-1 Fast Triage
                       dnstds §7 — Deployment Topologies (Phase 1 monolithic)

Responsibilities:
  1. Sniff raw DNS packets on the configured network interface.
  2. Parse each packet to extract QNAME, QTYPE, source/dest IP.
  3. For every query:
       a. extract_stateless_features()    — four lightweight features
       b. tier1.predict_proba()           — XGBoost score ∈ [0.0, 1.0]
       c. route_triage_result()           — dual-threshold routing

Constraints:
  - Strictly non-blocking w.r.t. Tier-2.  The Tier-2 queue state or
    consumer worker must never add latency to this loop.
  - Runs in the foreground process.  For the monolithic MVP, Tier-1 should
    be pinned to dedicated CPU cores (see main.py).
  - Uses scapy for packet capture. Falls back to a minimal stub when scapy
    is not installed so tests/unit-level usage still works.

Supported record types (QTYPE values mapped to human-readable strings):
  1 → A, 2 → NS, 5 → CNAME, 15 → MX, 16 → TXT, 28 → AAAA, 255 → ANY
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Callable, Optional

from config import get_config, AppConfig
from queue_broker import BaseQueueBroker
from router import AlertCallback, route_triage_result
from tier1_triage import Tier1Classifier, extract_stateless_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DNS QTYPE integer → string mapping
# ---------------------------------------------------------------------------

QTYPE_MAP: dict[int, str] = {
    1:   "A",
    2:   "NS",
    5:   "CNAME",
    12:  "PTR",
    15:  "MX",
    16:  "TXT",
    28:  "AAAA",
    33:  "SRV",
    255: "ANY",
}


# ---------------------------------------------------------------------------
# Optional scapy import — graceful fallback
# ---------------------------------------------------------------------------

try:
    from scapy.all import (  # type: ignore[import-untyped]
        DNS,
        DNSQR,
        IP,
        IPv6,
        UDP,
        sniff,
    )
    _SCAPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCAPY_AVAILABLE = False
    logger.warning(
        "scapy is not installed — CaptureLoop.start() will raise RuntimeError. "
        "Use CaptureLoop.process_packet_dict() directly in unit tests."
    )


# ---------------------------------------------------------------------------
# CaptureLoop
# ---------------------------------------------------------------------------

class CaptureLoop:
    """
    Main Tier-1 packet capture and inline scoring loop.

    Typical usage (from main.py):

        from memory_queue import MemoryQueueProvider
        from tier1_triage import Tier1Classifier
        from packet_capture import CaptureLoop

        queue = MemoryQueueProvider()
        classifier = Tier1Classifier()
        loop = CaptureLoop(queue, classifier, alert_callback=dispatcher.enqueue)
        loop.start()   # blocks until stop() or SIGINT

    Unit-test usage (no network, no scapy required):

        loop = CaptureLoop(queue, classifier)
        result = loop.process_packet_dict({
            "query_name": "abc123.evil.com",
            "record_type": "TXT",
            "source_ip": "10.0.0.1",
            "dest_ip": "8.8.8.8",
            "source_port": 53421,
            "dest_port": 53,
            "packet_ts": time.time(),
        })
    """

    def __init__(
        self,
        queue_broker: BaseQueueBroker,
        classifier: Optional[Tier1Classifier] = None,
        alert_callback: Optional[AlertCallback] = None,
        cfg: Optional[AppConfig] = None,
    ) -> None:
        self._queue = queue_broker
        self._classifier = classifier or Tier1Classifier()
        self._alert_callback = alert_callback
        self._cfg = cfg or get_config()

        self._stop_event = threading.Event()
        self._packet_count: int = 0
        self._error_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Begin sniffing on the configured interface.  Blocks until stop()
        is called or SIGINT/SIGTERM is received.

        Raises:
            RuntimeError: if scapy is not installed.
        """
        if not _SCAPY_AVAILABLE:
            raise RuntimeError(
                "scapy is required for live packet capture. "
                "Install it with: pip install scapy"
            )

        iface = self._cfg.capture_interface
        bpf = self._cfg.capture_bpf_filter

        # Graceful shutdown on SIGINT / SIGTERM
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(
            "Tier-1 capture loop starting — interface=%s  bpf=%r",
            iface, bpf,
        )

        sniff(
            iface=iface,
            filter=bpf,
            prn=self._handle_scapy_packet,
            store=False,          # don't buffer packets in memory
            stop_filter=lambda _: self._stop_event.is_set(),
        )

        logger.info(
            "Tier-1 capture loop stopped — packets=%d  errors=%d",
            self._packet_count,
            self._error_count,
        )

    def stop(self) -> None:
        """Signal the capture loop to exit cleanly."""
        logger.info("Tier-1 capture loop: stop requested.")
        self._stop_event.set()

    @property
    def packet_count(self) -> int:
        """Total DNS queries processed since loop start."""
        return self._packet_count

    @property
    def error_count(self) -> int:
        """Total packets that raised an exception during processing."""
        return self._error_count

    # ------------------------------------------------------------------
    # Core processing — also called directly in unit tests
    # ------------------------------------------------------------------

    def process_packet_dict(self, pkt: dict) -> object:
        """
        Process a pre-parsed DNS query dict through the full Tier-1 pipeline:
          extract_stateless_features → predict_proba → route_triage_result

        This is the testable core of the capture loop. It accepts a plain
        dict so unit tests do not need scapy.

        Args:
            pkt: Dict with keys: query_name (str), record_type (str),
                 source_ip (str), dest_ip (str), source_port (int),
                 dest_port (int), packet_ts (float).

        Returns:
            RoutingResult from router.route_triage_result().
        """
        self._packet_count += 1

        try:
            features = extract_stateless_features(
                query_name=pkt["query_name"],
                record_type=pkt.get("record_type", "A"),
            )

            score = self._classifier.predict_proba(features)

            packet_metadata = {
                "query_name":  pkt["query_name"],
                "record_type": pkt.get("record_type", "A"),
                "source_ip":   pkt.get("source_ip", ""),
                "dest_ip":     pkt.get("dest_ip", ""),
                "source_port": pkt.get("source_port", 0),
                "dest_port":   pkt.get("dest_port", 53),
                "packet_ts":   pkt.get("packet_ts", time.time()),
            }

            return route_triage_result(
                packet_metadata=packet_metadata,
                base_features=features,
                score=score,
                queue_broker=self._queue,
                alert_callback=self._alert_callback,
            )

        except Exception:  # noqa: BLE001
            self._error_count += 1
            logger.exception(
                "Error processing DNS query %r — skipping packet.",
                pkt.get("query_name", "<unknown>"),
            )
            return None

    # ------------------------------------------------------------------
    # Scapy packet handler
    # ------------------------------------------------------------------

    def _handle_scapy_packet(self, raw_pkt: object) -> None:
        """
        Callback invoked by scapy's sniff() for each captured packet.
        Extracts DNS fields and delegates to process_packet_dict().
        """
        try:
            pkt_dict = self._parse_scapy_packet(raw_pkt)
            if pkt_dict is None:
                return  # not a DNS query; silently skip
            self.process_packet_dict(pkt_dict)
        except Exception:  # noqa: BLE001
            self._error_count += 1
            logger.debug("Malformed packet skipped", exc_info=True)

    def _parse_scapy_packet(self, raw_pkt: object) -> Optional[dict]:
        """
        Parse a raw scapy packet into the dict expected by process_packet_dict().

        Returns None if the packet is not a DNS question (QR=0) or lacks
        the required layers.
        """
        if not raw_pkt.haslayer(DNS):  # type: ignore[union-attr]
            return None

        dns_layer = raw_pkt[DNS]  # type: ignore[index]

        # Only process queries (QR=0), skip responses (QR=1)
        if dns_layer.qr != 0:
            return None

        # We only score the first question in the query
        if dns_layer.qdcount < 1 or not raw_pkt.haslayer(DNSQR):  # type: ignore[union-attr]
            return None

        qr = raw_pkt[DNSQR]  # type: ignore[index]
        query_name: str = qr.qname.decode("utf-8", errors="replace").rstrip(".")
        qtype_int: int = int(qr.qtype)
        record_type: str = QTYPE_MAP.get(qtype_int, str(qtype_int))

        # Extract IP layer (supports both IPv4 and IPv6)
        source_ip = dest_ip = ""
        if raw_pkt.haslayer(IP):  # type: ignore[union-attr]
            source_ip = raw_pkt[IP].src   # type: ignore[index]
            dest_ip = raw_pkt[IP].dst     # type: ignore[index]
        elif raw_pkt.haslayer(IPv6):  # type: ignore[union-attr]
            source_ip = raw_pkt[IPv6].src  # type: ignore[index]
            dest_ip = raw_pkt[IPv6].dst    # type: ignore[index]

        source_port = dest_port = 0
        if raw_pkt.haslayer(UDP):  # type: ignore[union-attr]
            source_port = int(raw_pkt[UDP].sport)  # type: ignore[index]
            dest_port = int(raw_pkt[UDP].dport)    # type: ignore[index]

        return {
            "query_name":  query_name,
            "record_type": record_type,
            "source_ip":   source_ip,
            "dest_ip":     dest_ip,
            "source_port": source_port,
            "dest_port":   dest_port,
            "packet_ts":   time.time(),
        }

    # ------------------------------------------------------------------
    # Signal handler
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, frame: object) -> None:  # noqa: ARG002
        logger.info("Signal %d received — stopping capture loop.", signum)
        self.stop()
