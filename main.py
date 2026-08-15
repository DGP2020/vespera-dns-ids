"""
main.py — DNS Tunneling Detector
Monolithic MVP process supervisor.

Design doc reference: dnstds §7 — Deployment Topologies (Phase 1)

Responsibilities:
  - Spawn and supervise all sub-processes:
      • Tier-1 capture loop (pinned to CPU cores 0/1 via os.sched_setaffinity)
      • Tier-2 consumer worker(s) (async, off the inline path)
  - Instantiate the shared MemoryQueueProvider and inject it into both tiers.
  - Wire the AlertDispatcher (Phase 4) into CaptureLoop so fast-track alerts
    are persisted to SQLite and dispatched to the SOC webhook.
  - Handle graceful shutdown on SIGINT / SIGTERM.

Note on CPU pinning:
  os.sched_setaffinity is Linux-only.  On other platforms (Windows, macOS)
  the pinning step is skipped with a warning.

Usage:
    poetry run python main.py
    # or
    python main.py
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Logging setup (before importing project modules so they inherit handlers)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("dns_detector.main")


# ---------------------------------------------------------------------------
# Project imports (after logging is configured)
# ---------------------------------------------------------------------------

from config import get_config
from memory_queue import MemoryQueueProvider
from packet_capture import CaptureLoop
from tier1_triage import Tier1Classifier
from alert_dispatcher import AlertDispatcher


# ---------------------------------------------------------------------------
# CPU affinity helper
# ---------------------------------------------------------------------------

def _pin_process_to_cores(cores: list[int], label: str) -> None:
    """
    Attempt to pin the current process to the listed CPU cores.
    Silently skips on platforms that don't support sched_setaffinity.
    """
    try:
        os.sched_setaffinity(0, set(cores))  # type: ignore[attr-defined]
        logger.info("%s pinned to CPU cores %s", label, cores)
    except AttributeError:
        logger.warning(
            "%s CPU pinning skipped (os.sched_setaffinity not available on "
            "this platform — %s)",
            label, sys.platform,
        )
    except OSError as exc:
        logger.warning("%s CPU pinning failed: %s", label, exc)


# ---------------------------------------------------------------------------
# Tier-2 worker process entry point
# ---------------------------------------------------------------------------

def _tier2_worker(queue_provider: MemoryQueueProvider, worker_id: int) -> None:
    """
    Entry point for each Tier-2 consumer worker process.
    Imports are deferred so the forked process only loads Tier-2 modules.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s  %(levelname)-8s  [T2-worker-{worker_id}] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger_w = logging.getLogger(f"dns_detector.tier2_worker.{worker_id}")
    logger_w.info("Tier-2 worker %d starting.", worker_id)

    # Real Tier-2 implementation (Phase 3 + Phase 4 complete)
    from queue_consumer import Tier2ConsumerWorker  # noqa: PLC0415
    from alert_dispatcher import AlertDispatcher     # noqa: PLC0415

    dispatcher = AlertDispatcher()
    dispatcher.start()  # background HTTP delivery thread

    worker = Tier2ConsumerWorker(
        queue_provider,
        worker_id=worker_id,
        alert_callback=dispatcher.enqueue_alert,
    )
    try:
        worker.run()
    finally:
        dispatcher.stop()


# ---------------------------------------------------------------------------
# Alert callback — wires AlertDispatcher into the Tier-1 capture loop
# ---------------------------------------------------------------------------

# The main process keeps its own AlertDispatcher for fast-track events.
# It is instantiated once in DetectorSupervisor.start() and shared via closure.
_main_dispatcher: Optional[AlertDispatcher] = None


def _fast_track_alert_callback(event_payload: dict) -> None:
    """
    Alert callback fired by the router for FAST_TRACK events.
    Stages the payload in SQLite via AlertDispatcher.enqueue_alert()
    (non-blocking). The background thread handles HTTP delivery.
    """
    if _main_dispatcher is not None:
        _main_dispatcher.enqueue_alert(event_payload)
    else:
        # Fallback during early startup before dispatcher is ready
        logger.critical(
            "🚨 FAST_TRACK ALERT (dispatcher not ready)  event_id=%s  score=%.4f  query=%s  src=%s",
            event_payload.get("event_id"),
            event_payload.get("tier1_score", 0.0),
            event_payload.get("query_name"),
            event_payload.get("source_ip"),
        )


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class DetectorSupervisor:
    """
    Orchestrates the monolithic MVP process topology:

        Main process
          ├── AlertDispatcher (background thread, HTTP delivery)
          ├── CaptureLoop (Tier-1, runs inline in main process)
          └── Tier-2 worker processes  (spawned via multiprocessing)

    The shared MemoryQueueProvider bridges the two tiers.
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._queue: Optional[MemoryQueueProvider] = None
        self._classifier: Optional[Tier1Classifier] = None
        self._capture_loop: Optional[CaptureLoop] = None
        self._dispatcher: Optional[AlertDispatcher] = None
        self._tier2_processes: list[multiprocessing.Process] = []
        self._running = False

    def start(self) -> None:
        """Start all sub-processes and the Tier-1 capture loop."""
        global _main_dispatcher
        logger.info("DNS Tunneling Detector starting (monolithic MVP).")

        # 1. Start the AlertDispatcher for the main (Tier-1) process
        self._dispatcher = AlertDispatcher(self._cfg.alert)
        self._dispatcher.start()
        _main_dispatcher = self._dispatcher
        logger.info("AlertDispatcher started.")

        # 2. Shared queue
        self._queue = MemoryQueueProvider(self._cfg.queue)
        logger.info("MemoryQueueProvider ready (maxsize=%d).", self._cfg.queue.max_size)

        # 3. Spawn Tier-2 worker processes BEFORE entering the Tier-1 loop
        self._spawn_tier2_workers()

        # 4. Pin THIS process (Tier-1) to the configured CPU cores
        if self._cfg.tier1_cpu_affinity:
            _pin_process_to_cores(self._cfg.tier1_cpu_affinity, "Tier-1")

        # 5. Build Tier-1 classifier
        self._classifier = Tier1Classifier(self._cfg.tier1)

        # 6. Build capture loop and inject the real alert callback
        self._capture_loop = CaptureLoop(
            queue_broker=self._queue,
            classifier=self._classifier,
            alert_callback=_fast_track_alert_callback,
            cfg=self._cfg,
        )

        # 7. Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self._running = True
        logger.info(
            "Tier-1 capture loop starting on interface=%s …",
            self._cfg.capture_interface,
        )

        # Blocking call — returns when stop() signals the loop to exit
        self._capture_loop.start()

        # After capture loop exits, clean up
        if self._dispatcher:
            self._dispatcher.stop()
        self._shutdown_tier2_workers()
        logger.info(
            "Shutdown complete. packets=%d  errors=%d",
            self._capture_loop.packet_count,
            self._capture_loop.error_count,
        )

    def stop(self) -> None:
        """Request a graceful shutdown."""
        if self._capture_loop is not None:
            self._capture_loop.stop()

    def _spawn_tier2_workers(self) -> None:
        count = self._cfg.tier2_worker_count
        logger.info("Spawning %d Tier-2 worker process(es).", count)
        for i in range(count):
            p = multiprocessing.Process(
                target=_tier2_worker,
                args=(self._queue, i),
                name=f"dns-tier2-worker-{i}",
                daemon=True,
            )
            p.start()
            self._tier2_processes.append(p)
            logger.info("Tier-2 worker %d started (pid=%d).", i, p.pid)

    def _shutdown_tier2_workers(self) -> None:
        logger.info("Shutting down %d Tier-2 worker(s).", len(self._tier2_processes))
        for p in self._tier2_processes:
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                logger.warning("Tier-2 worker %s did not exit cleanly — killing.", p.name)
                p.kill()
        if self._queue is not None:
            self._queue.close()

    def _handle_shutdown(self, signum: int, frame: object) -> None:  # noqa: ARG002
        logger.info("Signal %d received — initiating graceful shutdown.", signum)
        self.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # multiprocessing on Windows / macOS requires the freeze_support call
    # and the "spawn" start method (default on those platforms).
    multiprocessing.freeze_support()

    supervisor = DetectorSupervisor()
    supervisor.start()
