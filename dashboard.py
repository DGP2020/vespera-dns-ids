"""
dashboard.py — DNS Tunneling Detector
3-State real-time terminal monitoring dashboard.

Design doc reference: dnstds §8 — Dashboard

Visual tiers (dnstds §8):
  🚨  Tier-1 Fast-Track  → Flashing red banner (immediate block/isolation)
  🟡  Tier-1 Triage      → Pulsing amber row (staged evaluation queue)
  🟠  Tier-2 Escalated   → Solid orange row (full stateful notes + XAI breakdown)

Also shows:
  - TIER2_QUEUE_OVERFLOW_DROP count (from MemoryQueueProvider.stats())
  - alert_dispatch_dropped_total (from alert_storage.get_dropped_total())
  - Alert delivery queue breakdown (PENDING / DELIVERED / FAILED_DLQ)

Usage:
    from dashboard import EventBus, DashboardEvent, Dashboard, EventType

    # Push events from any component:
    EventBus.push(DashboardEvent(
        event_type=EventType.FAST_TRACK,
        event_id="...",
        score=0.95,
        query_name="abc.evil.com",
        source_ip="10.0.0.5",
    ))

    # Start the dashboard (blocks until Ctrl-C):
    dash = Dashboard()
    dash.run()
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful rich import
# ---------------------------------------------------------------------------

try:
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
    logger.warning(
        "rich is not installed — Dashboard.run() will use a plain-text fallback. "
        "Install with: pip install rich"
    )


# ---------------------------------------------------------------------------
# Event types & data model
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Three visual tiers from dnstds §8."""
    FAST_TRACK  = "FAST_TRACK"     # 🚨 Tier-1 immediate alert
    TRIAGE      = "TRIAGE"         # 🟡 Tier-1 pending queue
    ESCALATED   = "ESCALATED"      # 🟠 Tier-2 verified critical


@dataclass
class DashboardEvent:
    """
    Lightweight event record pushed by any pipeline component.
    All fields are optional — missing values display as '—'.
    """
    event_type:    EventType
    event_id:      str         = ""
    ts:            float       = field(default_factory=time.time)
    score:         float       = 0.0        # Tier-1 score
    tier2_score:   float       = 0.0        # Tier-2 score (ESCALATED only)
    query_name:    str         = ""
    source_ip:     str         = ""
    record_type:   str         = ""
    parent_domain: str         = ""
    # Top-3 SHAP attributions [{feature, shap_value, raw_value}, ...]
    shap_top3:     List[dict]  = field(default_factory=list)


# ---------------------------------------------------------------------------
# EventBus — thread-safe ring buffer
# ---------------------------------------------------------------------------

class EventBus:
    """
    Singleton ring-buffer for DashboardEvents.

    Any component (router.py, queue_consumer.py, etc.) can push events
    without importing the Dashboard class — decoupled via this shared bus.

    Ring buffer size: 500 events (configurable via MAXLEN).
    """

    MAXLEN: int = 500
    _events: Deque[DashboardEvent] = deque(maxlen=MAXLEN)
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def push(cls, event: DashboardEvent) -> None:
        """Append a new event. Non-blocking. Thread-safe."""
        with cls._lock:
            cls._events.append(event)

    @classmethod
    def snapshot(cls) -> List[DashboardEvent]:
        """Return a copy of the current ring buffer (newest-last)."""
        with cls._lock:
            return list(cls._events)

    @classmethod
    def recent(cls, n: int = 50) -> List[DashboardEvent]:
        """Return the most recent `n` events (newest-last)."""
        with cls._lock:
            events = list(cls._events)
        return events[-n:]

    @classmethod
    def clear(cls) -> None:
        """Clear all events — useful in tests."""
        with cls._lock:
            cls._events.clear()


# ---------------------------------------------------------------------------
# Metric sources (injected externally or polled at render time)
# ---------------------------------------------------------------------------

class MetricSource:
    """
    Pluggable metric source for the dashboard.
    Override methods to inject real queue / alert storage data.
    Default implementations return safe zero values so the dashboard
    can start before all subsystems are available.
    """

    def queue_overflow_drop_count(self) -> int:
        """TIER2_QUEUE_OVERFLOW_DROP counter (from MemoryQueueProvider)."""
        return 0

    def alert_dispatch_dropped_total(self) -> int:
        """alert_dispatch_dropped_total (from alert_storage)."""
        try:
            from alert_storage import get_dropped_total  # noqa: PLC0415
            return get_dropped_total()
        except ImportError:
            return 0

    def alert_queue_by_status(self) -> Dict[str, int]:
        """Status → count breakdown from AlertStorage."""
        return {}

    def queue_depth(self) -> int:
        """Current Tier-2 queue depth (approximate)."""
        return 0


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

def _format_ts(ts: float) -> str:
    """Format a Unix timestamp as HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _shap_summary(shap_top3: List[dict]) -> str:
    """Format top-3 SHAP features as a compact string for the table."""
    if not shap_top3:
        return "—"
    parts = []
    for item in shap_top3[:3]:
        sign = "+" if item["shap_value"] >= 0 else ""
        parts.append(f"{item['feature']}={sign}{item['shap_value']:.2f}")
    return "  ".join(parts)


class Dashboard:
    """
    Rich-based TUI dashboard rendering the 3-state event matrix.

    Usage:
        dash = Dashboard()
        dash.run()    # blocks, Ctrl-C to exit
    """

    REFRESH_HZ: float = 0.5       # render rate (2 Hz = 0.5s)
    MAX_ROWS: int = 20            # max event rows per section

    def __init__(
        self,
        metric_source: Optional[MetricSource] = None,
    ) -> None:
        self._metrics = metric_source or MetricSource()
        self._stop_event = threading.Event()

    def run(self) -> None:
        """
        Start the dashboard loop. Blocks until Ctrl-C or stop() is called.
        Falls back to a plain-text stderr mode if rich is not installed.
        """
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        if not _RICH_AVAILABLE:
            self._run_plaintext_fallback()
            return

        console = Console()
        with Live(
            self._render(),
            console=console,
            refresh_per_second=1 / self.REFRESH_HZ,
            screen=True,
        ) as live:
            while not self._stop_event.is_set():
                live.update(self._render())
                time.sleep(self.REFRESH_HZ)

    def stop(self) -> None:
        """Signal the render loop to exit."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Render pipeline
    # ------------------------------------------------------------------

    def _render(self) -> Layout:
        """Build the full terminal layout for one frame."""
        events = EventBus.recent(n=200)

        fast_track = [e for e in events if e.event_type == EventType.FAST_TRACK]
        triage     = [e for e in events if e.event_type == EventType.TRIAGE]
        escalated  = [e for e in events if e.event_type == EventType.ESCALATED]

        layout = Layout()
        layout.split_column(
            Layout(self._render_header(), name="header",    size=3),
            Layout(name="alerts",                           size=12),
            Layout(name="triage",                           size=12),
            Layout(self._render_metrics(),   name="metrics", size=8),
        )
        layout["alerts"].split_row(
            Layout(self._render_fast_track_panel(fast_track[-self.MAX_ROWS:]), name="fast_track"),
            Layout(self._render_escalated_panel(escalated[-self.MAX_ROWS:]),   name="escalated"),
        )
        layout["triage"].update(self._render_triage_panel(triage[-self.MAX_ROWS:]))
        return layout

    def _render_header(self) -> Panel:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        title = Text("Vespera IDS — DNS Tunneling Detector", style="bold white")
        subtitle = Text(f"  {ts}  |  Live Threat Monitor", style="dim")
        return Panel(Columns([title, subtitle]), style="on dark_blue")

    def _render_fast_track_panel(self, events: List[DashboardEvent]) -> Panel:
        """🚨 Tier-1 Fast-Track — flashing red banner section."""
        table = Table(
            "Time", "Event ID", "Score", "Source IP", "Query Name",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white on red",
            style="red",
            expand=True,
        )
        for e in reversed(events):
            table.add_row(
                _format_ts(e.ts),
                e.event_id[:12] + "…" if len(e.event_id) > 12 else e.event_id,
                f"{e.score:.4f}",
                e.source_ip or "—",
                e.query_name or "—",
                style="bold red",
            )
        count = len(events)
        return Panel(
            table,
            title=f"[bold red]🚨 Tier-1 FAST-TRACK  ({count} recent)[/bold red]",
            border_style="red",
        )

    def _render_triage_panel(self, events: List[DashboardEvent]) -> Panel:
        """🟡 Tier-1 Triage — pulsing amber rows."""
        table = Table(
            "Time", "Event ID", "Score", "Source IP", "Query Name", "Record",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold yellow",
            style="yellow",
            expand=True,
        )
        for e in reversed(events):
            table.add_row(
                _format_ts(e.ts),
                e.event_id[:12] + "…" if len(e.event_id) > 12 else e.event_id,
                f"{e.score:.4f}",
                e.source_ip or "—",
                e.query_name or "—",
                e.record_type or "—",
            )
        count = len(events)
        return Panel(
            table,
            title=f"[bold yellow]🟡 Tier-1 TRIAGE QUEUE  ({count} recent)[/bold yellow]",
            border_style="yellow",
        )

    def _render_escalated_panel(self, events: List[DashboardEvent]) -> Panel:
        """🟠 Tier-2 Escalated — solid orange, full XAI breakdown."""
        table = Table(
            "Time", "T2 Score", "Domain", "Source IP", "Top-3 SHAP Features",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white on dark_orange",
            style="orange1",
            expand=True,
        )
        for e in reversed(events):
            table.add_row(
                _format_ts(e.ts),
                f"{e.tier2_score:.4f}",
                e.parent_domain or e.query_name or "—",
                e.source_ip or "—",
                _shap_summary(e.shap_top3),
            )
        count = len(events)
        return Panel(
            table,
            title=f"[bold orange1]🟠 Tier-2 ESCALATED  ({count} recent)[/bold orange1]",
            border_style="orange1",
        )

    def _render_metrics(self) -> Panel:
        """System health metrics bar at the bottom of the layout."""
        overflow_drops = self._metrics.queue_overflow_drop_count()
        dlq_drops      = self._metrics.alert_dispatch_dropped_total()
        status_counts  = self._metrics.alert_queue_by_status()
        queue_depth    = self._metrics.queue_depth()

        pending   = status_counts.get("PENDING", 0)
        delivered = status_counts.get("DELIVERED", 0)
        failed    = status_counts.get("FAILED_DLQ", 0)

        table = Table(box=box.SIMPLE, show_header=False, expand=True)
        table.add_column("Metric", style="dim")
        table.add_column("Value",  style="bold")
        table.add_row("T2 Queue Depth",          str(queue_depth))
        table.add_row("Queue Overflow Drops",    _color_int(overflow_drops, warn=1, crit=10))
        table.add_row("Alert Dispatch Drops",    _color_int(dlq_drops, warn=1, crit=5))
        table.add_row("Alerts Pending",          str(pending))
        table.add_row("Alerts Delivered",        f"[green]{delivered}[/green]")
        table.add_row("Alerts FAILED_DLQ",       _color_int(failed, warn=1, crit=5))

        return Panel(table, title="[bold]System Metrics[/bold]", border_style="dim")

    # ------------------------------------------------------------------
    # Plain-text fallback (no rich)
    # ------------------------------------------------------------------

    def _run_plaintext_fallback(self) -> None:
        """Minimal stderr polling loop when rich is not available."""
        import sys
        print("[Vespera IDS] Dashboard running (plain-text mode — install 'rich' for full UI)", file=sys.stderr)
        while not self._stop_event.is_set():
            events = EventBus.recent(n=10)
            for e in events:
                ts = _format_ts(e.ts)
                print(
                    f"[{ts}] {e.event_type.value:12s}  score={e.score:.4f}  "
                    f"query={e.query_name}  src={e.source_ip}",
                    file=sys.stderr,
                )
            time.sleep(2.0)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, frame: object) -> None:  # noqa: ARG002
        logger.info("Signal %d received — stopping dashboard.", signum)
        self.stop()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _color_int(value: int, warn: int, crit: int) -> str:
    """Return a rich-formatted string coloured green/yellow/red by threshold."""
    if not _RICH_AVAILABLE:
        return str(value)
    if value >= crit:
        return f"[bold red]{value}[/bold red]"
    if value >= warn:
        return f"[yellow]{value}[/yellow]"
    return f"[green]{value}[/green]"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Demo: push synthetic events so the dashboard shows something
    import uuid
    for _ in range(5):
        EventBus.push(DashboardEvent(
            event_type=EventType.FAST_TRACK,
            event_id=str(uuid.uuid4()),
            score=0.97,
            query_name="abc123def.evil.com",
            source_ip="10.0.0.1",
            record_type="TXT",
        ))
        EventBus.push(DashboardEvent(
            event_type=EventType.TRIAGE,
            event_id=str(uuid.uuid4()),
            score=0.75,
            query_name="data.cdn.example.com",
            source_ip="192.168.1.5",
            record_type="A",
        ))
        EventBus.push(DashboardEvent(
            event_type=EventType.ESCALATED,
            event_id=str(uuid.uuid4()),
            score=0.76,
            tier2_score=0.88,
            query_name="xyz.tunnel.evil.com",
            source_ip="10.0.0.5",
            parent_domain="evil.com",
            shap_top3=[
                {"feature": "ngram_transition_score", "shap_value": 0.42, "raw_value": 3.7},
                {"feature": "record_type_heterogeneity", "shap_value": 0.31, "raw_value": 0.85},
                {"feature": "shannon_entropy",  "shap_value": 0.19, "raw_value": 3.1},
            ],
        ))

    dash = Dashboard()
    dash.run()
