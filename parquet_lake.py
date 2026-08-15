"""
parquet_lake.py — DNS Tunneling Detector
Parquet ingestion pipeline for the retraining data lake.

Design doc reference: dnstds §9 — Retraining / Feedback Loop

Responsibilities:
  - Read daily feedback JSONL files produced by feedback_logger.py
  - Validate each record against the expected feature schema
  - Convert and append to partitioned Parquet files:
      {parquet_lake_dir}/year=YYYY/month=MM/day=DD/events.parquet
  - Support backfill for multiple days (catch-up after downtime)
  - Expose a CLI entrypoint:
      python parquet_lake.py --date 2026-08-01
      python parquet_lake.py --backfill 7

Partition scheme:
  Hive-style partitioning (year=, month=, day=) compatible with
  Spark / DuckDB / pandas parquet readers.

Schema columns:
  event_id, ts, date_utc, label, query_name, record_type, source_ip,
  tier1_score, tier2_score,
  fqdn_length, subdomain_count, shannon_entropy, numeric_alpha_ratio,
  record_type_heterogeneity, payload_variance, ngram_transition_score,
  unique_ip_concurrency,
  shap_top_feature, shap_top_value,
  ingested_at

Missing values are filled with sensible defaults (0 / empty string).
Records missing event_id are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful pyarrow / pandas imports
# ---------------------------------------------------------------------------

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    _PARQUET_AVAILABLE = True
except ImportError:
    _PARQUET_AVAILABLE = False
    pd = None   # type: ignore[assignment]
    pa = None   # type: ignore[assignment]
    pq = None   # type: ignore[assignment]
    logger.warning(
        "pyarrow/pandas not installed — ParquetLake ingestion will be a no-op. "
        "Install with: pip install pyarrow pandas"
    )

from config import get_config, FeedbackConfig

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Expected column names and their default fill values
_SCHEMA_DEFAULTS: Dict[str, Any] = {
    # Identity
    "event_id":                   "",
    "ts":                         0.0,
    "date_utc":                   "",
    "label":                      "",
    # Network
    "query_name":                 "",
    "record_type":                "",
    "source_ip":                  "",
    # Model scores
    "tier1_score":                0.0,
    "tier2_score":                0.0,
    # Tier-1 base features
    "fqdn_length":                0,
    "subdomain_count":            0,
    "shannon_entropy":            0.0,
    "numeric_alpha_ratio":        0.0,
    # Tier-2 stateful features
    "record_type_heterogeneity":  0.0,
    "payload_variance":           0.0,
    "ngram_transition_score":     0.0,
    "unique_ip_concurrency":      0.0,
    # XAI (top-1 feature by SHAP value — for quick filtering)
    "shap_top_feature":           "",
    "shap_top_value":             0.0,
    # Ingestion metadata
    "ingested_at":                0.0,
}

_REQUIRED_COLS = {"event_id", "label", "ts"}


def _flatten_record(raw: dict) -> Optional[dict]:
    """
    Flatten one raw feedback record into the Parquet schema columns.

    Nested dicts (base_features, stateful_features) are unpacked into
    top-level columns. Missing keys are filled from _SCHEMA_DEFAULTS.

    Returns None if the record is missing required fields.
    """
    # Check required fields
    for col in _REQUIRED_COLS:
        if not raw.get(col):
            logger.warning("Skipping record missing required field %r: %s", col, raw)
            return None

    flat = dict(_SCHEMA_DEFAULTS)  # start with defaults

    # Identity + network
    for key in ("event_id", "ts", "date_utc", "label",
                "query_name", "record_type", "source_ip",
                "tier1_score", "tier2_score"):
        if key in raw:
            flat[key] = raw[key]

    # Unpack base_features
    for feat in ("fqdn_length", "subdomain_count", "shannon_entropy", "numeric_alpha_ratio"):
        val = raw.get("base_features", {}).get(feat)
        if val is not None:
            flat[feat] = val

    # Unpack stateful_features
    for feat in ("record_type_heterogeneity", "payload_variance",
                 "ngram_transition_score", "unique_ip_concurrency"):
        val = raw.get("stateful_features", {}).get(feat)
        if val is not None:
            flat[feat] = val

    # Top-1 SHAP feature (for quick filtering/analysis)
    shap_list = raw.get("shap_attributions", [])
    if shap_list:
        top = shap_list[0]
        flat["shap_top_feature"] = top.get("feature", "")
        flat["shap_top_value"]   = float(top.get("shap_value", 0.0))

    flat["ingested_at"] = time.time()
    return flat


# ---------------------------------------------------------------------------
# ParquetLake
# ---------------------------------------------------------------------------

class ParquetLake:
    """
    Parquet data lake writer for the feedback/retraining pipeline.

    Partition layout:
        {parquet_lake_dir}/year=YYYY/month=MM/day=DD/events.parquet

    Usage:
        lake = ParquetLake()
        records_written = lake.ingest_daily("2026-08-01")

        # Backfill last 7 days:
        lake.run_backfill(days=7)
    """

    def __init__(self, cfg: Optional[FeedbackConfig] = None) -> None:
        self._cfg = cfg or get_config().feedback
        self._feedback_dir = Path(self._cfg.feedback_log_dir)
        self._lake_dir     = Path(self._cfg.parquet_lake_dir)
        self._lake_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_daily(self, date_str: str) -> int:
        """
        Read the feedback JSONL file for `date_str` (format: YYYY-MM-DD)
        and write/append records to the Parquet partition.

        Args:
            date_str: ISO date string, e.g. "2026-08-01".

        Returns:
            Number of records successfully written. 0 on missing file or error.
        """
        if not _PARQUET_AVAILABLE:
            logger.error("pyarrow/pandas not installed — cannot write Parquet.")
            return 0

        jsonl_path = self._feedback_path_for_date(date_str)
        if not jsonl_path.exists():
            logger.info("No feedback file for %s (%s) — skipping.", date_str, jsonl_path)
            return 0

        rows = self._read_jsonl(jsonl_path)
        if not rows:
            logger.info("Feedback file %s is empty — skipping.", jsonl_path.name)
            return 0

        flat_rows = []
        for raw in rows:
            flat = _flatten_record(raw)
            if flat is not None:
                flat_rows.append(flat)

        if not flat_rows:
            logger.warning("All records in %s were invalid — nothing written.", jsonl_path.name)
            return 0

        out_path = self._partition_path(date_str)
        self._write_parquet(flat_rows, out_path)

        logger.info(
            "Parquet lake: ingested %d records from %s → %s",
            len(flat_rows), jsonl_path.name, out_path,
        )
        return len(flat_rows)

    def run_backfill(self, days: int = 7) -> dict:
        """
        Batch-ingest the last `days` calendar days.

        Returns:
            Dict mapping date_str → records_written.
        """
        results: dict[str, int] = {}
        today = date.today()
        for delta in range(days - 1, -1, -1):
            d = today - timedelta(days=delta)
            date_str = d.isoformat()
            results[date_str] = self.ingest_daily(date_str)
        return results

    def list_partitions(self) -> List[Path]:
        """Return all Parquet partition files in the lake, sorted by path."""
        return sorted(self._lake_dir.glob("**/*.parquet"))

    def total_records(self) -> int:
        """Count total rows across all partition files in the lake."""
        if not _PARQUET_AVAILABLE:
            return 0
        total = 0
        for path in self.list_partitions():
            try:
                total += pq.read_metadata(str(path)).num_rows
            except Exception:  # noqa: BLE001
                pass
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _feedback_path_for_date(self, date_str: str) -> Path:
        """Convert '2026-08-01' → feedback_metrics_2026_08_01.jsonl"""
        date_underscored = date_str.replace("-", "_")
        return self._feedback_dir / f"feedback_metrics_{date_underscored}.jsonl"

    def _partition_path(self, date_str: str) -> Path:
        """
        Convert '2026-08-01' → {lake_dir}/year=2026/month=08/day=01/events.parquet
        """
        parts = date_str.split("-")
        year, month, day = parts[0], parts[1], parts[2]
        partition = (
            self._lake_dir
            / f"year={year}"
            / f"month={month}"
            / f"day={day}"
        )
        partition.mkdir(parents=True, exist_ok=True)
        return partition / "events.parquet"

    @staticmethod
    def _read_jsonl(path: Path) -> List[dict]:
        """Read all valid JSON lines from a JSONL file."""
        records = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d — %s", path.name, lineno, exc
                    )
        return records

    @staticmethod
    def _write_parquet(rows: List[dict], out_path: Path) -> None:
        """
        Write (or overwrite) a Parquet file at `out_path`.
        Uses pyarrow for correct type inference and schema enforcement.
        """
        df = pd.DataFrame(rows)

        # Enforce column order and fill missing columns
        for col, default in _SCHEMA_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        df = df[[col for col in _SCHEMA_DEFAULTS.keys() if col in df.columns]]

        # Type coercions for safe Parquet serialisation
        for float_col in ("ts", "tier1_score", "tier2_score",
                          "shannon_entropy", "numeric_alpha_ratio",
                          "record_type_heterogeneity", "payload_variance",
                          "ngram_transition_score", "unique_ip_concurrency",
                          "shap_top_value", "ingested_at"):
            if float_col in df.columns:
                df[float_col] = df[float_col].astype(float)

        for int_col in ("fqdn_length", "subdomain_count"):
            if int_col in df.columns:
                df[int_col] = df[int_col].astype(int)

        table = pa.Table.from_pandas(df)

        # Overwrite today's partition file
        # (safe: JSONL append is idempotent on re-run for the same day)
        pq.write_table(table, str(out_path), compression="snappy")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Parquet data lake ingestion for DNS Tunneling Detector feedback."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="Ingest feedback for a specific date.",
    )
    group.add_argument(
        "--backfill",
        type=int,
        metavar="N",
        help="Ingest feedback for the last N calendar days.",
    )
    args = parser.parse_args()

    lake = ParquetLake()

    if args.date:
        written = lake.ingest_daily(args.date)
        print(f"Ingested {written} records for {args.date}.")
    elif args.backfill:
        results = lake.run_backfill(days=args.backfill)
        total = sum(results.values())
        for d, n in results.items():
            print(f"  {d}: {n} records")
        print(f"Total: {total} records ingested across {args.backfill} days.")


if __name__ == "__main__":
    main()
