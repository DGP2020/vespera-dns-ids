# ADR-0008: Hive-Partitioned Parquet Data Lake for Feedback and Retraining

**Date**: 2026-11-05
**Status**: accepted
**Deciders**: Vespera IDS Engineering Team

## Context

Continuous adaptation against evolving DNS tunneling tools requires collecting query metadata, feature vectors, model inference scores, analyst triage feedback, and SHAP attributions. 

Storing millions of daily DNS inference events in an OLTP relational database (e.g., PostgreSQL or SQLite) incurs heavy storage bloat, index maintenance overhead, and poor analytical query performance when computing feature distributions or training new models. Conversely, storing raw unstructured logs in plain text or single large JSON files makes incremental daily retraining and date-range queries slow and memory-intensive.

## Decision

We implement a two-stage continual learning data pipeline (`feedback_logger.py` and `parquet_lake.py`):
1. **Append-Only Daily JSONL Logging**: Tier-1 and Tier-2 workers append event records with analyst feedback to date-stamped JSONL files (`data/feedback/events_YYYY-MM-DD.jsonl`).
2. **Batch Ingestion to Hive-Partitioned Parquet**: A scheduled ingestion pipeline validates record schemas, converts types, and appends records into a columnar Parquet lake partitioned by date:
   `{parquet_lake_dir}/year=YYYY/month=MM/day=DD/events.parquet`
3. **Partition-Pruned Analytical Access**: Retraining jobs query specific date windows (e.g., rolling 14-day training sets) directly utilizing columnar partition pruning via DuckDB, Apache Spark, PyArrow, or Pandas.

## Alternatives Considered

### Alternative 1: Centralized Relational SQL Database (PostgreSQL / MySQL)
- **Pros**: Strong consistency, SQL query support.
- **Cons**: High CPU and disk overhead for write-heavy network traffic; expensive to scale for hundreds of millions of feature vectors.
- **Why not**: Overkill for append-mostly analytics; tabular model training reads columnar data much faster from Parquet.

### Alternative 2: Compressed Daily JSONL (gzip) Without Columnar Conversion
- **Pros**: Simple cron job compressing daily logs.
- **Cons**: Retraining scripts must decompress and parse all JSON strings into memory, which is 10–20x slower and memory-intensive.
- **Why not**: Parquet provides column projection and predicate pushdown, drastically reducing I/O and RAM usage during model retraining.

## Consequences

### Positive
- 10x–20x compression ratio compared to raw JSON strings.
- Direct zero-copy integration with ML frameworks (Pandas, PyArrow, Polars, DuckDB, XGBoost).
- Hive directory structure (`year=.../month=.../day=...`) enables automatic partition pruning when training on sliding windows.
- Resilient ingestion with CLI backfill support for historical catch-up after offline periods.

### Negative
- Multi-step pipeline: requires a daily or hourly conversion job from JSONL to Parquet.
- Schema evolution requires handling backwards-compatible columns with default values.

### Risks
- "Small file problem" if ingestion is executed too frequently: mitigated by batching conversions to daily or hourly files per partition.
