"""
config.py — DNS Tunneling Detector
Central configuration engine.

All numeric constants that appear in dnstds(1).md are defined here.
Environment variables (prefixed DNS_) override defaults at startup.

Usage:
    from config import get_config
    cfg = get_config()
    threshold = cfg.tier1.p_fasttrack_min  # 0.92
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helper: read env var as float, fall back to default
# ---------------------------------------------------------------------------

def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Environment variable {key} must be a float, got: {val!r}")


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Environment variable {key} must be an int, got: {val!r}")


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Dataclasses — one per subsystem
# ---------------------------------------------------------------------------

@dataclass
class Tier1Config:
    """Configuration for the Tier-1 fast inline triage classifier."""

    # --- Routing thresholds (dnstds §5) ---
    p_benign_max: float = field(default_factory=lambda: _env_float("DNS_T1_P_BENIGN_MAX", 0.70))
    p_fasttrack_min: float = field(default_factory=lambda: _env_float("DNS_T1_P_FASTTRACK_MIN", 0.92))

    # --- XGBoost hyper-parameters (dnstds §3) ---
    xgb_max_depth: int = field(default_factory=lambda: _env_int("DNS_T1_XGB_MAX_DEPTH", 4))
    xgb_n_estimators: int = field(default_factory=lambda: _env_int("DNS_T1_XGB_N_ESTIMATORS", 50))
    xgb_tree_method: str = field(default_factory=lambda: _env_str("DNS_T1_XGB_TREE_METHOD", "hist"))

    # --- Performance budgets (dnstds §3) ---
    max_latency_ms: float = field(default_factory=lambda: _env_float("DNS_T1_MAX_LATENCY_MS", 1.5))
    target_qps: int = field(default_factory=lambda: _env_int("DNS_T1_TARGET_QPS", 50_000))

    # --- Model artefact path ---
    # Tier-1 active model is located via an atomic symlink (dnstds §10)
    model_symlink: Path = field(
        default_factory=lambda: Path(_env_str("DNS_T1_MODEL_SYMLINK", "models/xgboost_tier1_active.json"))
    )


@dataclass
class Tier2Config:
    """Configuration for the Tier-2 deep-analysis classifier."""

    # --- XGBoost hyper-parameters (dnstds §4) ---
    xgb_max_depth: int = field(default_factory=lambda: _env_int("DNS_T2_XGB_MAX_DEPTH", 6))
    xgb_n_estimators: int = field(default_factory=lambda: _env_int("DNS_T2_XGB_N_ESTIMATORS", 200))
    xgb_tree_method: str = field(default_factory=lambda: _env_str("DNS_T2_XGB_TREE_METHOD", "hist"))

    # --- Performance budgets (dnstds §4) ---
    max_latency_ms: float = field(default_factory=lambda: _env_float("DNS_T2_MAX_LATENCY_MS", 200.0))

    # --- Sliding context window (dnstds §4) ---
    state_window_seconds: int = field(default_factory=lambda: _env_int("DNS_T2_STATE_WINDOW_SEC", 30))

    # --- N-gram size for character transition matrix ---
    ngram_size: int = field(default_factory=lambda: _env_int("DNS_T2_NGRAM_SIZE", 2))

    # --- Model artefact path ---
    model_symlink: Path = field(
        default_factory=lambda: Path(_env_str("DNS_T2_MODEL_SYMLINK", "models/xgboost_tier2_active.json"))
    )


@dataclass
class QueueConfig:
    """Configuration for the BaseQueueBroker and its providers."""

    # --- Capacity (dnstds §6) ---
    max_size: int = field(default_factory=lambda: _env_int("DNS_QUEUE_MAX_SIZE", 10_000))

    # --- Overflow disk-log path (dnstds §6) ---
    overflow_log_path: Path = field(
        default_factory=lambda: Path(
            _env_str("DNS_QUEUE_OVERFLOW_LOG", "data/logs/queue_overflow.log")
        )
    )

    # --- Redis settings (Phase 2, dnstds §6) ---
    redis_host: str = field(default_factory=lambda: _env_str("DNS_REDIS_HOST", "127.0.0.1"))
    redis_port: int = field(default_factory=lambda: _env_int("DNS_REDIS_PORT", 6379))
    redis_password: str = field(default_factory=lambda: _env_str("DNS_REDIS_PASSWORD", ""))
    redis_ssl: bool = field(default_factory=lambda: _env_bool("DNS_REDIS_SSL", False))
    redis_queue_name: str = field(default_factory=lambda: _env_str("DNS_REDIS_QUEUE_NAME", "dns_triage_queue"))
    redis_max_memory_mb: int = field(default_factory=lambda: _env_int("DNS_REDIS_MAX_MEMORY_MB", 512))

    # --- Consumer polling interval ---
    consumer_poll_timeout_sec: float = field(
        default_factory=lambda: _env_float("DNS_QUEUE_POLL_TIMEOUT_SEC", 0.1)
    )


@dataclass
class AlertConfig:
    """Configuration for the SOC Alert Dispatcher."""

    # --- Webhook target ---
    webhook_url: str = field(default_factory=lambda: _env_str("DNS_ALERT_WEBHOOK_URL", ""))
    allow_insecure_local_webhooks: bool = field(
        default_factory=lambda: _env_bool("DNS_ALERT_ALLOW_LOCAL_WEBHOOKS", False)
    )

    # --- Auth (dnstds §11) ---
    auth_type: str = field(default_factory=lambda: _env_str("DNS_ALERT_AUTH_TYPE", "bearer"))  # 'bearer' | 'hmac'
    bearer_token: str = field(default_factory=lambda: _env_str("DNS_ALERT_BEARER_TOKEN", ""))
    hmac_secret: str = field(default_factory=lambda: _env_str("DNS_ALERT_HMAC_SECRET", ""))

    # --- Exponential backoff (dnstds §11) ---
    # delay = min(retry_max_sec, retry_base_sec * 2^retry_count) ± jitter
    retry_base_sec: int = field(default_factory=lambda: _env_int("DNS_ALERT_RETRY_BASE_SEC", 2))
    retry_max_sec: int = field(default_factory=lambda: _env_int("DNS_ALERT_RETRY_MAX_SEC", 300))
    max_retries: int = field(default_factory=lambda: _env_int("DNS_ALERT_MAX_RETRIES", 5))

    # --- SQLite WAL buffer (dnstds §11) ---
    sqlite_db_path: Path = field(
        default_factory=lambda: Path(_env_str("DNS_ALERT_SQLITE_PATH", "data/alerts.db"))
    )

    # --- Dead-letter log (dnstds §11) ---
    dead_letter_log_path: Path = field(
        default_factory=lambda: Path(
            _env_str("DNS_ALERT_DLQ_PATH", "data/logs/failed_alerts.jsonl")
        )
    )

    # --- Circuit breaker ---
    circuit_open_threshold: int = field(
        default_factory=lambda: _env_int("DNS_ALERT_CIRCUIT_THRESHOLD", 3)
    )


@dataclass
class PromotionConfig:
    """Configuration for the safe model promotion pipeline (dnstds §10)."""

    # --- Hard performance bounds ---
    tier1_min_recall: float = field(
        default_factory=lambda: _env_float("DNS_PROMO_T1_MIN_RECALL", 0.995)
    )
    tier2_min_precision: float = field(
        default_factory=lambda: _env_float("DNS_PROMO_T2_MIN_PRECISION", 0.980)
    )
    tier2_max_fpr_delta: float = field(
        default_factory=lambda: _env_float("DNS_PROMO_T2_MAX_FPR_DELTA", 0.001)
    )
    max_latency_regression_pct: float = field(
        default_factory=lambda: _env_float("DNS_PROMO_MAX_LATENCY_REGRESSION_PCT", 5.0)
    )

    # --- Shadow inference period (dnstds §10) ---
    shadow_duration_hours: int = field(
        default_factory=lambda: _env_int("DNS_PROMO_SHADOW_HOURS", 48)
    )

    # --- Validation dataset paths (dnstds §10 — golden + 14-day rolling) ---
    golden_dataset_path: Path = field(
        default_factory=lambda: Path(
            _env_str("DNS_PROMO_GOLDEN_DATASET", "data/golden_validation.parquet")
        )
    )
    rolling_days: int = field(
        default_factory=lambda: _env_int("DNS_PROMO_ROLLING_DAYS", 14)
    )

    # --- Symlink targets (dnstds §10) ---
    tier1_model_symlink: Path = field(
        default_factory=lambda: Path(
            _env_str("DNS_T1_MODEL_SYMLINK", "models/xgboost_tier1_active.json")
        )
    )
    tier2_model_symlink: Path = field(
        default_factory=lambda: Path(
            _env_str("DNS_T2_MODEL_SYMLINK", "models/xgboost_tier2_active.json")
        )
    )


@dataclass
class FeedbackConfig:
    """Configuration for the feedback / retraining data pipeline."""

    feedback_log_dir: Path = field(
        default_factory=lambda: Path(_env_str("DNS_FEEDBACK_LOG_DIR", "data/feedback"))
    )
    parquet_lake_dir: Path = field(
        default_factory=lambda: Path(_env_str("DNS_PARQUET_LAKE_DIR", "data/parquet_lake"))
    )


@dataclass
class AppConfig:
    """
    Root configuration object.
    Composes all subsystem configs into a single injectable object.
    """

    tier1: Tier1Config = field(default_factory=Tier1Config)
    tier2: Tier2Config = field(default_factory=Tier2Config)
    queue: QueueConfig = field(default_factory=QueueConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)

    # --- Network capture ---
    capture_interface: str = field(
        default_factory=lambda: _env_str("DNS_CAPTURE_INTERFACE", "eth0")
    )
    capture_bpf_filter: str = field(
        default_factory=lambda: _env_str("DNS_CAPTURE_BPF_FILTER", "udp port 53 or tcp port 53")
    )

    # --- Process topology (dnstds §7) ---
    # Tier-1 should be pinned to dedicated CPU cores to guarantee latency budget.
    tier1_cpu_affinity: list[int] = field(
        default_factory=lambda: [
            int(c) for c in _env_str("DNS_T1_CPU_AFFINITY", "0,1").split(",") if c.strip()
        ]
    )
    tier2_worker_count: int = field(
        default_factory=lambda: _env_int("DNS_T2_WORKER_COUNT", 2)
    )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config: AppConfig | None = None


def get_config() -> AppConfig:
    """
    Return the global singleton AppConfig, constructing it on first call.
    All environment variable reads happen at first call time.
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reset_config() -> None:
    """
    Force re-construction of the singleton (useful in tests when env vars
    are patched between test cases).
    """
    global _config
    _config = None
