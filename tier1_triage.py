"""
tier1_triage.py — DNS Tunneling Detector
Tier-1 fast stateless feature extractor and XGBoost classifier.

Design doc reference: dnstds §3 — Tier-1 Fast Triage

Features (computed from a single DNS packet — no state required):
  - fqdn_length         : total character length of the full query name
  - subdomain_count     : number of label separators (dots) in the hostname
  - shannon_entropy     : Shannon entropy of the top-level label
  - numeric_alpha_ratio : ratio of digit characters to alpha characters

Model: XGBoost, max_depth=4, n_estimators=50, tree_method='hist'
Latency budget: ≤1.5 ms/packet  |  Throughput: ≥50,000 QPS

The Tier1Classifier also exposes a benchmark() helper that samples
average per-packet latency so the latency budget can be verified.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False
    xgb = None  # type: ignore[assignment]

from config import get_config, Tier1Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names — order must match training schema
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "fqdn_length",
    "subdomain_count",
    "shannon_entropy",
    "numeric_alpha_ratio",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _shannon_entropy(label: str) -> float:
    """
    Compute Shannon entropy of a string:
        H = -Σ p(c) * log2(p(c))
    where p(c) is the frequency of character c in `label`.
    Returns 0.0 for empty strings.
    """
    if not label:
        return 0.0
    length = len(label)
    freq: dict[str, int] = {}
    for ch in label:
        freq[ch] = freq.get(ch, 0) + 1
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def _numeric_alpha_ratio(s: str) -> float:
    """
    Ratio of digit characters to alpha characters in string `s`.
    Returns 0.0 when there are no alphabetic characters (avoids division by zero).
    """
    digit_count = sum(1 for c in s if c.isdigit())
    alpha_count = sum(1 for c in s if c.isalpha())
    if alpha_count == 0:
        return 0.0
    return digit_count / alpha_count


def extract_stateless_features(
    query_name: str,
    record_type: str = "A",
) -> dict:
    """
    Extract the four stateless Tier-1 features from a single DNS query.

    Args:
        query_name:  Full FQDN string (e.g. "sub.evil.example.com").
                     Trailing dot is stripped if present.
        record_type: DNS record type string (e.g. "A", "TXT").
                     Included in the returned dict for downstream routing /
                     Tier-2 state cache — not used by Tier-1 model directly.

    Returns:
        dict with keys: fqdn_length, subdomain_count, shannon_entropy,
                        numeric_alpha_ratio, query_name, record_type.
    """
    # Normalise: strip trailing dot common in wire-format FQDNs
    fqdn = query_name.rstrip(".").lower()

    labels = fqdn.split(".")
    top_label = labels[0] if labels else ""

    return {
        "query_name": fqdn,
        "record_type": record_type,
        # --- Model features ---
        "fqdn_length": len(fqdn),
        "subdomain_count": max(0, len(labels) - 2),  # exclude TLD and second-level
        "shannon_entropy": _shannon_entropy(top_label),
        "numeric_alpha_ratio": _numeric_alpha_ratio(fqdn),
    }


def features_to_array(features: dict) -> np.ndarray:
    """
    Convert a feature dict (from extract_stateless_features) into a
    2-D numpy array suitable for xgb.DMatrix / XGBClassifier.predict_proba().
    Shape: (1, len(FEATURE_NAMES)).
    """
    return np.array([[features[k] for k in FEATURE_NAMES]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Tier-1 Classifier
# ---------------------------------------------------------------------------

class Tier1Classifier:
    """
    Thin wrapper around an XGBoost binary classifier for fast inline scoring.

    Lifecycle:
        1. Instantiate at process start — loads model from disk once.
        2. Call predict_proba(features) for every DNS query inline.
        3. Never call from the Tier-2 process — it has its own instance.

    The classifier is intentionally stateless: no query history is kept here.
    All state accumulation belongs to Tier-2's StateCache.
    """

    def __init__(self, cfg: Optional[Tier1Config] = None) -> None:
        self._cfg = cfg or get_config().tier1
        self._model: Optional[xgb.XGBClassifier] = None  # type: ignore[type-arg]
        self._model_path: Optional[Path] = None
        self._load_or_init_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_or_init_model(self) -> None:
        """
        Attempt to load a trained model from the configured symlink path.
        If the symlink / file does not exist, initialise an untrained model
        with the correct hyper-parameters so the code is usable even before
        a trained artefact exists (training is handled separately).
        """
        symlink: Path = self._cfg.model_symlink

        if not _XGB_AVAILABLE:
            logger.warning(
                "xgboost is not installed — Tier1Classifier will return "
                "constant probability 0.0 for all queries."
            )
            return

        model = xgb.XGBClassifier(
            max_depth=self._cfg.xgb_max_depth,
            n_estimators=self._cfg.xgb_n_estimators,
            tree_method=self._cfg.xgb_tree_method,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

        if symlink.exists():
            try:
                model.load_model(str(symlink))
                self._model_path = symlink.resolve()
                logger.info("Tier-1 model loaded from %s", self._model_path)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to load Tier-1 model from %s — "
                    "running with uninitialised weights.",
                    symlink,
                )
        else:
            logger.warning(
                "Tier-1 model file not found at %s. "
                "predict_proba() will return 0.0 until a model is trained "
                "and saved to that path.",
                symlink,
            )

        self._model = model

    def reload_model(self) -> None:
        """
        Hot-reload the model from disk — called after an atomic symlink swap
        by the model promotion pipeline (dnstds §10).
        """
        logger.info("Reloading Tier-1 model from %s", self._cfg.model_symlink)
        self._load_or_init_model()

    @property
    def is_fitted(self) -> bool:
        """Return True if the underlying XGBoost model is loaded/fitted and ready for inference."""
        if self._model is None or not _XGB_AVAILABLE:
            return False
        return getattr(self._model, "_Booster", None) is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: dict) -> float:
        """
        Score a single DNS query.

        Args:
            features: dict produced by extract_stateless_features().

        Returns:
            Probability in [0.0, 1.0] that the query is malicious.
            Returns 0.0 if xgboost is unavailable or model not trained.
        """
        if not self.is_fitted:
            return 0.0

        try:
            x = features_to_array(features)
            # XGBClassifier.predict_proba returns shape (n_samples, n_classes)
            # class 1 = malicious
            proba = self._model.predict_proba(x)
            return float(proba[0][1])
        except Exception:  # noqa: BLE001
            logger.exception("Tier-1 predict_proba raised an error; returning 0.0")
            return 0.0

    # ------------------------------------------------------------------
    # Benchmarking helper
    # ------------------------------------------------------------------

    def benchmark(self, n_samples: int = 10_000) -> dict:
        """
        Measure average per-packet inference latency using synthetic
        feature vectors.  Used to verify the ≤1.5 ms/packet budget.

        Args:
            n_samples: Number of synthetic queries to score.

        Returns:
            dict with keys: n_samples, mean_ms, min_ms, max_ms,
                            p99_ms, budget_ms, budget_ok.
        """
        if self._model is None or not _XGB_AVAILABLE:
            logger.warning("Cannot benchmark — xgboost not available.")
            return {}

        rng = np.random.default_rng(seed=42)

        if not self.is_fitted:
            logger.info("Tier-1 model is unfitted — fitting synthetic baseline model for latency benchmarking...")
            X_dummy = rng.uniform(0, 10, size=(100, len(FEATURE_NAMES))).astype(np.float32)
            y_dummy = rng.integers(0, 2, size=100)
            self._model.fit(X_dummy, y_dummy)

        latencies: list[float] = []

        for _ in range(n_samples):
            # Synthetic feature dict that mimics extract_stateless_features output
            features = {
                "fqdn_length": int(rng.integers(8, 253)),
                "subdomain_count": int(rng.integers(0, 6)),
                "shannon_entropy": float(rng.uniform(0, 4)),
                "numeric_alpha_ratio": float(rng.uniform(0, 2)),
            }
            t0 = time.perf_counter()
            self.predict_proba(features)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

        arr = np.array(latencies)
        result = {
            "n_samples": n_samples,
            "mean_ms": float(np.mean(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "p99_ms": float(np.percentile(arr, 99)),
            "budget_ms": self._cfg.max_latency_ms,
            "budget_ok": float(np.mean(arr)) <= self._cfg.max_latency_ms,
        }
        logger.info(
            "Tier-1 benchmark: mean=%.3f ms  p99=%.3f ms  budget=%s ms  ok=%s",
            result["mean_ms"],
            result["p99_ms"],
            result["budget_ms"],
            result["budget_ok"],
        )
        return result

    # ------------------------------------------------------------------
    # Training helper (not used inline — called from offline pipeline)
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
    ) -> None:
        """
        Fit the XGBoost classifier.  Call from the retraining pipeline,
        not from the inline capture loop.

        Training objective: maximise recall (dnstds §3), so use
        scale_pos_weight to handle class imbalance from public-domain data.

        Args:
            X:        Feature matrix, shape (n_samples, 4), column order
                      matches FEATURE_NAMES.
            y:        Binary labels: 0 = benign, 1 = malicious.
            eval_set: Optional list of (X_val, y_val) tuples for early stopping.
        """
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost is not installed.")

        if self._model is None:
            self._load_or_init_model()

        # Scale positive weight to boost recall on imbalanced datasets
        n_benign = int(np.sum(y == 0))
        n_malicious = int(np.sum(y == 1))
        scale_pos_weight = n_benign / max(1, n_malicious)
        self._model.set_params(scale_pos_weight=scale_pos_weight)

        fit_kwargs: dict = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False

        self._model.fit(X, y, **fit_kwargs)
        logger.info(
            "Tier-1 model trained on %d samples (benign=%d, malicious=%d, "
            "scale_pos_weight=%.2f)",
            len(y),
            n_benign,
            n_malicious,
            scale_pos_weight,
        )

    def save_model(self, path: Path) -> None:
        """Persist the trained model to `path` in XGBoost JSON format."""
        if self._model is None:
            raise RuntimeError("No model to save — call train() first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("Tier-1 model saved to %s", path)
