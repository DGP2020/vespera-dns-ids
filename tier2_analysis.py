"""
tier2_analysis.py — DNS Tunneling Detector
Tier-2 stateful feature extractor, XGBoost classifier, and TreeSHAP XAI engine.

Design doc reference: dnstds §4 — Tier-2 Deep Analysis
                       dnstds §11.5 — Explainable AI (XAI) Architecture

Features (stateful, computed from the 30-second window cache):
  Inherited base features (from Tier-1 payload — not recomputed):
    - fqdn_length, subdomain_count, shannon_entropy, numeric_alpha_ratio

  Incremental stateful features:
    - record_type_heterogeneity : TXT+CNAME+AAAA ratio over total queries
    - payload_variance          : delta variance of successive payload lengths
    - ngram_transition_score    : n-gram character matrix deviation from baseline
    - unique_ip_concurrency     : distinct client IPs querying same anomalous zone

XAI: TreeSHAP via shap.TreeExplainer over all 8 combined features.
     Computed asynchronously — never on the Tier-1 inline path.

Verdicts:
    VERIFIED_CRITICAL      → alert with SHAP attribution report
    FALSE_POSITIVE_ANOMALY → retained for retraining, no alert
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    xgb = None  # type: ignore[assignment]

try:
    import shap  # type: ignore[import-untyped]
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    shap = None  # type: ignore[assignment]

from config import get_config, Tier2Config
from state_cache import StateCache, WindowEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------

VERDICT_CRITICAL = "VERIFIED_CRITICAL"
VERDICT_FP = "FALSE_POSITIVE_ANOMALY"

# ---------------------------------------------------------------------------
# Combined feature column order — must match training schema
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    # Tier-1 base (stateless) — inherited from queue payload
    "fqdn_length",
    "subdomain_count",
    "shannon_entropy",
    "numeric_alpha_ratio",
    # Tier-2 stateful — computed here
    "record_type_heterogeneity",
    "payload_variance",
    "ngram_transition_score",
    "unique_ip_concurrency",
]


# ---------------------------------------------------------------------------
# N-gram character transition matrix helper
# ---------------------------------------------------------------------------

def _build_ngram_freq(text: str, n: int = 2) -> dict[str, float]:
    """
    Build a character n-gram frequency distribution from `text`.
    Returns dict mapping n-gram string → relative frequency.
    """
    if len(text) < n:
        return {}
    ngrams: dict[str, int] = {}
    for i in range(len(text) - n + 1):
        gram = text[i: i + n]
        ngrams[gram] = ngrams.get(gram, 0) + 1
    total = sum(ngrams.values())
    return {gram: count / total for gram, count in ngrams.items()}


def _ngram_kl_divergence(observed: dict[str, float], baseline: dict[str, float]) -> float:
    """
    Compute KL divergence D_KL(observed ‖ baseline) as a deviation score.
    A high value means the domain's character distribution is unusual vs.
    the baseline (normal English/Internet domain corpus).

    Uses smoothed baseline probabilities to avoid division by zero.
    """
    score = 0.0
    smoothing = 1e-9
    for gram, p in observed.items():
        q = baseline.get(gram, smoothing)
        if p > 0 and q > 0:
            score += p * math.log(p / q)
    return max(0.0, score)


# Approximate bigram baseline derived from a representative Internet domain
# corpus (Alexa/Tranco top-1M). These are the most common letter pairs in
# legitimate domain names. Tunneling traffic tends to use random base32/64
# characters which deviate strongly from this distribution.
_DOMAIN_BIGRAM_BASELINE: dict[str, float] = {
    "co": 0.031, "om": 0.028, "in": 0.027, "an": 0.024, "er": 0.022,
    "en": 0.020, "on": 0.019, "al": 0.018, "at": 0.018, "ar": 0.017,
    "es": 0.017, "re": 0.016, "or": 0.016, "ng": 0.015, "ed": 0.014,
    "le": 0.014, "io": 0.013, "ti": 0.013, "de": 0.013, "ne": 0.012,
    "te": 0.012, "is": 0.011, "ha": 0.010, "nt": 0.010, "ma": 0.009,
    "me": 0.009, "li": 0.009, "ra": 0.009, "st": 0.009, "it": 0.008,
    "se": 0.008, "ta": 0.008, "ro": 0.008, "la": 0.008, "ri": 0.007,
    "si": 0.007, "ac": 0.007, "ad": 0.007, "ca": 0.007, "et": 0.006,
    "ic": 0.006, "un": 0.006, "he": 0.006, "lo": 0.006, "ge": 0.006,
}


def compute_ngram_score(query_name: str, n: int = 2) -> float:
    """
    Compute the KL-divergence-based n-gram deviation score for a DNS query.
    A score near 0 suggests a legitimate domain name character distribution.
    High scores indicate random/encoded strings typical of DNS tunneling.

    Args:
        query_name: Full FQDN string (already normalised to lower-case).
        n:          N-gram size (default 2 = bigrams).

    Returns:
        Non-negative float deviation score.
    """
    observed = _build_ngram_freq(query_name, n=n)
    if not observed:
        return 0.0
    return _ngram_kl_divergence(observed, _DOMAIN_BIGRAM_BASELINE)


# ---------------------------------------------------------------------------
# Stateful feature extractor
# ---------------------------------------------------------------------------

def extract_stateful_features(
    base_features: dict,
    state_cache: StateCache,
    parent_domain: str,
    client_ip: str,
    ts: Optional[float] = None,
    cfg: Optional[Tier2Config] = None,
) -> dict:
    """
    Extract the four stateful Tier-2 features from the sliding window cache.
    Combines them with the four Tier-1 base features already packed in the
    queue payload (shared-base/extended-branch pipeline — dnstds §4).

    Args:
        base_features:  Dict from tier1_triage.extract_stateless_features(),
                        already carried in the queue payload.
        state_cache:    Live StateCache instance for this Tier-2 worker.
        parent_domain:  Extracted second-level domain + TLD (e.g. "evil.com").
        client_ip:      Source IP of the DNS client.
        ts:             Snapshot timestamp (default: now).
        cfg:            Tier2Config override (uses global if None).

    Returns:
        Dict with all 8 feature values (keys match FEATURE_NAMES order),
        plus 'parent_domain' and 'client_ip' for traceability.
    """
    _cfg = cfg or get_config().tier2
    now = ts or time.time()

    # Retrieve the window snapshot for this key
    snap: Optional[WindowEntry] = state_cache.snapshot(parent_domain, client_ip, ts=now)

    # --- record_type_heterogeneity (dnstds §4) ---
    if snap is not None and snap.total_queries() > 0:
        suspicious_types = {"TXT", "CNAME", "AAAA"}
        suspicious_count = sum(
            count for rtype, count in snap.record_type_counts.items()
            if rtype in suspicious_types
        )
        record_type_heterogeneity = suspicious_count / snap.total_queries()
    else:
        record_type_heterogeneity = 0.0

    # --- payload_variance (dnstds §4) ---
    # Delta variance of successive payload lengths (second-order feature)
    if snap is not None and len(snap.payload_lengths) >= 2:
        deltas = [
            abs(snap.payload_lengths[i] - snap.payload_lengths[i - 1])
            for i in range(1, len(snap.payload_lengths))
        ]
        payload_variance = float(np.var(deltas)) if deltas else 0.0
    else:
        payload_variance = 0.0

    # --- ngram_transition_score (dnstds §4) ---
    query_name = base_features.get("query_name", parent_domain)
    ngram_transition_score = compute_ngram_score(query_name, n=_cfg.ngram_size)

    # --- unique_ip_concurrency (dnstds §4) ---
    unique_ip_concurrency = float(state_cache.zone_unique_ip_count(parent_domain))

    return {
        # --- Inherited Tier-1 base features ---
        "fqdn_length":             base_features.get("fqdn_length", 0),
        "subdomain_count":         base_features.get("subdomain_count", 0),
        "shannon_entropy":         base_features.get("shannon_entropy", 0.0),
        "numeric_alpha_ratio":     base_features.get("numeric_alpha_ratio", 0.0),
        # --- Tier-2 stateful features ---
        "record_type_heterogeneity": record_type_heterogeneity,
        "payload_variance":           payload_variance,
        "ngram_transition_score":     ngram_transition_score,
        "unique_ip_concurrency":      unique_ip_concurrency,
        # --- Metadata ---
        "parent_domain": parent_domain,
        "client_ip":     client_ip,
    }


def features_to_array(features: dict) -> np.ndarray:
    """
    Convert an 8-feature dict (from extract_stateful_features) to a
    (1, 8) numpy array in FEATURE_NAMES column order.
    """
    return np.array([[features[k] for k in FEATURE_NAMES]], dtype=np.float32)


# ---------------------------------------------------------------------------
# TreeSHAP XAI engine
# ---------------------------------------------------------------------------

def compute_shap_attributions(
    model: Any,
    features: dict,
    top_n: int = 5,
) -> list[dict]:
    """
    Compute TreeSHAP feature attributions for one event using the
    fitted Tier-2 XGBoost model.

    Runs asynchronously in the Tier-2 consumer worker — never on the
    Tier-1 inline path (dnstds §11.5).

    Args:
        model:    Fitted xgb.XGBClassifier instance.
        features: 8-feature dict from extract_stateful_features().
        top_n:    Number of top attributions to return (default 5).

    Returns:
        Ranked list of dicts:
            [{"feature": str, "shap_value": float, "raw_value": float}, ...]
        Sorted descending by abs(shap_value). Returns [] if shap unavailable.
    """
    if not _SHAP_AVAILABLE or not _XGB_AVAILABLE:
        logger.warning("shap or xgboost not available — SHAP attributions skipped.")
        return []

    try:
        x = features_to_array(features)
        explainer = shap.TreeExplainer(model)
        # shap_values shape: (n_samples, n_features) for binary classification
        shap_values = explainer.shap_values(x)

        # For XGBClassifier shap returns shape (1, n_features) for class 1
        if isinstance(shap_values, list):
            # Multi-output format: index 1 = malicious class
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        attributions = [
            {
                "feature":    FEATURE_NAMES[i],
                "shap_value": float(sv[i]),
                "raw_value":  float(x[0][i]),
            }
            for i in range(len(FEATURE_NAMES))
        ]

        # Sort by absolute SHAP value descending
        attributions.sort(key=lambda a: abs(a["shap_value"]), reverse=True)
        return attributions[:top_n]

    except Exception:  # noqa: BLE001
        logger.exception("TreeSHAP computation failed — returning empty attributions.")
        return []


# ---------------------------------------------------------------------------
# Tier-2 Classifier
# ---------------------------------------------------------------------------

class Tier2Classifier:
    """
    High-precision XGBoost classifier for Tier-2 deep analysis.

    Training objective: maximize precision on a hard-negative dataset that
    includes CDN queries, DKIM/SPF TXT records, and telemetry beacons —
    traffic that fools naive entropy-only classifiers (dnstds §4).

    Includes integrated TreeSHAP XAI explanations on every VERIFIED_CRITICAL
    verdict.
    """

    def __init__(self, cfg: Optional[Tier2Config] = None) -> None:
        self._cfg = cfg or get_config().tier2
        self._model: Optional[Any] = None
        self._load_or_init_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_or_init_model(self) -> None:
        symlink: Path = self._cfg.model_symlink

        if not _XGB_AVAILABLE:
            logger.warning(
                "xgboost not installed — Tier2Classifier.evaluate() will "
                "return FALSE_POSITIVE_ANOMALY for all events."
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
                logger.info("Tier-2 model loaded from %s", symlink.resolve())
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to load Tier-2 model from %s — using uninitialised weights.",
                    symlink,
                )
        else:
            logger.warning(
                "Tier-2 model not found at %s — evaluate() will return "
                "FALSE_POSITIVE_ANOMALY until a model is trained.",
                symlink,
            )

        self._model = model

    @property
    def is_fitted(self) -> bool:
        """Return True if the underlying XGBoost model is loaded/fitted and ready for inference."""
        if self._model is None or not _XGB_AVAILABLE:
            return False
        return getattr(self._model, "_Booster", None) is not None

    # ------------------------------------------------------------------
    # Inference + XAI
    # ------------------------------------------------------------------

    def evaluate(self, event_payload: dict) -> dict:
        """
        Score a Tier-2 event and compute TreeSHAP feature attributions.

        Args:
            event_payload: Queue payload dict containing:
                - 'base_features': dict (from Tier-1)
                - 'stateful_features': dict (from extract_stateful_features)
                - 'event_id', 'tier1_score', 'tier2_mode', etc.

        Returns:
            Result dict with keys:
                event_id         : str
                verdict          : VERIFIED_CRITICAL | FALSE_POSITIVE_ANOMALY
                tier2_score      : float  probability [0.0, 1.0]
                shap_attributions: list[dict]  top-5 TreeSHAP values
                latency_ms       : float  time taken for this evaluation
                features         : dict   the 8-feature vector used
        """
        t0 = time.perf_counter()

        stateful_features: dict = event_payload.get("stateful_features", {})
        event_id: str = event_payload.get("event_id", "")
        tier2_mode: str = event_payload.get("tier2_mode", "TRIAGE")

        if not self.is_fitted:
            return self._default_result(
                event_id, stateful_features, tier2_mode, t0,
                reason="model-not-loaded"
            )

        try:
            x = features_to_array(stateful_features)
            proba = self._model.predict_proba(x)
            tier2_score = float(proba[0][1])

            # Verdict threshold — Tier-2 maximises precision, so use 0.50
            # as a simple decision boundary (tunable via training calibration)
            verdict = (
                VERDICT_CRITICAL
                if tier2_score >= 0.50
                else VERDICT_FP
            )

            # Compute TreeSHAP attributions for VERIFIED_CRITICAL events
            # and for FAST_TRACK AUDIT_ENRICHMENT copies (audit trail).
            shap_attributions: list[dict] = []
            if verdict == VERDICT_CRITICAL or tier2_mode == "AUDIT_ENRICHMENT":
                shap_attributions = compute_shap_attributions(
                    self._model, stateful_features, top_n=5
                )

            latency_ms = (time.perf_counter() - t0) * 1000

            if latency_ms > self._cfg.max_latency_ms:
                logger.warning(
                    "Tier-2 evaluation exceeded latency budget: "
                    "%.1f ms > %.1f ms  event_id=%s",
                    latency_ms, self._cfg.max_latency_ms, event_id,
                )

            logger.info(
                "Tier-2 verdict=%s  score=%.4f  latency=%.1fms  event=%s",
                verdict, tier2_score, latency_ms, event_id,
            )

            return {
                "event_id":          event_id,
                "verdict":           verdict,
                "tier2_score":       tier2_score,
                "tier2_mode":        tier2_mode,
                "shap_attributions": shap_attributions,
                "latency_ms":        latency_ms,
                "features":          stateful_features,
            }

        except Exception:  # noqa: BLE001
            logger.exception("Tier-2 evaluate() raised — falling back to FP verdict.")
            return self._default_result(
                event_id, stateful_features, tier2_mode, t0,
                reason="exception"
            )

    def _default_result(
        self,
        event_id: str,
        features: dict,
        tier2_mode: str,
        t0: float,
        reason: str = "",
    ) -> dict:
        return {
            "event_id":          event_id,
            "verdict":           VERDICT_FP,
            "tier2_score":       0.0,
            "tier2_mode":        tier2_mode,
            "shap_attributions": [],
            "latency_ms":        (time.perf_counter() - t0) * 1000,
            "features":          features,
            "fallback_reason":   reason,
        }

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
    ) -> None:
        """
        Fit the Tier-2 XGBoost classifier.

        Training objective: maximize precision on hard-negative dataset.
        Does NOT use scale_pos_weight — precision matters more than recall here.
        Use eval_metric='aucpr' (area under precision-recall curve) for
        calibration.
        """
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost is not installed.")
        if self._model is None:
            self._load_or_init_model()

        fit_kwargs: dict = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False

        self._model.fit(X, y, **fit_kwargs)
        logger.info("Tier-2 model trained on %d samples.", len(y))

    def save_model(self, path: Path) -> None:
        """Persist the trained Tier-2 model in XGBoost JSON format."""
        if self._model is None:
            raise RuntimeError("No model to save — call train() first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("Tier-2 model saved to %s", path)


# ---------------------------------------------------------------------------
# Utility: extract parent domain from FQDN
# ---------------------------------------------------------------------------

def extract_parent_domain(fqdn: str) -> str:
    """
    Extract the second-level domain + TLD from an FQDN.

    Examples:
        "data.abc123.evil.com"  →  "evil.com"
        "example.co.uk"         →  "co.uk"   (simple split — not PSL-aware)
        "evil.com"              →  "evil.com"
    """
    parts = fqdn.rstrip(".").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return fqdn
