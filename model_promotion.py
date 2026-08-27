"""
model_promotion.py — DNS Tunneling Detector
Safe model promotion pipeline.

Design doc reference: dnstds §10 — Model Promotion Pipeline

Pipeline:
    Cron retraining → Performance Gate → Shadow Inference → Manual Approval → Symlink swap

Performance gate (hard bounds — auto-reject on failure):
    Tier-1: recall ≥ 99.5%,  0% latency regression vs live model
    Tier-2: precision ≥ 98.0%, FPR degradation < 0.1%
    Latency: >5% regression over budget kills the build immediately

Shadow inference:
    Candidate model runs silently against synthetic or replayed traffic
    for `shadow_duration_hours` (default 48h; set DNS_PROMO_SHADOW_HOURS=0
    to skip for local testing).

Atomic promotion:
    On Windows: copy-then-rename (atomic within same volume).
    On Linux/macOS: os.symlink + os.replace (atomic symlink swap).

CLI usage (local testing):
    # Evaluate a candidate Tier-1 model against the golden dataset:
    python model_promotion.py evaluate --tier 1 --candidate models/candidate_t1.json

    # Run shadow inference for N hours then promote:
    python model_promotion.py promote --tier 1 --candidate models/candidate_t1.json

    # Skip shadow period entirely (local dev):
    DNS_PROMO_SHADOW_HOURS=0 python model_promotion.py promote --tier 1 --candidate models/candidate_t1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import (  # type: ignore[import-untyped]
        precision_score,
        recall_score,
        confusion_matrix,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger_preimport = logging.getLogger(__name__)
    logger_preimport.warning(
        "scikit-learn not installed — PerformanceGate will raise RuntimeError. "
        "Install with: pip install scikit-learn"
    )

try:
    import pandas as pd
    import pyarrow.parquet as pq
    _PARQUET_AVAILABLE = True
except ImportError:
    _PARQUET_AVAILABLE = False

from config import get_config, PromotionConfig
from tier1_triage import Tier1Classifier, FEATURE_NAMES as T1_FEATURES, features_to_array as t1_to_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Result of one performance gate evaluation."""
    tier: int                  # 1 or 2
    passed: bool
    candidate_path: Path
    # Tier-1 metrics
    recall: float = 0.0
    # Tier-2 metrics
    precision: float = 0.0
    fpr: float = 0.0
    fpr_delta: float = 0.0    # FPR(candidate) - FPR(champion)
    # Latency
    candidate_latency_ms: float = 0.0
    champion_latency_ms: float = 0.0
    latency_regression_pct: float = 0.0
    # Failure reason (if not passed)
    failure_reason: str = ""

    def summary(self) -> str:
        lines = [
            f"=== Gate Result: Tier-{self.tier} {'PASSED ✅' if self.passed else 'FAILED ❌'} ===",
            f"  Candidate : {self.candidate_path}",
        ]
        if self.tier == 1:
            lines += [
                f"  Recall    : {self.recall:.4f}  (min {get_config().promotion.tier1_min_recall:.4f})",
            ]
        else:
            lines += [
                f"  Precision : {self.precision:.4f}  (min {get_config().promotion.tier2_min_precision:.4f})",
                f"  FPR delta : {self.fpr_delta:+.6f}  (max {get_config().promotion.tier2_max_fpr_delta:.6f})",
            ]
        lines += [
            f"  Latency   : candidate={self.candidate_latency_ms:.3f}ms  "
            f"champion={self.champion_latency_ms:.3f}ms  "
            f"regression={self.latency_regression_pct:+.1f}%  "
            f"(max 5.0%)",
        ]
        if not self.passed:
            lines.append(f"  Reason    : {self.failure_reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Performance Gate
# ---------------------------------------------------------------------------

class PerformanceGate:
    """
    Evaluates a candidate model against hard performance bounds.

    Loads the validation dataset (golden Parquet + optional rolling 14-day
    data from the Parquet lake) and measures recall/precision/FPR and
    per-sample latency.

    For local testing without a real dataset, call evaluate_synthetic()
    which generates a synthetic test set from random features.
    """

    def __init__(self, cfg: Optional[PromotionConfig] = None) -> None:
        self._cfg = cfg or get_config().promotion

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_tier1(self, candidate_path: Path) -> GateResult:
        """
        Gate check for a Tier-1 candidate model.

        Loads the golden validation dataset (Parquet), scores it with
        both the candidate and the current champion, then checks:
          - recall ≥ 99.5%
          - latency regression ≤ 5%
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for model evaluation.")

        result = GateResult(tier=1, passed=False, candidate_path=candidate_path)

        # Load validation data
        X, y = self._load_validation_data(tier=1)
        if X is None:
            logger.warning("No validation dataset found — using synthetic data for local testing.")
            X, y = self._synthetic_dataset(n=2000, tier=1)

        # Load models
        candidate = self._load_tier1_model(candidate_path)
        champion_path = self._cfg.tier1_model_symlink
        champion = self._load_tier1_model(champion_path) if champion_path.exists() else None

        # Score with candidate
        y_pred_candidate, latency_candidate = self._score_tier1(candidate, X)
        result.recall = recall_score(y, y_pred_candidate, zero_division=0)
        result.candidate_latency_ms = latency_candidate

        # Score with champion (for latency baseline)
        if champion is not None:
            _, latency_champion = self._score_tier1(champion, X)
            result.champion_latency_ms = latency_champion
            if latency_champion > 0:
                result.latency_regression_pct = (
                    (latency_candidate - latency_champion) / latency_champion * 100
                )
        else:
            # No champion yet — use budget as baseline
            result.champion_latency_ms = self._cfg.tier1_model_symlink and 1.0 or 1.0
            result.latency_regression_pct = 0.0

        # Apply gates
        if result.recall < self._cfg.tier1_min_recall:
            result.failure_reason = (
                f"recall {result.recall:.4f} < required {self._cfg.tier1_min_recall:.4f}"
            )
        elif result.latency_regression_pct > self._cfg.max_latency_regression_pct:
            result.failure_reason = (
                f"latency regression {result.latency_regression_pct:+.1f}% "
                f"> allowed {self._cfg.max_latency_regression_pct:.1f}%"
            )
        else:
            result.passed = True

        logger.info("%s", result.summary())
        return result

    def evaluate_tier2(self, candidate_path: Path) -> GateResult:
        """
        Gate check for a Tier-2 candidate model.

        Checks:
          - precision ≥ 98.0%
          - FPR degradation < 0.1% vs champion
          - latency regression ≤ 5%
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for model evaluation.")

        result = GateResult(tier=2, passed=False, candidate_path=candidate_path)

        X, y = self._load_validation_data(tier=2)
        if X is None:
            logger.warning("No validation dataset found — using synthetic data for local testing.")
            X, y = self._synthetic_dataset(n=2000, tier=2)

        from tier2_analysis import Tier2Classifier  # noqa: PLC0415

        candidate = self._load_tier2_model(candidate_path)
        champion_path = self._cfg.tier2_model_symlink
        champion = self._load_tier2_model(champion_path) if champion_path.exists() else None

        y_pred_candidate, latency_candidate = self._score_tier2(candidate, X)
        result.precision = precision_score(y, y_pred_candidate, zero_division=0)

        # FPR = FP / (FP + TN)
        try:
            tn, fp, fn, tp = confusion_matrix(y, y_pred_candidate, labels=[0, 1]).ravel()
            result.fpr = fp / max(1, fp + tn)
        except ValueError:
            result.fpr = 0.0

        result.candidate_latency_ms = latency_candidate

        if champion is not None:
            y_pred_champ, latency_champion = self._score_tier2(champion, X)
            result.champion_latency_ms = latency_champion
            try:
                tn_c, fp_c, _, _ = confusion_matrix(y, y_pred_champ, labels=[0, 1]).ravel()
                fpr_champion = fp_c / max(1, fp_c + tn_c)
            except ValueError:
                fpr_champion = 0.0
            result.fpr_delta = result.fpr - fpr_champion
            if latency_champion > 0:
                result.latency_regression_pct = (
                    (latency_candidate - latency_champion) / latency_champion * 100
                )

        # Apply gates
        if result.precision < self._cfg.tier2_min_precision:
            result.failure_reason = (
                f"precision {result.precision:.4f} < required {self._cfg.tier2_min_precision:.4f}"
            )
        elif result.fpr_delta > self._cfg.tier2_max_fpr_delta:
            result.failure_reason = (
                f"FPR delta {result.fpr_delta:.6f} > allowed {self._cfg.tier2_max_fpr_delta:.6f}"
            )
        elif result.latency_regression_pct > self._cfg.max_latency_regression_pct:
            result.failure_reason = (
                f"latency regression {result.latency_regression_pct:+.1f}% "
                f"> allowed {self._cfg.max_latency_regression_pct:.1f}%"
            )
        else:
            result.passed = True

        logger.info("%s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def _load_tier1_model(self, path: Path) -> Tier1Classifier:
        from tier1_triage import Tier1Classifier  # noqa: PLC0415
        clf = Tier1Classifier()
        if path.exists():
            clf._model.load_model(str(path))
        return clf

    def _load_tier2_model(self, path: Path):
        from tier2_analysis import Tier2Classifier  # noqa: PLC0415
        clf = Tier2Classifier()
        if path.exists():
            clf._model.load_model(str(path))
        return clf

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_tier1(self, clf: Tier1Classifier, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Score X and return (binary predictions, mean latency ms/sample)."""
        latencies = []
        preds = []
        for row in X:
            features = dict(zip(T1_FEATURES, row))
            t0 = time.perf_counter()
            prob = clf.predict_proba(features)
            latencies.append((time.perf_counter() - t0) * 1000)
            preds.append(1 if prob >= 0.5 else 0)
        return np.array(preds), float(np.mean(latencies))

    def _score_tier2(self, clf, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Score 8-feature Tier-2 vectors and return (binary predictions, mean latency ms)."""
        from tier2_analysis import FEATURE_NAMES as T2_FEATURES  # noqa: PLC0415
        latencies = []
        preds = []
        for row in X:
            features = dict(zip(T2_FEATURES, row))
            event = {"stateful_features": features, "event_id": "", "tier2_mode": "TRIAGE"}
            t0 = time.perf_counter()
            result = clf.evaluate(event)
            latencies.append((time.perf_counter() - t0) * 1000)
            preds.append(1 if result["verdict"] == "VERIFIED_CRITICAL" else 0)
        return np.array(preds), float(np.mean(latencies))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_validation_data(self, tier: int):
        """
        Load golden validation Parquet. Returns (X, y) numpy arrays or
        (None, None) if the file doesn't exist.
        """
        if not _PARQUET_AVAILABLE:
            return None, None
        path = self._cfg.golden_dataset_path
        if not path.exists():
            return None, None
        try:
            df = pq.read_table(str(path)).to_pandas()
            label_col = "label"
            if label_col not in df.columns:
                return None, None
            if tier == 1:
                feat_cols = T1_FEATURES
            else:
                from tier2_analysis import FEATURE_NAMES as T2_FEATURES  # noqa: PLC0415
                feat_cols = T2_FEATURES
            missing = [c for c in feat_cols if c not in df.columns]
            if missing:
                logger.warning("Validation dataset missing columns: %s", missing)
                return None, None
            X = df[feat_cols].values.astype(np.float32)
            y = (df[label_col] == "CONFIRMED_CRITICAL").astype(int).values
            return X, y
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load validation dataset from %s", path)
            return None, None

    @staticmethod
    def _synthetic_dataset(n: int = 2000, tier: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic balanced test set for local testing.
        Tunnelling samples have high entropy / long FQDNs.
        """
        rng = np.random.default_rng(seed=42)
        if tier == 1:
            n_features = 4
        else:
            n_features = 8

        # Benign samples
        X_benign = rng.uniform(0, 0.5, size=(n // 2, n_features)).astype(np.float32)
        # Tunnelling samples (exaggerated values)
        X_tunnel = rng.uniform(0.5, 1.0, size=(n // 2, n_features)).astype(np.float32)
        if tier == 1:
            # High fqdn_length, entropy for tunnelling
            X_tunnel[:, 0] = rng.uniform(180, 253, n // 2)  # fqdn_length
            X_tunnel[:, 2] = rng.uniform(3.5, 4.0, n // 2)  # shannon_entropy
        X = np.vstack([X_benign, X_tunnel])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        return X, y


# ---------------------------------------------------------------------------
# Shadow Inference Runner
# ---------------------------------------------------------------------------

class ShadowRunner:
    """
    Runs a candidate model silently against synthetic traffic for
    `shadow_duration_hours` hours.

    During shadow mode:
      - The candidate scores every packet in parallel with the champion.
      - Deviations (different verdict) are logged to shadow_log_path.
      - No alerts are sent.

    For local testing, set DNS_PROMO_SHADOW_HOURS=0 to skip entirely.
    """

    TICK_INTERVAL_SEC: float = 5.0   # how often to log progress
    SYNTHETIC_QPS: int = 1_000       # synthetic packets/sec for local testing

    def __init__(
        self,
        candidate_path: Path,
        tier: int,
        shadow_log_path: Optional[Path] = None,
        cfg: Optional[PromotionConfig] = None,
    ) -> None:
        self._candidate_path = candidate_path
        self._tier = tier
        self._cfg = cfg or get_config().promotion
        self._shadow_log_path = shadow_log_path or Path(
            f"data/shadow/shadow_log_tier{tier}.jsonl"
        )
        self._shadow_log_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """
        Run shadow inference for the configured duration.

        Returns:
            Summary dict with deviation_count, total_scored, duration_sec.
        """
        duration_hours = self._cfg.shadow_duration_hours
        duration_sec = duration_hours * 3600

        if duration_sec == 0:
            logger.info(
                "Shadow inference skipped (DNS_PROMO_SHADOW_HOURS=0 — local testing mode)."
            )
            return {"deviation_count": 0, "total_scored": 0, "duration_sec": 0, "skipped": True}

        logger.info(
            "Shadow inference starting for Tier-%d candidate — duration=%dh  "
            "log=%s",
            self._tier, duration_hours, self._shadow_log_path,
        )

        start = time.time()
        deadline = start + duration_sec
        deviation_count = 0
        total_scored = 0
        last_log = start

        gate = PerformanceGate(self._cfg)
        if self._tier == 1:
            candidate = gate._load_tier1_model(self._candidate_path)
            champion = (
                gate._load_tier1_model(self._cfg.tier1_model_symlink)
                if self._cfg.tier1_model_symlink.exists()
                else None
            )
        else:
            candidate = gate._load_tier2_model(self._candidate_path)
            champion = (
                gate._load_tier2_model(self._cfg.tier2_model_symlink)
                if self._cfg.tier2_model_symlink.exists()
                else None
            )

        rng = np.random.default_rng()

        while time.time() < deadline:
            # Generate a synthetic batch
            X_batch, _ = PerformanceGate._synthetic_dataset(n=self.SYNTHETIC_QPS, tier=self._tier)

            for row in X_batch:
                if self._tier == 1:
                    features = dict(zip(T1_FEATURES, row))
                    cand_score = candidate.predict_proba(features)
                    cand_verdict = "CRITICAL" if cand_score >= 0.5 else "BENIGN"
                    if champion:
                        champ_score = champion.predict_proba(features)
                        champ_verdict = "CRITICAL" if champ_score >= 0.5 else "BENIGN"
                        if cand_verdict != champ_verdict:
                            deviation_count += 1
                            self._log_deviation(features, cand_verdict, champ_verdict)
                else:
                    from tier2_analysis import FEATURE_NAMES as T2  # noqa: PLC0415
                    features = dict(zip(T2, row))
                    event = {"stateful_features": features, "event_id": "", "tier2_mode": "TRIAGE"}
                    cand_result = candidate.evaluate(event)
                    cand_verdict = cand_result["verdict"]
                    if champion:
                        champ_result = champion.evaluate(event)
                        if cand_verdict != champ_result["verdict"]:
                            deviation_count += 1
                            self._log_deviation(features, cand_verdict, champ_result["verdict"])
                total_scored += 1

            now = time.time()
            if now - last_log >= 60:
                elapsed_min = (now - start) / 60
                remain_min = (deadline - now) / 60
                logger.info(
                    "Shadow Tier-%d: elapsed=%.0fm  remaining=%.0fm  "
                    "scored=%d  deviations=%d",
                    self._tier, elapsed_min, remain_min, total_scored, deviation_count,
                )
                last_log = now

            # Simulate real-time pace
            time.sleep(self.TICK_INTERVAL_SEC)

        duration_actual = time.time() - start
        deviation_rate = deviation_count / max(1, total_scored)
        summary = {
            "deviation_count":  deviation_count,
            "deviation_rate":   deviation_rate,
            "total_scored":     total_scored,
            "duration_sec":     duration_actual,
            "skipped":          False,
        }
        logger.info(
            "Shadow inference complete: scored=%d  deviations=%d (%.4f%%)  log=%s",
            total_scored, deviation_count, deviation_rate * 100, self._shadow_log_path,
        )
        return summary

    def _log_deviation(self, features: dict, cand_verdict: str, champ_verdict: str) -> None:
        """Append one deviation record to the shadow log."""
        try:
            import json  # noqa: PLC0415
            with open(self._shadow_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": time.time(),
                    "features": {k: float(v) for k, v in features.items()},
                    "candidate": cand_verdict,
                    "champion": champ_verdict,
                }) + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Atomic Promotion
# ---------------------------------------------------------------------------

def promote_model(candidate_path: Path, target_symlink: Path) -> str:
    """
    Atomically promote a candidate model to the active symlink.
    Computes and logs SHA-256 checksum for audit and integrity verification.

    Strategy:
      Linux/macOS: create a temp symlink in the same directory, then
                   os.replace() it over the target — atomic on POSIX.
      Windows:     copy candidate to a temp file in the same directory,
                   then os.replace() — atomic within the same volume.

    Args:
        candidate_path: Path to the trained candidate model file.
        target_symlink: Path of the active model symlink/file to replace.

    Returns:
        SHA-256 checksum string of the promoted model.

    Raises:
        FileNotFoundError: if candidate_path does not exist.
        ValueError: if candidate_path is not a regular file.
    """
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate model not found: {candidate_path}")
    if not candidate_path.is_file():
        raise ValueError(f"Candidate model path must be a regular file: {candidate_path}")

    # Compute SHA-256 checksum
    hasher = hashlib.sha256()
    with open(candidate_path, "rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
    sha256_digest = hasher.hexdigest()

    target_symlink.parent.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        # Windows: atomic copy-rename (symlinks require admin rights)
        tmp_path = target_symlink.with_suffix(".tmp")
        shutil.copy2(str(candidate_path), str(tmp_path))
        os.replace(str(tmp_path), str(target_symlink))
        logger.info(
            "Model promoted (Windows copy-rename): %s → %s [SHA256: %s]",
            candidate_path, target_symlink, sha256_digest,
        )
    else:
        # POSIX: atomic symlink swap
        tmp_link = target_symlink.with_suffix(".tmp_link")
        if tmp_link.exists():
            tmp_link.unlink()
        os.symlink(candidate_path.resolve(), tmp_link)
        os.replace(str(tmp_link), str(target_symlink))
        logger.info(
            "Model promoted (atomic symlink): %s → %s [SHA256: %s]",
            candidate_path.resolve(), target_symlink, sha256_digest,
        )

    return sha256_digest


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

class PromotionPipeline:
    """
    Full promotion workflow:
        1. Performance gate evaluation
        2. Shadow inference (configurable duration — 0 = skip for local testing)
        3. Manual confirmation prompt
        4. Atomic symlink promotion

    Usage:
        pipeline = PromotionPipeline()
        pipeline.run(tier=1, candidate_path=Path("models/candidate_t1.json"))
    """

    def __init__(self, cfg: Optional[PromotionConfig] = None) -> None:
        self._cfg = cfg or get_config().promotion
        self._gate = PerformanceGate(self._cfg)

    def run(
        self,
        tier: int,
        candidate_path: Path,
        skip_confirm: bool = False,
    ) -> bool:
        """
        Run the full promotion pipeline for one model tier.

        Args:
            tier:           1 or 2.
            candidate_path: Path to the candidate model JSON file.
            skip_confirm:   Skip the interactive manual approval prompt.
                            Set True in CI or when DNS_PROMO_SHADOW_HOURS=0.

        Returns:
            True if the model was promoted, False if rejected at any gate.
        """
        logger.info("=== Promotion Pipeline: Tier-%d ===", tier)
        logger.info("Candidate: %s", candidate_path)

        # Step 1: Performance gate
        logger.info("Step 1/3: Performance Gate Evaluation")
        if tier == 1:
            gate_result = self._gate.evaluate_tier1(candidate_path)
            target_symlink = self._cfg.tier1_model_symlink
        else:
            gate_result = self._gate.evaluate_tier2(candidate_path)
            target_symlink = self._cfg.tier2_model_symlink

        if not gate_result.passed:
            logger.error("Gate FAILED — promotion aborted. Reason: %s", gate_result.failure_reason)
            return False

        # Step 2: Shadow inference
        logger.info("Step 2/3: Shadow Inference")
        runner = ShadowRunner(candidate_path, tier=tier, cfg=self._cfg)
        shadow_summary = runner.run()

        logger.info(
            "Shadow complete: scored=%d  deviations=%d  skipped=%s",
            shadow_summary["total_scored"],
            shadow_summary["deviation_count"],
            shadow_summary.get("skipped", False),
        )

        # Step 3: Manual approval
        if not skip_confirm:
            logger.info("Step 3/3: Manual Approval")
            print("\n" + gate_result.summary())
            print(f"\nShadow deviations: {shadow_summary['deviation_count']}")
            answer = input(
                f"\nPromote Tier-{tier} candidate to {target_symlink}? [yes/no]: "
            ).strip().lower()
            if answer not in ("yes", "y"):
                logger.info("Promotion declined by operator.")
                return False
        else:
            logger.info("Step 3/3: Manual approval skipped (skip_confirm=True).")

        # Step 4: Atomic promotion
        promote_model(candidate_path, target_symlink)
        logger.info("✅ Tier-%d model promoted successfully.", tier)
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vespera IDS — Safe Model Promotion Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # evaluate sub-command
    ev = sub.add_parser("evaluate", help="Run performance gate only (no promotion).")
    ev.add_argument("--tier", type=int, choices=[1, 2], required=True)
    ev.add_argument("--candidate", type=Path, required=True, help="Path to candidate model JSON.")

    # promote sub-command
    pr = sub.add_parser("promote", help="Full pipeline: gate + shadow + approve + swap.")
    pr.add_argument("--tier", type=int, choices=[1, 2], required=True)
    pr.add_argument("--candidate", type=Path, required=True, help="Path to candidate model JSON.")
    pr.add_argument(
        "--yes", action="store_true",
        help="Skip manual confirmation (useful for CI / local testing).",
    )

    # shadow sub-command
    sh = sub.add_parser("shadow", help="Run shadow inference only (no gate, no promotion).")
    sh.add_argument("--tier", type=int, choices=[1, 2], required=True)
    sh.add_argument("--candidate", type=Path, required=True)

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()
    cfg = get_config().promotion

    if args.command == "evaluate":
        gate = PerformanceGate(cfg)
        if args.tier == 1:
            result = gate.evaluate_tier1(args.candidate)
        else:
            result = gate.evaluate_tier2(args.candidate)
        print(result.summary())
        sys.exit(0 if result.passed else 1)

    elif args.command == "promote":
        pipeline = PromotionPipeline(cfg)
        success = pipeline.run(
            tier=args.tier,
            candidate_path=args.candidate,
            skip_confirm=args.yes,
        )
        sys.exit(0 if success else 1)

    elif args.command == "shadow":
        runner = ShadowRunner(args.candidate, tier=args.tier, cfg=cfg)
        summary = runner.run()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import json
    main()
