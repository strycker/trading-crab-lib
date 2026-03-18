"""
Pipeline step 9 — Tactics layer (buy-and-hold / swing-trade / stand-aside).

Reads ETF prices and regime labels, computes per-asset volatility/trend/
correlation metrics, classifies tactics labels, and writes a machine-readable
parquet for the weekly report and downstream use.

Run:
    python pipelines/09_tactics.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# Prefer the installed package; fall back to ./src for local runs.
try:
    import trading_crab_lib as crab  # noqa: F401
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import trading_crab_lib as crab  # type: ignore[no-redef]
from trading_crab_lib.config import load, setup_logging
from trading_crab_lib.asset_returns import compute_quarterly_returns
from trading_crab_lib.tactics import compute_tactics_metrics, classify_tactics

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = load()

    prices_path = crab.DATA_DIR / "raw" / "asset_prices.parquet"
    labels_path = crab.DATA_DIR / "regimes" / "cluster_labels.parquet"

    if not prices_path.exists():
        log.warning("ETF prices checkpoint %s not found; skipping tactics step.", prices_path)
        return
    if not labels_path.exists():
        log.warning("Cluster labels %s not found; skipping tactics step.", labels_path)
        return

    prices = pd.read_parquet(prices_path)
    _labels = pd.read_parquet(labels_path)["balanced_cluster"]

    # Tactics are computed on returns, not price levels.
    returns = compute_quarterly_returns(prices)

    # Optional benchmark for correlation (defaults to SPY if present).
    benchmark = returns["SPY"] if "SPY" in returns.columns else None

    tcfg = cfg.get("tactics") or {}
    vol_window = int(tcfg.get("vol_window", 8))
    trend_window = int(tcfg.get("trend_window", 4))
    vol_threshold = float(tcfg.get("vol_threshold", 0.20))
    trend_threshold = float(tcfg.get("trend_threshold", 0.0))

    metrics = compute_tactics_metrics(
        returns=returns,
        benchmark_returns=benchmark,
        vol_window=vol_window,
        trend_window=trend_window,
    )
    tactics_df = classify_tactics(
        metrics=metrics,
        vol_threshold=vol_threshold,
        trend_threshold=trend_threshold,
    )
    if "tactic" in tactics_df.columns and "tactics_label" not in tactics_df.columns:
        tactics_df = tactics_df.rename(columns={"tactic": "tactics_label"})

    out_dir = crab.OUTPUT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tactics_signals.parquet"
    tactics_df.to_parquet(out_path, index=False)
    print(f"Wrote tactics signals → {out_path}")


if __name__ == "__main__":
    main()
