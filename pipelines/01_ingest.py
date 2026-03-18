"""
Pipeline step 1 — Data Ingestion

Fetches macro data from FRED and multpl.com, merges into one wide DataFrame,
and writes data/raw/macro_raw.parquet.

Run:
    python pipelines/01_ingest.py
"""

from pathlib import Path

# Prefer the installed package; fall back to ./src for local runs.
try:
    import trading_crab_lib as crab  # noqa: F401
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import trading_crab_lib as crab  # type: ignore[no-redef]
from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.config import load, setup_logging
from trading_crab_lib.ingestion import fred as fred_module
from trading_crab_lib.ingestion import multpl as multpl_module

import pandas as pd


def main(argv=None) -> None:
    setup_logging()
    cfg = load(settings_path=(crab.CONFIG_DIR / "settings.yaml") if crab.CONFIG_DIR else None)

    # ── FRED ─────────────────────────────────────────────────────────────
    fred_df = fred_module.fetch_all(cfg)

    # ── multpl.com ────────────────────────────────────────────────────────
    multpl_df = multpl_module.fetch_all(cfg)

    # ── Merge ─────────────────────────────────────────────────────────────
    if not multpl_df.empty:
        combined = fred_df.join(multpl_df, how="outer")
    else:
        combined = fred_df

    # Filter to configured date range
    start = cfg["data"]["start_date"]
    combined = combined[combined.index >= start]

    # Persist
    out_path = crab.DATA_DIR / "raw" / "macro_raw.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)
    CheckpointManager().save(combined, "macro_raw")
    print(f"Wrote {len(combined)} rows × {len(combined.columns)} cols → {out_path}")


if __name__ == "__main__":
    main()
