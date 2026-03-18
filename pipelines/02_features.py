"""
Pipeline step 2 — Feature Engineering

Reads data/raw/macro_raw.parquet, applies log transforms, smoothed
derivatives, cross-ratios, and Bernstein gap filling.

Writes two feature files:
  data/processed/features.parquet            — centered rolling windows
                                               (for clustering in step 3-4)
  data/processed/features_supervised.parquet — causal/backward rolling windows
                                               (for supervised learning in step 5-7;
                                               no look-ahead bias)

Run:
    python pipelines/02_features.py
"""

from pathlib import Path

# Prefer the installed package; fall back to ./src for local runs.
try:
    import trading_crab_lib as crab  # noqa: F401
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import trading_crab_lib as crab  # type: ignore[no-redef]
from trading_crab_lib import transforms as _transforms_module
from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.config import load, setup_logging

import pandas as pd


def main() -> None:
    setup_logging()
    cfg = load(settings_path=(crab.CONFIG_DIR / "settings.yaml") if crab.CONFIG_DIR else None)

    raw = pd.read_parquet(crab.DATA_DIR / "raw" / "macro_raw.parquet")
    print(f"Loaded raw data: {raw.shape}")

    out_dir = crab.DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Centered features — for clustering (steps 3-4)
    features = _transforms_module.engineer_all(raw, cfg, causal=False)
    out_path = out_dir / "features.parquet"
    features.to_parquet(out_path)
    cm = CheckpointManager()
    cm.save(features, "features_noncausal")
    print(f"Wrote {features.shape} → {out_path}  (centered)")

    # Causal features — for supervised learning and live scoring (steps 5-7)
    features_sup = _transforms_module.engineer_all(raw, cfg, causal=True)
    out_path_sup = out_dir / "features_supervised.parquet"
    features_sup.to_parquet(out_path_sup)
    cm.save(features_sup, "features_causal")
    print(f"Wrote {features_sup.shape} → {out_path_sup}  (causal/backward)")


if __name__ == "__main__":
    main()
