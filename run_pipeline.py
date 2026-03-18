"""
run_pipeline.py — local orchestration entrypoint for end-to-end testing.

This script exists to let you run the full pipeline inside this repo to
materialize `data/checkpoints/*` so that `pytest -v` can exercise constraints
tests end-to-end.

Important:
- This file is NOT part of the published PyPI package. `MANIFEST.in` excludes it.
- Likewise, `pipelines/` and `notebooks/` are excluded from sdists/wheels.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent / "src"))

import trading_crab_lib as crab
from trading_crab_lib.config import load, setup_logging
from trading_crab_lib.runtime import RunConfig


STEPS: dict[int, tuple[str, str]] = {
    1: ("Ingest macro data", "pipelines.01_ingest"),
    2: ("Engineer features", "pipelines.02_features"),
    3: ("PCA + clustering", "pipelines.03_cluster"),
    4: ("Regime profiling + labeling", "pipelines.04_regime_label"),
    5: ("Supervised prediction", "pipelines.05_predict"),
    6: ("Asset returns", "pipelines.06_asset_returns"),
    7: ("Dashboard", "pipelines.07_dashboard"),
    8: ("Diagnostics", "pipelines.08_diagnostics"),
    9: ("Tactics signals", "pipelines.09_tactics"),
}


def _parse_steps(raw: str | None) -> set[int]:
    if not raw:
        return set(STEPS.keys())
    requested = {int(s.strip()) for s in raw.split(",") if s.strip()}
    invalid = requested - set(STEPS.keys())
    if invalid:
        raise SystemExit(f"Unknown step numbers: {sorted(invalid)}. Valid: {sorted(STEPS)}")
    return requested


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trading Crab pipeline runner (local)")
    p.add_argument("--steps", type=str, default=None, help="Comma-separated step numbers, e.g. 1,2,6")
    p.add_argument("--refresh", action="store_true", help="Refresh source datasets (step 1)")
    p.add_argument("--recompute", action="store_true", help="Recompute derived datasets (step 2)")
    p.add_argument("--refresh-assets", action="store_true", help="Refresh ETF prices (step 6)")
    p.add_argument("--plots", action="store_true", help="Generate plots (where supported)")
    p.add_argument("--show-plots", action="store_true", help="Show plots interactively")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--no-constrained", action="store_true", help="Skip constrained kmeans if installed")
    p.add_argument("--no-drop-tail", action="store_true", help="Do not drop incomplete trailing quarter")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    setup_logging()
    run_cfg = RunConfig.from_args(args)
    run_cfg.apply_logging()

    # Ensure config is loadable early (will raise if config/settings.yaml missing)
    _ = load()

    requested = _parse_steps(args.steps)
    print(f"\nTrading-Crab pipeline runner [{run_cfg}]")
    print(f"Repo ROOT: {crab.ROOT}")
    print(f"DATA_DIR:  {crab.DATA_DIR}")
    print(f"Steps:     {sorted(requested)}\n")

    for step in sorted(requested):
        label, module_name = STEPS[step]
        print(f"── Step {step}: {label} ──")
        mod = importlib.import_module(module_name)

        # Step 6 supports a CLI flag; emulate its argv contract.
        if step == 6 and getattr(mod, "main", None):
            old_argv = sys.argv[:]
            try:
                sys.argv = [sys.argv[0]] + (["--refresh-assets"] if args.refresh_assets else [])
                mod.main()
            finally:
                sys.argv = old_argv
        else:
            mod.main()

        print("   ✓ done\n")

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

