"""
trading_crab_lib — Market Regime Classification & Prediction Pipeline utilities.

This package is functions-only and has no built-in paths to config or data.
Calling code (pipelines, notebooks, other repos) must either:

  - Set environment variables so this package can resolve paths:
      TRADING_CRAB_ROOT       — repo/project root (optional)
      TRADING_CRAB_CONFIG_DIR — config directory (optional; default ROOT/config if ROOT set)
      TRADING_CRAB_DATA_DIR   — data directory (optional; default ROOT/data if ROOT set)
      TRADING_CRAB_OUTPUT_DIR — outputs directory (optional; default ROOT/outputs if ROOT set)

  - Or set these attributes after import:
      import trading_crab_lib as crab
      crab.ROOT = Path("/path/to/your/repo")
      crab.DATA_DIR = crab.ROOT / "data"
      ...

  - And pass explicit paths/params into functions (e.g. load(settings_path=...),
    load_portfolio(portfolio_path=...), CheckpointManager(checkpoint_dir=...)).
"""

from pathlib import Path
import os

def _env_path(name: str) -> Path | None:
    val = os.environ.get(name)
    return Path(val) if val else None

_ROOT = _env_path("TRADING_CRAB_ROOT")
_CONFIG_DIR = _env_path("TRADING_CRAB_CONFIG_DIR") or (_ROOT / "config" if _ROOT else None)
_DATA_DIR = _env_path("TRADING_CRAB_DATA_DIR") or (_ROOT / "data" if _ROOT else None)
_OUTPUT_DIR = _env_path("TRADING_CRAB_OUTPUT_DIR") or (_ROOT / "outputs" if _ROOT else None)

# Public path roots; None when not set (caller must set or pass explicit paths).
ROOT: Path | None = _ROOT
CONFIG_DIR: Path | None = _CONFIG_DIR
DATA_DIR: Path | None = _DATA_DIR
OUTPUT_DIR: Path | None = _OUTPUT_DIR
