"""
trading_crab_lib — Market Regime Classification & Prediction Pipeline utilities.
"""

from pathlib import Path

# When used within this repository, ROOT points at the repo root so that
# CONFIG_DIR, DATA_DIR and OUTPUT_DIR line up with the existing layout.
# When installed as a library, users can override these paths explicitly.
ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
