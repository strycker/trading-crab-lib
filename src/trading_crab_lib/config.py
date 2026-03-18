"""
Central config loader — caller-driven; no default file paths.

Calling code passes settings_path (from their repo) and/or overrides.
Secrets (e.g. FRED_API_KEY) are read from the environment.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)


def load(settings_path: Path | None = None, **overrides) -> dict:
    """
    Load settings from an optional YAML file and merge with overrides and env.

    Args:
        settings_path: Path to settings YAML. If None, no file is read; config
            is built only from overrides and environment (e.g. FRED_API_KEY).
        **overrides: Keys to merge into the config (override file and env).

    Returns:
        Config dict. Always includes at least fred.api_key from FRED_API_KEY
        when no file is given or when merging.
    """
    load_dotenv()

    if settings_path is not None and settings_path.exists():
        with open(settings_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    cfg.setdefault("fred", {})
    if "api_key" not in cfg["fred"] or cfg["fred"]["api_key"] is None:
        cfg["fred"]["api_key"] = os.getenv("FRED_API_KEY")
    if not cfg["fred"].get("api_key"):
        log.warning("FRED_API_KEY not set — FRED ingestion will fail")

    for key, value in overrides.items():
        if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value

    return cfg


def load_portfolio(portfolio_path: Path | None = None) -> dict[str, float]:
    """
    Load current portfolio weights from a YAML file (ticker -> weight fraction).
    Weights are normalized to sum to 1.

    Args:
        portfolio_path: Path to portfolio YAML. If None, returns {} (no file read).

    Returns:
        Dict of ticker -> weight fraction, or {} if path is None/missing/empty.
    """
    if portfolio_path is None or not portfolio_path.exists():
        log.debug("No portfolio file at %s", portfolio_path)
        return {}
    with open(portfolio_path) as f:
        raw = yaml.safe_load(f)
    if not raw or not isinstance(raw, dict):
        return {}
    weights: dict[str, float] = {}
    for k, v in raw.items():
        if str(k).startswith("#"):
            continue
        try:
            w = float(v)
            if w > 0:
                weights[str(k).strip()] = w
        except (TypeError, ValueError):
            continue
    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {t: w / total for t, w in weights.items()}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
    )
