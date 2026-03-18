"""
Release smoke test: library is importable and works without any config files.

Run after a clean install (e.g. pip install . or pip install dist/*.whl) to verify
PyPI-style usage: no dependence on repo layout or config files; callers pass paths/params.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_import_library():
    """Package imports without requiring config or data directories."""
    import trading_crab_lib as crab  # noqa: F401

    assert crab.ROOT is None or isinstance(crab.ROOT, Path)
    assert crab.CONFIG_DIR is None or isinstance(crab.CONFIG_DIR, Path)
    assert crab.DATA_DIR is None or isinstance(crab.DATA_DIR, Path)
    assert crab.OUTPUT_DIR is None or isinstance(crab.OUTPUT_DIR, Path)


def test_load_without_path_returns_dict():
    """load(settings_path=None) returns a dict with fred key; no file read."""
    from trading_crab_lib.config import load

    cfg = load()
    assert isinstance(cfg, dict)
    assert "fred" in cfg
    assert "api_key" in cfg["fred"]


def test_load_accepts_overrides():
    """load(settings_path=None, **overrides) merges overrides."""
    from trading_crab_lib.config import load

    cfg = load(assets={"etfs": ["SPY", "GLD"]})
    assert cfg.get("assets", {}).get("etfs") == ["SPY", "GLD"]


def test_load_portfolio_without_path_returns_empty():
    """load_portfolio(portfolio_path=None) returns {}."""
    from trading_crab_lib.config import load_portfolio

    assert load_portfolio() == {}


def test_load_email_config_without_path_returns_empty():
    """load_email_config(config_path=None) returns {}."""
    from trading_crab_lib.email import load_email_config

    assert load_email_config() == {}


def test_checkpoint_manager_requires_dir_or_data_dir():
    """CheckpointManager() raises when neither checkpoint_dir nor DATA_DIR is set."""
    from trading_crab_lib import DATA_DIR
    from trading_crab_lib.checkpoints import CheckpointManager

    if DATA_DIR is not None:
        pytest.skip("DATA_DIR set by env (e.g. TRADING_CRAB_ROOT); cannot test 'no path' behavior")
    with pytest.raises(ValueError, match="checkpoint_dir|DATA_DIR"):
        CheckpointManager()


def test_send_weekly_email_accepts_both_schemas():
    """send_weekly_email normalizes from_address/to_address to sender/recipients."""
    from trading_crab_lib.email import _normalize_email_config

    cfg = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "username": "u",
        "password": "p",
        "from_address": "from@example.com",
        "to_address": "to@example.com",
    }
    out = _normalize_email_config(cfg)
    assert out["sender"] == "from@example.com"
    assert out["recipients"] == ["to@example.com"]
