"""
Pipeline step 7 — Stoplight Dashboard + Portfolio Recommendations

Loads all previously computed artifacts and prints a concise summary:
  - Current predicted regime
  - Asset stoplight signals (GREEN / YELLOW / RED)
  - Forward transition probabilities
  - Portfolio weights (simple + blended)
  - BUY / SELL / HOLD trade recommendations vs all-cash baseline

Features are read from features_supervised.parquet (causal/backward rolling
windows — consistent with how the model was trained in step 5).

Saves to outputs/reports/:
  dashboard.csv              — asset signals
  portfolio_simple.csv       — equal-weight top-3 assets for current regime
  portfolio_blended.csv      — probability-weighted allocation across all regimes
  trade_recommendations.csv  — BUY/SELL/HOLD signals vs all-cash

Run:
    python pipelines/07_dashboard.py
"""

import pickle
from pathlib import Path

# Prefer the installed package; fall back to ./src for local runs.
try:
    import trading_crab_lib as crab  # noqa: F401
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import trading_crab_lib as crab  # type: ignore[no-redef]
from trading_crab_lib.config import load, setup_logging
from trading_crab_lib.config import load_portfolio
from trading_crab_lib.prediction import predict_current
from trading_crab_lib.asset_returns import rank_assets_by_regime
from trading_crab_lib.reporting import (
    asset_signals,
    print_dashboard,
    save_dashboard_csv,
    simple_regime_portfolio,
    blended_regime_portfolio,
    generate_recommendation,
    build_recommendation_digest,
    save_recommendation_bundle,
    write_weekly_report_md,
)

import pandas as pd
import yaml


def load_regime_names() -> dict[int, str]:
    # Prefer manually edited config/regime_labels.yaml, fall back to auto-suggestions
    override_path = (crab.CONFIG_DIR / "regime_labels.yaml") if crab.CONFIG_DIR else None
    suggested_path = (crab.DATA_DIR / "regimes" / "regime_names_suggested.yaml") if crab.DATA_DIR else None

    for path in [override_path, suggested_path]:
        if path is not None and path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            names = {int(k): v for k, v in raw.items() if not str(k).startswith("#")}
            if names:
                return names
    return {}


def main() -> None:
    setup_logging()
    cfg = load(settings_path=(crab.CONFIG_DIR / "settings.yaml") if crab.CONFIG_DIR else None)

    # Load current-regime model
    model_dir = crab.OUTPUT_DIR / "models"
    with open(model_dir / "current_regime.pkl", "rb") as f:
        current_model = pickle.load(f)

    # Use causal features for live scoring — same as training in step 5
    sup_path = crab.DATA_DIR / "processed" / "features_supervised.parquet"
    feat_path = sup_path if sup_path.exists() else crab.DATA_DIR / "processed" / "features.parquet"
    if not sup_path.exists():
        print(
            "WARNING: features_supervised.parquet not found — falling back to features.parquet.\n"
            "Re-run step 2 to generate causal features."
        )
    features = pd.read_parquet(feat_path)
    X = features.drop(columns=["market_code"], errors="ignore")
    if hasattr(current_model, "feature_names_in_"):
        X = X[current_model.feature_names_in_]
    else:
        X = X.dropna(axis=1, how="any")
    prediction = predict_current(current_model, X)

    # Load supporting data
    tm = pd.read_parquet(crab.DATA_DIR / "regimes" / "transition_matrix.parquet")
    regime_names = load_regime_names()
    thresholds = cfg.get("dashboard", {}).get("signal_thresholds", None)

    # ── Asset signals ──────────────────────────────────────────────────────
    asset_signals_df = pd.DataFrame()
    profile_path = crab.DATA_DIR / "regimes" / "asset_return_profile.parquet"
    profile: pd.DataFrame | None = None
    if profile_path.exists():
        profile = pd.read_parquet(profile_path)
        ranked = rank_assets_by_regime(profile)
        asset_signals_df = asset_signals(ranked, prediction["regime"], thresholds=thresholds)

    print_dashboard(prediction, regime_names, asset_signals_df, tm)

    report_dir = crab.OUTPUT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not asset_signals_df.empty:
        save_dashboard_csv(asset_signals_df, report_dir)

    # ── Portfolio construction ─────────────────────────────────────────────
    if profile is not None and not profile.empty:
        current_regime = prediction["regime"]
        probs = prediction["probabilities"]

        simple_weights = simple_regime_portfolio(profile, current_regime, top_n=3)
        blended_weights = blended_regime_portfolio(profile, probs, top_n=3)
        # Use current holdings (if configured) to produce BUY/SELL/HOLD deltas.
        portfolio_path = (crab.CONFIG_DIR / "portfolio.yaml") if crab.CONFIG_DIR else None
        current_weights = pd.Series(load_portfolio(portfolio_path), dtype=float)
        recommendations = generate_recommendation(blended_weights, current_weights=current_weights)

        print("\n── Simple portfolio (top-3 for current regime) ──")
        for asset, w in simple_weights.items():
            print(f"  {asset:<12s}  {w:.1%}")

        print("\n── Blended portfolio (probability-weighted) ──")
        for asset, w in blended_weights.items():
            print(f"  {asset:<12s}  {w:.1%}")

        print("\n── Trade recommendations (blended vs all-cash) ──")
        print(recommendations.to_string())

        if not simple_weights.empty:
            simple_weights.to_frame("weight").to_csv(report_dir / "portfolio_simple.csv")
        if not blended_weights.empty:
            blended_weights.to_frame("weight").to_csv(report_dir / "portfolio_blended.csv")
        if not recommendations.empty:
            recommendations.to_csv(report_dir / "trade_recommendations.csv")
            print(f"\nReports saved to {report_dir}")

        # ── Weekly report bundle ───────────────────────────────────────────
        # This is the single markdown artifact used by the email sender.
        regime_name = regime_names.get(int(current_regime), f"Regime {current_regime}")
        transition_row = tm.loc[current_regime] if current_regime in tm.index else None

        # build_recommendation_digest() expects a behavior table with a `regime` column
        # (one row per asset per regime). Use ranked returns as the base and attach
        # stoplight signal info for the current regime where available.
        behavior_df = ranked.rename(columns={"median_quarterly_return": "median_return"}).copy()
        # Add placeholders expected by the reporting helpers.
        if "signal_display" not in behavior_df.columns:
            behavior_df["signal_display"] = None
        if "score_relative" not in behavior_df.columns:
            behavior_df["score_relative"] = None
        if "score_absolute" not in behavior_df.columns:
            behavior_df["score_absolute"] = None

        # Populate signal_display for current regime from stoplight signals.
        if not asset_signals_df.empty:
            sig_map = {"GREEN": "green_strong", "YELLOW": "yellow", "RED": "red"}
            current_sig = asset_signals_df.copy()
            current_sig["signal_display"] = current_sig["signal"].map(sig_map).fillna(current_sig["signal"])
            behavior_df = behavior_df.merge(
                current_sig[["asset", "signal_display"]],
                on="asset",
                how="left",
                suffixes=("", "_cur"),
            )
            # Prefer current-regime signal_display when present.
            if "signal_display_cur" in behavior_df.columns:
                behavior_df["signal_display"] = behavior_df["signal_display_cur"].combine_first(behavior_df["signal_display"])
                behavior_df = behavior_df.drop(columns=["signal_display_cur"])

        digest_df = build_recommendation_digest(
            behavior_df=behavior_df,
            current_regime=int(current_regime),
            current_weights=current_weights,
            target_weights=blended_weights,
            rec_df=recommendations,
        )
        save_recommendation_bundle(
            digest_df=digest_df,
            current_regime=int(current_regime),
            regime_name=regime_name,
            regime_probabilities={int(k): float(v) for k, v in probs.items()},
            output_path=report_dir / "recommendation_bundle.parquet",
        )

        write_weekly_report_md(
            current_regime=int(current_regime),
            regime_name=regime_name,
            regime_probabilities={int(k): float(v) for k, v in probs.items()},
            rec_df=recommendations,
            transition_row=transition_row,
            output_path=report_dir / "weekly_report.md",
        )


if __name__ == "__main__":
    main()
