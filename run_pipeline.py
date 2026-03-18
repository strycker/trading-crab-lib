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
import shutil
import sys
from datetime import date
from pathlib import Path

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent / "src"))

import trading_crab_lib as crab
from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib.config import load, setup_logging
from trading_crab_lib.email import build_weekly_email_body, load_email_config, send_weekly_email
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
    p.add_argument(
        "--market-code",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Attach a market_code label column. "
            "Use 'grok' to load Grok labels (if present), or any other value "
            "to load checkpoint 'market_code_{NAME}'."
        ),
    )
    p.add_argument(
        "--save-market-code",
        action="store_true",
        help="After step 3, save balanced_cluster as checkpoint 'market_code_clustered'.",
    )
    p.add_argument(
        "--weekly-report",
        action="store_true",
        help="After running steps, archive outputs/reports/weekly_report.md to a dated copy and write email_body.txt.",
    )
    p.add_argument(
        "--send-email",
        action="store_true",
        help="After weekly-report post-processing, send the weekly report email using config/email.yaml.",
    )
    return p


def _load_market_code(source: str) -> "crab.pd.Series | None":  # type: ignore[name-defined]
    import pandas as pd

    cm = CheckpointManager()

    if source == "grok":
        try:
            from trading_crab_lib.ingestion.grok import load_grok_labels
        except Exception:
            return None
        mc = load_grok_labels(crab.DATA_DIR)
        if mc is None:
            return None
        cm.save(mc.to_frame(), "market_code_grok")
        return mc

    ckpt = f"market_code_{source}"
    try:
        df = cm.load(ckpt)
    except FileNotFoundError:
        return None
    s = df.iloc[:, 0]
    s.name = "market_code"
    return pd.to_numeric(s, errors="coerce")


def _save_market_code(labels, name: str) -> None:
    cm = CheckpointManager()
    cm.save(labels.rename("market_code").to_frame(), f"market_code_{name}")


def archive_weekly_report(reports_dir: Path | None = None) -> None:
    reports = reports_dir or (crab.OUTPUT_DIR / "reports")
    report_path = reports / "weekly_report.md"
    if not report_path.exists():
        print(f"No weekly_report.md at {report_path} — skip archive/email body.")
        return

    today = date.today().isoformat()
    stamped = reports / f"weekly_{today}.md"
    stamped.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, stamped)
    print(f"Archived report → {stamped}")

    email_body_path = reports / "email_body.txt"
    body = report_path.read_text(encoding="utf-8")
    email_body_path.write_text(body, encoding="utf-8")
    print(f"Email body → {email_body_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    setup_logging()
    run_cfg = RunConfig.from_args(args)
    run_cfg.market_code_source = getattr(args, "market_code", None)
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

        # Attach market_code after step 1 if requested (keeps pipelines simple)
        if step == 1 and args.market_code:
            import pandas as pd

            macro_path = crab.DATA_DIR / "raw" / "macro_raw.parquet"
            if macro_path.exists():
                macro = pd.read_parquet(macro_path)
                mc = _load_market_code(args.market_code)
                if mc is not None:
                    macro["market_code"] = mc.reindex(macro.index)
                    macro.to_parquet(macro_path)
                    CheckpointManager().save(macro, "macro_raw")
                    print(f"Attached market_code={args.market_code} to macro_raw")
                else:
                    print(f"WARNING: could not load market_code={args.market_code}; continuing without it")

        # Save balanced_cluster as market_code_clustered after step 3 if requested
        if step == 3 and args.save_market_code:
            import pandas as pd

            labels_path = crab.DATA_DIR / "regimes" / "cluster_labels.parquet"
            if labels_path.exists():
                labels = pd.read_parquet(labels_path)["balanced_cluster"].astype(int)
                _save_market_code(labels, "clustered")
                print("Saved market_code_clustered checkpoint")

        print("   ✓ done\n")

    if args.weekly_report or args.send_email:
        archive_weekly_report()

    if args.send_email:
        email_cfg = load_email_config()
        if not email_cfg:
            print("Email config not found or invalid; skipping send.")
        else:
            subject, body = build_weekly_email_body(crab.OUTPUT_DIR / "reports")
            ok = send_weekly_email(email_cfg, subject, body)
            print("Weekly report email sent." if ok else "Weekly report email failed to send (see logs).")

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

