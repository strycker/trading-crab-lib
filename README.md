## trading-crab-lib

Library of functions and helpers for the Trading-Crab project. **Functions-only:** no built-in paths to config or data; callers pass paths and parameters (or set environment variables).

### Design: caller-driven config and paths

- **No default config files.** The library does not open `config/settings.yaml` or any path by default. Callers pass a settings path and/or overrides:
  - `load(settings_path=Path("config/settings.yaml"))` or `load(settings_path=None, **overrides)`
  - `load_portfolio(portfolio_path=...)`, `load_email_config(config_path=...)`
- **Secrets from environment.** e.g. `FRED_API_KEY`; the library reads env and merges into config.
- **Path roots are optional.** Set them so that code that writes to “data” or “outputs” has a target:
  - **Environment:** `TRADING_CRAB_ROOT`, `TRADING_CRAB_DATA_DIR`, `TRADING_CRAB_OUTPUT_DIR`, `TRADING_CRAB_CONFIG_DIR`
  - **Or after import:** `import trading_crab_lib as crab; crab.ROOT = Path("/your/repo"); crab.DATA_DIR = crab.ROOT / "data"; ...`
- **CheckpointManager** requires either `checkpoint_dir=...` or `trading_crab_lib.DATA_DIR` (or `TRADING_CRAB_DATA_DIR`) to be set.

Pipelines and notebooks live in your own repo; they set paths (or env) and pass `settings_path` and other paths into the library.

### Relationship to `trading-crab` pipelines repo

This repo contains the reusable core logic for:
- data ingestion (`trading_crab_lib.ingestion.*`)
- feature engineering and transforms (`trading_crab_lib.transforms`)
- clustering / regimes / diagnostics
- prediction, reporting, email helpers, and plotting

Your orchestration, notebooks, and data live in a separate repo (e.g. [`trading-crab`](https://github.com/strycker/trading-crab)).

**Typical wiring from the pipelines/notebooks repo:**

1. **Install the library** (from a checkout or PyPI):

```bash
pip install -e ../trading-crab-lib   # or: pip install trading-crab-lib
```

2. **Set path roots and load config** (your repo owns the paths):

```python
from pathlib import Path
import trading_crab_lib as crab

# Optional: set roots so helpers that write to data/outputs know where to go
crab.ROOT = Path("/path/to/your/repo")
crab.CONFIG_DIR = crab.ROOT / "config"
crab.DATA_DIR = crab.ROOT / "data"
crab.OUTPUT_DIR = crab.ROOT / "outputs"

from trading_crab_lib.config import load
from trading_crab_lib.ingestion import fred, multpl, assets
from trading_crab_lib.transforms import engineer_all

# Caller provides settings path (or use overrides only)
cfg = load(settings_path=crab.CONFIG_DIR / "settings.yaml")
macro = fred.fetch_all(cfg)
features = engineer_all(macro, cfg, causal=True)
```

3. **Or use environment variables** so you don’t set attributes in code:

```bash
export TRADING_CRAB_ROOT=/path/to/your/repo
# CONFIG_DIR/DATA_DIR/OUTPUT_DIR default to $TRADING_CRAB_ROOT/{config,data,outputs}
```

### Testing the pipeline (this repo, after `pip install`)

**Quick check that the installed package sees your root:**

```bash
cd /path/to/trading-crab-lib
export TRADING_CRAB_ROOT="$(pwd)"
python -c "
import trading_crab_lib as crab
from trading_crab_lib.config import load
print('ROOT:', crab.ROOT)
print('CONFIG_DIR:', crab.CONFIG_DIR)
cfg = load(settings_path=crab.CONFIG_DIR / 'settings.yaml')
print('Config keys:', list(cfg.keys())[:5])
"
```

**Run the pipeline** (same shell with `TRADING_CRAB_ROOT` set):

```bash
export TRADING_CRAB_ROOT="$(pwd)"
python run_pipeline.py --steps 1,2,3
```

Or in one shot:

```bash
TRADING_CRAB_ROOT="$(pwd)" python run_pipeline.py --steps 1,2,3
```

Then `import trading_crab_lib as crab` will see `crab.CONFIG_DIR`, `crab.DATA_DIR`, and `crab.OUTPUT_DIR` derived from `TRADING_CRAB_ROOT`. No need to set `CONFIG_DIR`/`DATA_DIR`/`OUTPUT_DIR` unless you want to override them.

**Email:** There is no env var that replaces `config/email.local.yaml`. Keep that file (with your SMTP credentials) in your repo; it is **gitignored**. The runner (`run_pipeline.py`, `scripts/run_weekly_report.py`) looks for `config/email.local.yaml` or `config/email.yaml` under your root and passes that path to `load_email_config(path)`. So you don’t set an env var for email — you just keep the file and ensure `TRADING_CRAB_ROOT` (or your script’s path setup) points at the repo that contains `config/email.local.yaml`.

### Secrets and git (don’t upload credentials)

These are **ignored** so they are not committed or uploaded:

| File / pattern      | Purpose              | In `.gitignore` |
|---------------------|----------------------|-----------------|
| `.env` / `*.env`    | FRED_API_KEY, etc.   | Yes             |
| `config/email.local.yaml` | SMTP credentials | Yes             |
| `config/secrets.yaml`    | Other secrets   | Yes             |

**Check that secrets are ignored before `git add`:**

```bash
git check-ignore -v .env config/email.local.yaml
```

You should see both paths listed. If either is **not** ignored, fix `.gitignore` before committing. Never run `git add config/email.local.yaml` or `git add .env`; keep credentials only in `.env` and `config/email.local.yaml` and rely on the library reading them at runtime.

### Release smoke test (after install)

From a clean env (or with path env unset), the library should work with no config files:

```bash
pip install .
pytest tests/test_release_smoke.py -v
```

Then run the full test suite with path roots set (e.g. `TRADING_CRAB_ROOT` or repo `conftest`).
