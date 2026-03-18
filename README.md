## trading-crab-lib

Library of classes and support functions for the Trading-Crab project.

### Relationship to `trading-crab` pipelines repo

This repo (`trading-crab-lib`) contains the reusable, testable core logic for:
- data ingestion (`trading_crab_lib.ingestion.*`)
- feature engineering and transforms (`trading_crab_lib.transforms`)
- clustering / regimes / diagnostics
- prediction, reporting, email helpers, and plotting

Your orchestration code, notebooks, and data live in a separate repo:
- Pipelines / notebooks repo: [`trading-crab`](https://github.com/strycker/trading-crab)

Typical wiring from the `trading-crab` repo:

1. **Install the library** (from a checkout or PyPI once published):

```bash
pip install -e ../trading-crab-lib  # or: pip install trading-crab-lib
```

2. **Import and use from pipelines / notebooks** in `trading-crab`:

```python
import trading_crab_lib as crab

from trading_crab_lib.config import load
from trading_crab_lib.ingestion import fred, multpl, assets
from trading_crab_lib.transforms import engineer_all
from trading_crab_lib.runtime import RunConfig

cfg = load()
run_cfg = RunConfig(generate_plots=True, verbose=True)
macro = fred.fetch_all(cfg)
features = engineer_all(macro, cfg, causal=True)
```

3. **Share filesystem layout** between the two repos:
- This repo expects `config/`, `data/`, and `outputs/` to live at the repo root.
- In `trading-crab`, point your scripts or notebooks at the same folders (or
  override paths in your own glue code before calling into `trading_crab_lib`).
