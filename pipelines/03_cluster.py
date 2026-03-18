"""
Pipeline step 3 — Unsupervised Clustering

Reads features.parquet, runs:
  1. PCA (fixed n_components from config)
  2. KMeans k-sweep (silhouette / CH / DB scoring)
  3. Standard KMeans at best_k
  4. Size-constrained KMeans at balanced_k

Writes:
  data/regimes/cluster_labels.parquet   — quarter → cluster, balanced_cluster
  data/regimes/pca_components.parquet   — quarter → PC1…PCn
  data/regimes/kmeans_scores.parquet    — k-sweep evaluation table

Run:
    python pipelines/03_cluster.py
"""

import sys
from pathlib import Path

# Prefer the installed package; fall back to ./src for local runs.
try:
    import trading_crab_lib as crab  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_crab_lib.config import load, setup_logging
from trading_crab_lib.clustering import (
    reduce_pca,
    evaluate_kmeans,
    pick_best_k,
    fit_clusters,
)

import pandas as pd
import numpy as np


def main() -> None:
    setup_logging()
    cfg = load()
    clust_cfg = cfg.get("clustering") or {}
    # Sensible defaults so the runner works with minimal configs.
    n_pca = int(clust_cfg.get("n_pca_components", 5))
    n_search = int(clust_cfg.get("n_clusters_search", 8))
    k_cap = int(clust_cfg.get("k_cap", n_search))
    balanced_k = int(clust_cfg.get("balanced_k", 5))
    random_state = int(clust_cfg.get("random_state", 42))

    features = pd.read_parquet(crab.DATA_DIR / "processed" / "features.parquet")
    X = features.drop(columns=["market_code"], errors="ignore")
    print(f"\nLoaded features: {X.shape}")

    # PCA cannot handle NaNs. Centered derivative features commonly produce NaNs
    # at the beginning/end of the series (and sometimes for sparse inputs).
    # Keep the pipeline resilient by dropping affected rows before PCA.
    X = X.replace([np.inf, -np.inf], np.nan)
    n_before = len(X)
    X = X.dropna(axis=0, how="any")
    n_dropped = n_before - len(X)
    if n_dropped:
        print(f"Dropped {n_dropped} rows with NaN/inf before PCA; remaining: {X.shape}")
    if X.empty:
        raise ValueError(
            "All feature rows contain NaN/inf after preprocessing; cannot run PCA. "
            "Inspect data/processed/features.parquet and your feature engineering config."
        )

    # ── 1. PCA ─────────────────────────────────────────────────────────────
    # Library logs: "Running PCA... done." + variance ratios
    pca_df, pca_model, scaler = reduce_pca(
        X,
        n_components=n_pca,
        random_state=random_state,
    )

    # ── 2. Evaluate k values ────────────────────────────────────────────────
    # Library logs: "Evaluating cluster counts... done." + full sorted table
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(pca_df.values)
    scores = evaluate_kmeans(
        X_scaled,
        k_range=range(2, n_search + 1),
        random_state=random_state,
    )
    best_k = pick_best_k(scores, k_cap=k_cap)
    print(f"\nChosen k={best_k}  (silhouette winner capped at k_cap={k_cap})")

    # ── 3 & 4. Fit both clusterings ─────────────────────────────────────────
    # Library logs: "N quarters clustered into K regimes (balanced into K)."
    clustered = fit_clusters(
        pca_df,
        best_k=best_k,
        balanced_k=balanced_k,
        random_state=random_state,
    )

    # Restore market_code for downstream steps
    if "market_code" in features.columns:
        clustered["market_code"] = features.loc[clustered.index, "market_code"]

    # ── Persist ─────────────────────────────────────────────────────────────
    out_dir = crab.DATA_DIR / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_cols = ["cluster", "balanced_cluster"] + (
        ["market_code"] if "market_code" in clustered.columns else []
    )
    clustered[label_cols].to_parquet(out_dir / "cluster_labels.parquet")
    clustered.drop(columns=label_cols, errors="ignore").to_parquet(out_dir / "pca_components.parquet")
    scores.to_parquet(out_dir / "kmeans_scores.parquet", index=False)

    print(f"\nStandard clusters (k={best_k}):")
    print(clustered["cluster"].value_counts().sort_index().to_string())

    print(f"\nBalanced clusters (k={balanced_k}):")
    print(clustered["balanced_cluster"].value_counts().sort_index().to_string())

    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
