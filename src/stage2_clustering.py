"""
Stage 2 — Task-type clustering via sentence embeddings + K-Means.

Why this exists: the rule-based classifier in stage 1 matches on leading verb
only, so any indirectly-phrased instruction falls into 'Other'. Clustering
assigns 100% of rows a task type and surfaces categories the rules never had.

Strategy for 52K rows without a GPU:
  1. embed a sample (default 15K)
  2. sweep k, pick by silhouette score
  3. fit K-Means on the sample
  4. assign the remaining rows to the nearest centroid

Run:  python src/stage2_clustering.py
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_SAMPLE_SIZE,
    ENRICHED_PARQUET,
    K_RANGE,
    OUTPUT_DIR,
    RANDOM_SEED,
    SILHOUETTE_SAMPLE,
)


def embed(texts: list[str]) -> np.ndarray:
    """Encode instructions into dense vectors."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine == dot product after this
    )


def choose_k(X: np.ndarray, k_range=K_RANGE, seed: int = RANDOM_SEED) -> tuple[int, dict]:
    """
    Sweep k and pick the best silhouette score.

    Silhouette is computed on a subsample because it is O(n^2) in memory.
    Scores are saved so the choice of k is auditable, not asserted.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)

    scores: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = float(silhouette_score(X[idx], labels[idx], metric="cosine"))
        print(f"  k={k:<3} silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"[choose_k] best k = {best_k} (silhouette {scores[best_k]:.4f})")
    return best_k, scores


def label_clusters(
    df: pd.DataFrame, X: np.ndarray, km: KMeans, sample_idx: np.ndarray, top_n: int = 10
) -> dict[int, list[str]]:
    """
    Pull the instructions closest to each centroid.

    You read these and hand-write the cluster names. Do not skip this: an
    unnamed cluster is a number, a named cluster is a finding.
    """
    exemplars: dict[int, list[str]] = {}
    for cid in range(km.n_clusters):
        member_mask = km.labels_ == cid
        if member_mask.sum() == 0:
            exemplars[cid] = []
            continue
        members = X[member_mask]
        dists = np.linalg.norm(members - km.cluster_centers_[cid], axis=1)
        closest = np.argsort(dists)[:top_n]
        rows = df.iloc[sample_idx[member_mask][closest]]
        exemplars[cid] = rows["instruction"].str.slice(0, 110).tolist()
    return exemplars


def main(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(ENRICHED_PARQUET)
    if limit:
        df = df.head(limit).copy()
    print(f"[load] {len(df):,} rows")

    rng = np.random.default_rng(RANDOM_SEED)
    n_sample = min(EMBEDDING_SAMPLE_SIZE, len(df))
    sample_idx = rng.choice(len(df), size=n_sample, replace=False)

    print(f"[embed] encoding {n_sample:,} instructions with {EMBEDDING_MODEL}")
    X_sample = embed(df.iloc[sample_idx]["instruction"].tolist())

    print("[choose_k] sweeping candidate cluster counts")
    best_k, scores = choose_k(X_sample)

    km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    km.fit(X_sample)

    exemplars = label_clusters(df, X_sample, km, sample_idx)
    print("\n[exemplars] name each cluster from these, then edit CLUSTER_NAMES below")
    for cid, items in exemplars.items():
        print(f"\n  --- cluster {cid} ({(km.labels_ == cid).sum():,} of sample) ---")
        for it in items[:5]:
            print(f"     - {it}")

    # Assign every row: sampled rows keep their fitted label, the rest are
    # embedded in batches and mapped to the nearest centroid.
    df["cluster_id"] = -1
    df.iloc[sample_idx, df.columns.get_loc("cluster_id")] = km.labels_

    remaining = df.index[df["cluster_id"] == -1].to_numpy()
    if len(remaining):
        print(f"\n[assign] embedding remaining {len(remaining):,} rows")
        X_rest = embed(df.loc[remaining, "instruction"].tolist())
        df.loc[remaining, "cluster_id"] = km.predict(X_rest)

    df["cluster_id"] = df["cluster_id"].astype(int)

    # Placeholder names — replace with your own after reading the exemplars.
    CLUSTER_NAMES = {cid: f"cluster_{cid}" for cid in range(best_k)}
    df["cluster_label"] = df["cluster_id"].map(CLUSTER_NAMES)

    with open(OUTPUT_DIR / "cluster_selection.json", "w") as f:
        json.dump(
            {
                "best_k": best_k,
                "silhouette_by_k": scores,
                "exemplars": {str(k): v for k, v in exemplars.items()},
                "sizes": df["cluster_id"].value_counts().sort_index().to_dict(),
            },
            f,
            indent=2,
        )

    df.to_parquet(ENRICHED_PARQUET, index=False)
    print(f"\n[save] cluster_id + cluster_label written to {ENRICHED_PARQUET}")
    print("[next] edit CLUSTER_NAMES with real names, re-run the mapping line")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(**vars(ap.parse_args()))
