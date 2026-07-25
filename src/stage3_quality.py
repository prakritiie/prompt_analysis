"""
Stage 3 — Near-duplicate detection and response quality flags.

Two independent questions:
  1. Redundancy  — how much of this corpus is saying the same thing twice?
  2. Adequacy    — how many responses are truncated, stunted, or degenerate?

Both produce row-level flags that roll up into the corpus recommendation.

Run:  python src/stage3_quality.py
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.config import (
    AUDIT_REPORT,
    DUPLICATE_THRESHOLD,
    EMPTY_OUTPUT_WORDS,
    ENRICHED_PARQUET,
    FLAG_PENALTY,
    LOW_ADEQUACY_CUTOFF,
    MIN_OUTPUT_WORDS,
    REPEAT_NGRAM_MIN_COUNT,
    REVIEW_SAMPLE_CSV,
    SAMPLE_CSV,
    TFIDF_MAX_FEATURES,
)
from src.utils import has_repeated_ngram, is_truncated

QUALITY_FLAGS = [
    "flag_truncated",
    "flag_too_short",
    "flag_empty_ish",
    "flag_repetitive",
    "flag_readability_outlier",
]


# ------------------------------------------------------------------ duplicates
def flag_near_duplicates(
    df: pd.DataFrame, threshold: float = DUPLICATE_THRESHOLD
) -> pd.DataFrame:
    """
    Flag instructions that are near-verbatim restatements of an earlier row.

    Comparison runs WITHIN each cluster. Two reasons: near-duplicates are
    semantically similar so they land in the same cluster anyway, and it turns
    one 52K x 52K problem into a dozen small ones that fit in memory.

    Only the later row of a pair is flagged, so 'drop the flagged rows' always
    leaves one copy behind.
    """
    df = df.copy()
    df["is_near_duplicate"] = 0
    df["duplicate_of"] = pd.NA

    group_col = "cluster_id" if "cluster_id" in df.columns else "prompt_type_rule"

    for cid, block in df.groupby(group_col):
        if len(block) < 2:
            continue

        vec = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        )
        try:
            X = vec.fit_transform(block["instruction_clean"])
        except ValueError:
            continue  # vocabulary empty for a tiny block
        if X.shape[1] == 0:
            continue

        nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X)
        dist, idx = nn.kneighbors(X)

        # column 0 is the row itself (distance 0); column 1 is its nearest other
        sim = 1 - dist[:, 1]
        neighbour_pos = idx[:, 1]

        positions = block.index.to_numpy()
        for i, (s, j) in enumerate(zip(sim, neighbour_pos)):
            if s < threshold:
                continue
            row_a, row_b = positions[i], positions[j]
            # keep the earlier row, flag the later one
            if row_a > row_b:
                df.at[row_a, "is_near_duplicate"] = 1
                df.at[row_a, "duplicate_of"] = int(df.at[row_b, "row_id"])

        print(f"  cluster {cid}: {len(block):>6,} rows, "
              f"{int(df.loc[positions, 'is_near_duplicate'].sum()):>5,} flagged")

    return df


# ------------------------------------------------------------------ quality
def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Five boolean flags rolled into a 0-100 adequacy score."""
    df = df.copy()

    df["flag_truncated"] = df["output"].map(is_truncated).astype(int)
    df["flag_too_short"] = (df["output_word_count"] < MIN_OUTPUT_WORDS).astype(int)
    df["flag_empty_ish"] = (df["output_word_count"] < EMPTY_OUTPUT_WORDS).astype(int)
    df["flag_repetitive"] = (
        df["output"]
        .map(lambda t: has_repeated_ngram(t, min_count=REPEAT_NGRAM_MIN_COUNT))
        .astype(int)
    )

    # Readability outliers are judged against the row's OWN cluster, because
    # a code-explanation cluster legitimately reads harder than a poetry one.
    # Code-like outputs are excluded: their Flesch scores are not meaningful.
    df["flag_readability_outlier"] = 0
    group_col = "cluster_id" if "cluster_id" in df.columns else "prompt_type_rule"
    prose = (df["is_code_like"] == 0) & df["flesch_score"].notna()

    for _, block in df[prose].groupby(group_col):
        q1, q3 = block["flesch_score"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = block.index[
            (block["flesch_score"] < lo) | (block["flesch_score"] > hi)
        ]
        df.loc[outliers, "flag_readability_outlier"] = 1

    df["n_flags"] = df[QUALITY_FLAGS].sum(axis=1)
    df["adequacy_score"] = (100 - df["n_flags"] * FLAG_PENALTY).clip(lower=0)

    return df


# ------------------------------------------------------------------ reporting
def build_report(df: pd.DataFrame) -> dict:
    n = len(df)

    # A row can be flagged both exact and near duplicate. Union, don't sum,
    # or the reported rate exceeds 100%.
    any_dupe = (df["is_near_duplicate"] == 1) | (df["is_exact_duplicate"] == 1)
    low_adq_mask = df["adequacy_score"] <= LOW_ADEQUACY_CUTOFF

    dupes = int(any_dupe.sum())
    low_adq = int(low_adq_mask.sum())
    droppable = int((any_dupe | low_adq_mask).sum())

    return {
        "total_rows": n,
        "exact_duplicates": int(df["is_exact_duplicate"].sum()),
        "near_duplicates": int(df["is_near_duplicate"].sum()),
        "duplicate_rows_union": dupes,
        "duplicate_pct": round(100 * dupes / n, 2),
        "low_adequacy_rows": low_adq,
        "low_adequacy_pct": round(100 * low_adq / n, 2),
        "flag_counts": {f: int(df[f].sum()) for f in QUALITY_FLAGS},
        "mean_adequacy": round(float(df["adequacy_score"].mean()), 2),
        "recommended_drop_rows": droppable,
        "recommended_drop_pct": round(100 * droppable / n, 2),
        "rows_remaining_after_clean": n - droppable,
        "cluster_sizes": (
            df["cluster_label"].value_counts().to_dict()
            if "cluster_label" in df.columns
            else {}
        ),
    }


def main(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(ENRICHED_PARQUET)
    if limit:
        df = df.head(limit).copy()
    print(f"[load] {len(df):,} rows")

    print("\n[duplicates] scanning within clusters")
    df = flag_near_duplicates(df)

    print("\n[quality] applying flags")
    df = flag_quality(df)

    report = build_report(df)
    print("\n" + "=" * 52)
    print("AUDIT SUMMARY")
    print("=" * 52)
    print(json.dumps(report, indent=2))

    with open(AUDIT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    # Manual validation set. Read these 20 yourself and record in the README
    # how many were genuinely bad. An unvalidated score is an assertion.
    review = (
        df[df["adequacy_score"] <= LOW_ADEQUACY_CUTOFF]
        .sample(min(20, int((df["adequacy_score"] <= LOW_ADEQUACY_CUTOFF).sum())),
                random_state=42)
        [["row_id", "instruction", "output", "n_flags", "adequacy_score"] + QUALITY_FLAGS]
    )
    review.to_csv(REVIEW_SAMPLE_CSV, index=False)
    print(f"\n[save] manual review set -> {REVIEW_SAMPLE_CSV}")

    df.to_parquet(ENRICHED_PARQUET, index=False)
    df.head(500).to_csv(SAMPLE_CSV, index=False)
    print(f"[save] full table -> {ENRICHED_PARQUET}")
    print(f"[save] 500-row sample -> {SAMPLE_CSV}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(**vars(ap.parse_args()))
