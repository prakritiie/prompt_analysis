"""
Stage 1 — Load, clean, and engineer base features.

Input : alpaca-gpt4 parquet (52,002 rows, 4 columns)
Output: dataframe with structural + readability features attached

Run:  python src/stage1_clean_enrich.py
"""

import argparse

import pandas as pd

from src.config import ENRICHED_PARQUET, RAW_PARQUET
from src.utils import (
    categorize_prompt,
    flesch,
    looks_like_code,
    normalise,
    readability_level,
    sentence_count,
    word_count,
)


def load_raw(path: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """
    Read the corpus.

    `path` may be the hf:// URI (needs `pip install fsspec huggingface_hub`)
    or a local .parquet you downloaded once. Local is faster and makes the
    pipeline reproducible offline, which is worth doing before a demo.
    """
    path = path or RAW_PARQUET
    try:
        df = pd.read_parquet(path)
    except ImportError as e:
        raise ImportError(
            f"Could not read {path}: {e}\n"
            "For hf:// URIs run:  pip install fsspec huggingface_hub\n"
            "Or download the parquet once and set RAW_PARQUET in config.py "
            "to a local path."
        ) from e

    if limit:
        df = df.head(limit).copy()
    print(f"[load] {len(df):,} rows, columns: {list(df.columns)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning happens once, up front, before anything derives from these columns.

    'text' is dropped: it is just instruction+input+output concatenated for
    training, so keeping it triples the storage for zero analytical value.
    """
    df = df.copy()

    if "text" in df.columns:
        df = df.drop(columns=["text"])

    df = df.rename(columns={"input": "input_text"})

    for col in ("instruction", "input_text", "output"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    before = len(df)
    df = df[df["instruction"] != ""]
    df = df[df["output"] != ""]
    dropped = before - len(df)
    if dropped:
        print(f"[clean] dropped {dropped:,} rows with empty instruction/output")

    exact_dupes = df.duplicated(subset=["instruction", "input_text"]).sum()
    print(f"[clean] exact duplicate instruction+input pairs: {exact_dupes:,}")
    df["is_exact_duplicate"] = df.duplicated(
        subset=["instruction", "input_text"], keep="first"
    ).astype(int)

    df = df.reset_index(drop=True)
    df.insert(0, "row_id", df.index + 1)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Attach structural, readability, and baseline-category features."""
    df = df.copy()

    df["instruction_clean"] = df["instruction"].map(normalise)
    df["has_input"] = (df["input_text"].str.len() > 0).astype(int)

    df["instruction_word_count"] = df["instruction"].map(word_count)
    df["output_word_count"] = df["output"].map(word_count)
    df["sentence_count"] = df["output"].map(sentence_count)
    df["words_per_sentence"] = (
        df["output_word_count"] / df["sentence_count"]
    ).round(2)

    df["is_code_like"] = df["output"].map(looks_like_code).astype(int)

    # Only score prose. Code blocks return large negative Flesch values that
    # would otherwise dominate the distribution and every downstream cut.
    df["flesch_score"] = pd.NA
    prose = df["is_code_like"] == 0
    df.loc[prose, "flesch_score"] = df.loc[prose, "output"].map(flesch)
    df["flesch_score"] = pd.to_numeric(df["flesch_score"], errors="coerce").round(2)
    df["readability_level"] = df["flesch_score"].map(readability_level)

    df["prompt_type_rule"] = df["instruction"].map(categorize_prompt)

    return df


def report_baseline(df: pd.DataFrame) -> None:
    """
    The headline number that justifies stage 2.

    Print the FULL distribution including 'Other'. Slicing it off is how you
    end up unable to explain why you clustered at all.
    """
    counts = df["prompt_type_rule"].value_counts()
    pct = (counts / len(df) * 100).round(1)

    print("\n[baseline] rule-based classifier coverage")
    print("-" * 46)
    for label, n in counts.items():
        print(f"  {label:<20} {n:>7,}  ({pct[label]:>5.1f}%)")
    print("-" * 46)
    other_pct = pct.get("Other", 0.0)
    print(f"  UNCATEGORISED ('Other'): {other_pct:.1f}% of corpus")
    print("  -> this is the motivation for embedding-based clustering\n")


def main(limit: int | None = None) -> pd.DataFrame:
    df = load_raw(limit=limit)
    df = clean(df)
    df = engineer(df)
    report_baseline(df)

    df.to_parquet(ENRICHED_PARQUET, index=False)
    print(f"[save] {ENRICHED_PARQUET}  ({len(df):,} rows, {len(df.columns)} cols)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="row cap for a fast smoke test")
    main(**vars(ap.parse_args()))
