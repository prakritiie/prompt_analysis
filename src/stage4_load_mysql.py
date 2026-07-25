"""
Stage 4 — Load the enriched table into MySQL.

The schema is created by sql/01_schema.sql, NOT inferred by pandas. Letting
to_sql() guess gives you TEXT columns for integers and a utf8mb3 default that
mangles the unicode in GPT-4 outputs.

Prereqs:
    pip install sqlalchemy pymysql
    mysql -u root -p < sql/01_schema.sql

Run:  python src/stage4_load_mysql.py
"""

import argparse

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import (
    DATABASE_URL, ENRICHED_PARQUET, LOAD_CHUNKSIZE, MYSQL, MYSQL_TABLE,
)

# Only these columns go to MySQL. The raw 'output' text is included because
# the dashboard needs previews, but 'instruction_clean' (a TF-IDF artefact)
# is not — it has no analytical meaning downstream.
DB_COLUMNS = [
    "row_id", "instruction", "input_text", "output",
    "has_input", "prompt_type_rule", "cluster_id", "cluster_label",
    "instruction_word_count", "output_word_count", "sentence_count",
    "words_per_sentence", "flesch_score", "readability_level", "is_code_like",
    "is_exact_duplicate", "is_near_duplicate", "duplicate_of",
    "flag_truncated", "flag_too_short", "flag_empty_ish",
    "flag_repetitive", "flag_readability_outlier",
    "n_flags", "adequacy_score",
]


def build_engine():
    """
    Engine is built here, not at config import time, so stages 1-3 do not
    require a running database.
    """
    if not MYSQL["password"]:
        raise RuntimeError(
            "MYSQL_PASSWORD is not set.\n"
            '  PowerShell:  $env:MYSQL_PASSWORD = "your_password"\n'
            "  cmd.exe:     set MYSQL_PASSWORD=your_password"
        )
    print(f"[mysql] {DATABASE_URL.render_as_string(hide_password=True)}")
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def check_version(engine) -> None:
    """Window functions need MySQL 8.0+. Fail loudly now, not in sql/02_views.sql."""
    with engine.connect() as conn:
        version = conn.execute(text("SELECT VERSION()")).scalar()
    print(f"[mysql] server version {version}")
    major = int(str(version).split(".")[0])
    if major < 8:
        raise RuntimeError(
            f"MySQL {version} does not support window functions. "
            "The views in sql/02_views.sql require 8.0+."
        )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Align to the DB schema and coerce nullable types MySQL won't accept."""
    missing = [c for c in DB_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns: {missing}. Run stages 1-3 before loading."
        )

    out = df[DB_COLUMNS].copy()

    # pandas nullable NA -> None so pymysql writes SQL NULL
    out["duplicate_of"] = out["duplicate_of"].astype("object").where(
        out["duplicate_of"].notna(), None
    )
    out["flesch_score"] = pd.to_numeric(out["flesch_score"], errors="coerce")
    out = out.astype({"row_id": int, "cluster_id": int, "adequacy_score": int})
    return out


def main(truncate: bool = True) -> None:
    df = pd.read_parquet(ENRICHED_PARQUET)
    print(f"[load] {len(df):,} rows from {ENRICHED_PARQUET.name}")

    out = prepare(df)
    engine = build_engine()
    check_version(engine)

    if truncate:
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {MYSQL_TABLE}"))
        print(f"[mysql] truncated {MYSQL_TABLE}")

    out.to_sql(
        MYSQL_TABLE,
        engine,
        if_exists="append",   # append into the hand-made schema, never replace
        index=False,
        chunksize=LOAD_CHUNKSIZE,
        method="multi",
    )

    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {MYSQL_TABLE}")).scalar()
    print(f"[mysql] {n:,} rows now in {MYSQL['database']}.{MYSQL_TABLE}")
    print("[next] run: mysql -u root -p corpus_audit < sql/02_views.sql")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-truncate", dest="truncate", action="store_false")
    main(**vars(ap.parse_args()))