"""
Run the full audit pipeline end to end.

    python src/run_all.py              # full 52K corpus
    python src/run_all.py --limit 2000 # fast smoke test
    python src/run_all.py --skip-mysql # stop before the database load

Stage 2 needs sentence-transformers. If it is not installed the pipeline
continues with the rule-based categories as the grouping key, so you can
still get a working audit before setting up embeddings.
"""

import argparse
import sys
import time

import src.stage1_clean_enrich as stage1_clean_enrich
import src.stage3_quality as stage3_quality


def banner(msg: str) -> None:
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60)


def main(limit: int | None, skip_mysql: bool, skip_clustering: bool) -> None:
    t0 = time.time()

    banner("STAGE 1 — clean & enrich")
    stage1_clean_enrich.main(limit=limit)

    if skip_clustering:
        print("\n[skip] stage 2 clustering disabled by flag")
    else:
        banner("STAGE 2 — embedding clustering")
        try:
            import src.stage2_clustering as stage2_clustering
            stage2_clustering.main(limit=limit)
        except ImportError as e:
            print(f"[warn] {e}")
            print("[warn] pip install sentence-transformers to enable stage 2")
            print("[warn] continuing — stage 3 will group by prompt_type_rule")

    banner("STAGE 3 — duplicates & quality flags")
    stage3_quality.main(limit=limit)

    if skip_mysql:
        print("\n[skip] stage 4 MySQL load disabled by flag")
    else:
        banner("STAGE 4 — load to MySQL")
        try:
            import src.stage4_load_mysql as stage4_load_mysql
            stage4_load_mysql.main()
        except Exception as e:
            print(f"[warn] MySQL load failed: {e}")
            print("[warn] check credentials in config.py and that "
                  "sql/01_schema.sql has been run")

    print(f"\n[done] pipeline finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-mysql", action="store_true")
    ap.add_argument("--skip-clustering", action="store_true")
    sys.exit(main(**vars(ap.parse_args())))
