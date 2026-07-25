import pandas as pd
from src.stage4_load_mysql import build_engine

# The 5 analytical views created in MySQL
VIEWS = [
    "v_cluster_composition",
    "v_worst_by_cluster",
    "v_length_quality",
    "v_vs_cluster_avg",
    "v_duplicate_rate",
]

def main():
    engine = build_engine()
    print("[export] Connecting to MySQL and exporting views to CSV...")

    for view in VIEWS:
        try:
            df = pd.read_sql_table(view, con=engine)
            output_path = f"data/{view}.csv"
            df.to_csv(output_path, index=False)
            print(f"  ✓ Exported {view}.csv ({len(df):,} rows)")
        except Exception as e:
            print(f"  ✗ Failed to export {view}: {e}")

    print("\n[done] All available views exported to data/ folder!")

if __name__ == "__main__":
    main()