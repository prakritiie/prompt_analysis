import pandas as pd
from src.config import ENRICHED_PARQUET

# Make sure EVERY cluster_id printed in Step 1 is in this dictionary
CLUSTER_NAMES = {
    0: "Creative Writing",
    1: "Code & Technical",
    2: "Factual Explanation",
    3: "Math & Reasoning",
    4: "Text Editing & Summarization",
    5: "Roleplay & Conversation",
    6: "Classification & Extraction",
    7: "General Knowledge QA",
    # ...add any missing cluster_ids here!
}

df = pd.read_parquet(ENRICHED_PARQUET)
df["cluster_label"] = df["cluster_id"].map(CLUSTER_NAMES)

# Debug helper to show exactly which IDs are missing if it fails again
missing_ids = df[df["cluster_label"].isna()]["cluster_id"].unique()
missing_count = df["cluster_label"].isna().sum()

assert missing_count == 0, f"Unmapped cluster_ids found: {missing_ids}. Add these keys to CLUSTER_NAMES!"

df.to_parquet(ENRICHED_PARQUET, index=False)
print("Clusters renamed successfully!")
print(df["cluster_label"].value_counts())