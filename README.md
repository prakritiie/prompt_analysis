# CorpusLens: Alpaca-GPT4 Corpus Audit & Quality Engineering Pipeline


An end-to-end data pipeline that audits a 52,002-row instruction-tuning dataset (`alpaca-gpt4`) for redundancy, task imbalance, and degenerate outputs—recommending exactly which rows to prune before fine-tuning.

`Python` · `Pandas` · `Scikit-Learn` · `Sentence-Transformers` · `MySQL` 



## Executive Summary & Impact

Fine-tuning open-source LLMs on unaudited synthetic datasets wastes expensive GPU compute and risks inheriting model degeneration (e.g., repetition loops, truncated outputs). **CorpusLens** automates pre-training data hygiene to maximize dataset efficiency.

* **52,002 rows** audited across 7 semantic task clusters.
* **3,529 rows (6.79%)** recommended for deletion or regeneration.
* **48,473 clean rows (93.21%)** preserved for optimal fine-tuning.


## Key Metrics & Audit Results

| Metric | Value | Insights / Impact |
|---|---|---|
| **Rows Analysed** | **52,002** | Full synthetic `vicgalle/alpaca-gpt4` dataset |
| **Rule-Based "Other" Rate** | **39.3%** | Motivated using `all-MiniLM-L6-v2` embeddings + K-Means |
| **Task Clusters ($K$)** | **7** | Selected via silhouette sweep ($K=5..12$, score: **0.0603**) |
| **Near-Duplicates ($\ge$0.90 Cosine)** | **501 (0.96%)** | Identified via in-cluster TF-IDF vectorization |
| **Low-Adequacy Rows ($\le$ 60)** | **3,052 (5.87%)** | Tripped $\ge 2$ quality failure flags |
| **Mean Corpus Adequacy** | **92.23 / 100** | Overall dataset health score |
| **Recommended Drop** | **3,529 (6.79%)** | **3,052 low-quality + 477 duplicate non-overlaps** |
| **Clean Corpus Remaining** | **48,473 (93.21%)** | Production-ready fine-tuning dataset |

---

## Architecture & Data Flow

```text
  [Raw Parquet] 
       │
       ▼
┌───────────────┐
│ Stage 1: Clean│ ➔ Pandas: Word counts, readability (Flesch), code-detection.
└──────┬────────┘
       ▼
┌───────────────┐
│ Stage 2: ML   │ ➔ Sentence-Transformers + K-Means clustering (Silhouette sweep).
└──────┬────────┘
       ▼
┌───────────────┐
│ Stage 3: Audit│ ➔ TF-IDF (Cosine ≥ 0.90) + 5 Boolean Quality Flags ➔ Adequacy Score.
└──────┬────────┘
       ▼
┌───────────────┐
│ Stage 4: MySQL│ ➔ Hand-written DDL + 9 Analytical Views (Window Functions & CTEs).
└──────┬────────┘
       ▼
 [Power BI Dash] ➔ Visualizing cluster share, duplicate density & pruning candidates.

```

---

## Key Technical Highlights

1. **In-Cluster Deduplication:** Instead of an $O(N^2)$ cross-comparison, near-duplicates are flagged using **TF-IDF + Cosine Similarity ($\ge 0.90$) within each cluster**, preserving the earliest copy of every instruction.
2. **Context-Aware Quality Scoring:** Bypasses readability scoring for code snippets (`is_code_like`) and evaluates length/readability outliers relative to cluster averages rather than global metrics.
3. **Relational Serving Layer:** MySQL 8.0 views utilize `RANK() OVER (PARTITION BY...)`, `NTILE(4)`, `PERCENT_RANK()`, and windowed aggregates (`SUM() OVER ()`) to power interactive BI dashboards without data duplication.

---

## Strategic Recommendations

1. **Deduplicate:** Drop **501 flagged near-duplicate instructions** to reduce redundant gradient updates and prevent memorization.
2. **Rebalance:** Shift data generation efforts to under-represented clusters (e.g., *Math & Reasoning* vs. *Text Editing & Summarization*).
3. **Regenerate (Don't Delete):** For the **3,052 low-adequacy rows**, re-prompt an LLM to regenerate the responses rather than discarding valid prompt instructions.

---

## How to Run

```bash
# Clone & install dependencies
git clone [https://github.com/your-username/llm-corpus-audit.git](https://github.com/your-username/llm-corpus-audit.git)
cd llm-corpus-audit
pip install -r requirements.txt

# Run full pipeline (Stages 1-3, skipping MySQL load)
python src/run_all.py --skip-mysql

# Run Stage 4 & load analytical views (MySQL required)
mysql -u root -p < sql/01_schema.sql
python -m src.export_views
mysql -u root -p corpus_audit < sql/02_views.sql

```
---

## Repository Structure

```text
llm-corpus-audit/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py                 # Centralized configuration & thresholds
│   ├── utils.py                  # Shared text extraction & processing utilities
│   ├── stage1_clean_enrich.py    # Structural features & rule-based classification
│   ├── stage2_clustering.py      # Embeddings, K-Means & silhouette selection
│   ├── stage3_quality.py         # TF-IDF deduplication & quality flagging
│   ├── stage4_load_mysql.py      # MySQL database ingestion
│   ├── export_views.py           # SQL views export to CSV for Power BI
│   └── run_all.py                # End-to-end pipeline execution runner
├── sql/
│   ├── 01_schema.sql             # MySQL table definition
│   └── 02_views.sql              # 9 analytical views with window functions
├── outputs/
│   ├── audit_summary.json        # Quantitative pipeline outputs
│   └── manual_review_sample.csv  # 20-row worst-performer review sample
└── data/                         # Project datasets (gitignored)

```

## Limitations

* **Corpus-Level Analysis:** Evaluates static dataset properties only; does not model live human prompting behavior or session dynamics.
* **Heuristic Thresholds:** Quality and duplicate parameters ($0.90$ similarity, $15$-word minimum, $5$-gram loops) are configurable heuristics centralized in `src/config.py`.
* **No Direct Fine-Tuning Delta:** Identifies data quality actions, but downstream LLM accuracy improvement requires training and benchmarking two separate models.

