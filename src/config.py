"""
Central configuration for the corpus audit pipeline.

Every threshold that affects a finding lives here, not buried in the code.
If a reviewer asks "why 0.90 for duplicates?", the answer should be one
line in this file plus a sentence in the README.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Set this to a local .parquet path once downloaded — faster and offline-safe.
RAW_PARQUET = (
    "hf://datasets/vicgalle/alpaca-gpt4/data/"
    "train-00000-of-00001-6ef3991c06080e14.parquet"
)

ENRICHED_PARQUET = DATA_DIR / "enriched.parquet"      # full table, gitignored
SAMPLE_CSV = DATA_DIR / "enriched_sample.csv"         # 500 rows, committed
AUDIT_REPORT = OUTPUT_DIR / "audit_summary.json"
REVIEW_SAMPLE_CSV = OUTPUT_DIR / "manual_review_sample.csv"

# ---------------------------------------------------------------- clustering
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_SAMPLE_SIZE = 15_000   # fit K-Means on this many, assign the rest
K_RANGE = range(5, 13)           # candidate cluster counts for silhouette sweep
SILHOUETTE_SAMPLE = 5_000        # silhouette on a subsample; full 52K is O(n^2)
RANDOM_SEED = 42

# ---------------------------------------------------------------- duplicates
# Cosine similarity on TF-IDF instruction vectors. 0.90 is deliberately
# conservative: it catches near-verbatim restatements without flagging
# instructions that merely share a template ("Write a poem about X").
DUPLICATE_THRESHOLD = 0.90
TFIDF_MAX_FEATURES = 20_000

# ---------------------------------------------------------------- quality flags
MIN_OUTPUT_WORDS = 15        # below this = too short for a substantive answer
EMPTY_OUTPUT_WORDS = 3       # below this = effectively empty
REPEAT_NGRAM_N = 5           # n-gram size for degeneration check
REPEAT_NGRAM_MIN_COUNT = 3   # same 5-gram 3+ times = repetitive
TERMINAL_PUNCT = ('.', '!', '?', '"', ')', ':', '`')
FLAG_PENALTY = 20            # adequacy_score = 100 - (n_flags * FLAG_PENALTY)
LOW_ADEQUACY_CUTOFF = 60     # <= this is reported as "low adequacy"

# Flesch scores are only meaningful on prose. Outputs that are mostly code
# or tables get excluded from the readability-outlier flag rather than
# silently producing large negative scores.
CODE_MARKERS = ('```', 'def ', 'import ', '<html', 'SELECT ', '{', ';')
CODE_MARKER_MIN_HITS = 2

# ---------------------------------------------------------------- mysql
import os
from sqlalchemy import URL

MYSQL = {
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD"),  # never hardcode
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("MYSQL_PORT", 3306)),
    "database": os.environ.get("MYSQL_DATABASE", "corpus_audit"),
}

# URL.create escapes special characters in the password automatically.
# Hand-built f-string URLs break on passwords containing '@', ':', '/' etc.
DATABASE_URL = URL.create(
    "mysql+pymysql",
    username=MYSQL["user"],
    password=MYSQL["password"],
    host=MYSQL["host"],
    port=MYSQL["port"],
    database=MYSQL["database"],
    query={"charset": "utf8mb4"},
)
MYSQL_TABLE = "instruction_corpus"
LOAD_CHUNKSIZE = 1_000