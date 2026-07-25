-- =====================================================================
-- 01_schema.sql — corpus audit fact table
--
-- Run first:  mysql -u root -p < sql/01_schema.sql
--
-- Notes:
--   * utf8mb4 is required. GPT-4 outputs contain emoji and non-BMP
--     characters that utf8mb3 silently truncates.
--   * `input` and `output` are avoided as bare column names; `input_text`
--     and a backticked `output` keep things unambiguous.
--   * Indexes exist because the window-function views in 02_views.sql
--     partition and sort on these columns across 52K rows.
-- =====================================================================

CREATE DATABASE IF NOT EXISTS corpus_audit
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE corpus_audit;

DROP TABLE IF EXISTS instruction_corpus;

CREATE TABLE instruction_corpus (
    row_id                   INT           NOT NULL PRIMARY KEY,

    -- raw content
    instruction              TEXT,
    input_text               TEXT,
    `output`                 LONGTEXT,

    -- structural features
    has_input                TINYINT       DEFAULT 0,
    instruction_word_count   INT,
    output_word_count        INT,
    sentence_count           INT,
    words_per_sentence       DECIMAL(8,2),

    -- categorisation
    prompt_type_rule         VARCHAR(50),   -- baseline classifier
    cluster_id               INT,           -- embedding K-Means
    cluster_label            VARCHAR(100),

    -- readability
    flesch_score             DECIMAL(8,2),  -- NULL for code-like outputs
    readability_level        VARCHAR(30),
    is_code_like             TINYINT       DEFAULT 0,

    -- redundancy
    is_exact_duplicate       TINYINT       DEFAULT 0,
    is_near_duplicate        TINYINT       DEFAULT 0,
    duplicate_of             INT           NULL,  -- row_id of the kept copy

    -- quality flags
    flag_truncated           TINYINT       DEFAULT 0,
    flag_too_short           TINYINT       DEFAULT 0,
    flag_empty_ish           TINYINT       DEFAULT 0,
    flag_repetitive          TINYINT       DEFAULT 0,
    flag_readability_outlier TINYINT       DEFAULT 0,
    n_flags                  INT           DEFAULT 0,
    adequacy_score           INT           DEFAULT 100,

    INDEX idx_cluster    (cluster_id),
    INDEX idx_adequacy   (adequacy_score),
    INDEX idx_dupe       (is_near_duplicate),
    INDEX idx_len        (output_word_count)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
