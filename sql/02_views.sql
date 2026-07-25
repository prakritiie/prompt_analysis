-- =====================================================================
-- 02_views.sql — analytical layer
--
-- Run after loading:  mysql -u root -p corpus_audit < sql/02_views.sql
--
-- One view per dashboard visual. Each answers exactly one question from
-- the problem statement:
--    Q1 is the corpus balanced?     -> v1, v2, v7
--    Q2 how much is redundant?      -> v3, v4
--    Q3 which rows are low quality? -> v5, v6, v8
-- Requires MySQL 8.0+ (window functions).
-- =====================================================================

USE corpus_audit;

-- ---------------------------------------------------------------------
-- V1. Corpus composition by cluster
-- SUM() OVER () gives each row's share of the whole without a self-join.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_cluster_composition AS
SELECT
    cluster_label,
    COUNT(*)                                                   AS row_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)         AS pct_of_corpus,
    ROUND(AVG(instruction_word_count), 1)                      AS avg_instruction_len,
    ROUND(AVG(output_word_count), 1)                           AS avg_response_len,
    ROUND(AVG(flesch_score), 1)                                AS avg_readability,
    ROUND(AVG(adequacy_score), 1)                              AS avg_adequacy
FROM instruction_corpus
GROUP BY cluster_label
ORDER BY row_count DESC;


-- ---------------------------------------------------------------------
-- V2. Rule-based classifier vs clustering coverage
-- The 'Other' share here is the justification for stage 2.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_rule_vs_cluster AS
SELECT
    prompt_type_rule,
    COUNT(*)                                            AS row_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)  AS pct_of_corpus,
    COUNT(DISTINCT cluster_label)                       AS distinct_clusters_spanned
FROM instruction_corpus
GROUP BY prompt_type_rule
ORDER BY row_count DESC;


-- ---------------------------------------------------------------------
-- V3. Redundancy by cluster -> drives the deduplication recommendation
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_duplicate_rate AS
SELECT
    cluster_label,
    COUNT(*)                                                        AS total_rows,
    SUM(is_exact_duplicate)                                         AS exact_dupes,
    SUM(is_near_duplicate)                                          AS near_dupes,
    SUM(CASE WHEN is_exact_duplicate = 1 OR is_near_duplicate = 1
             THEN 1 ELSE 0 END)                                     AS any_dupe,
    ROUND(100.0 * SUM(CASE WHEN is_exact_duplicate = 1
                             OR is_near_duplicate = 1
                           THEN 1 ELSE 0 END) / COUNT(*), 2)        AS dupe_pct
FROM instruction_corpus
GROUP BY cluster_label
ORDER BY dupe_pct DESC;


-- ---------------------------------------------------------------------
-- V4. Duplicate pairs, kept row alongside the flagged copy
-- Self-join on duplicate_of so a reviewer can eyeball whether the
-- 0.90 threshold is behaving.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_duplicate_pairs AS
SELECT
    d.row_id                        AS flagged_row,
    LEFT(d.instruction, 90)         AS flagged_instruction,
    k.row_id                        AS kept_row,
    LEFT(k.instruction, 90)         AS kept_instruction,
    d.cluster_label
FROM instruction_corpus d
JOIN instruction_corpus k
  ON d.duplicate_of = k.row_id
WHERE d.is_near_duplicate = 1;


-- ---------------------------------------------------------------------
-- V5. Worst responses ranked WITHIN their own cluster
-- RANK() PARTITION BY, wrapped in a subquery because window functions
-- cannot appear in WHERE.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_worst_by_cluster AS
SELECT *
FROM (
    SELECT
        row_id,
        cluster_label,
        adequacy_score,
        n_flags,
        output_word_count,
        LEFT(instruction, 80)  AS instruction_preview,
        LEFT(`output`, 120)    AS output_preview,
        RANK() OVER (PARTITION BY cluster_id
                     ORDER BY adequacy_score ASC, output_word_count ASC) AS rk
    FROM instruction_corpus
) ranked
WHERE rk <= 20;


-- ---------------------------------------------------------------------
-- V6. Flag prevalence — which failure mode dominates
-- UNION ALL to pivot five columns into rows the dashboard can bar-chart.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_flag_prevalence AS
SELECT 'truncated'          AS flag_name, SUM(flag_truncated)           AS n,
       ROUND(100.0 * SUM(flag_truncated) / COUNT(*), 2)           AS pct
FROM instruction_corpus
UNION ALL
SELECT 'too_short',          SUM(flag_too_short),
       ROUND(100.0 * SUM(flag_too_short) / COUNT(*), 2)
FROM instruction_corpus
UNION ALL
SELECT 'empty_ish',          SUM(flag_empty_ish),
       ROUND(100.0 * SUM(flag_empty_ish) / COUNT(*), 2)
FROM instruction_corpus
UNION ALL
SELECT 'repetitive',         SUM(flag_repetitive),
       ROUND(100.0 * SUM(flag_repetitive) / COUNT(*), 2)
FROM instruction_corpus
UNION ALL
SELECT 'readability_outlier', SUM(flag_readability_outlier),
       ROUND(100.0 * SUM(flag_readability_outlier) / COUNT(*), 2)
FROM instruction_corpus;


-- ---------------------------------------------------------------------
-- V7. Response length quartiles vs quality
-- NTILE(4) to test whether short responses really are the bad ones.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_length_quality AS
SELECT
    quartile,
    COUNT(*)                          AS n_rows,
    MIN(output_word_count)            AS min_words,
    MAX(output_word_count)            AS max_words,
    ROUND(AVG(adequacy_score), 1)     AS avg_adequacy,
    ROUND(AVG(flesch_score), 1)       AS avg_readability
FROM (
    SELECT
        NTILE(4) OVER (ORDER BY output_word_count) AS quartile,
        output_word_count,
        adequacy_score,
        flesch_score
    FROM instruction_corpus
) q
GROUP BY quartile
ORDER BY quartile;


-- ---------------------------------------------------------------------
-- V8. Every row against its own cluster average
-- AVG() OVER (PARTITION BY ...) — the deviation column is what finds
-- responses that are anomalously stunted *for their task type*.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_deviation_from_cluster AS
SELECT
    row_id,
    cluster_label,
    output_word_count,
    ROUND(AVG(output_word_count) OVER (PARTITION BY cluster_id), 1) AS cluster_avg_len,
    ROUND(output_word_count
          - AVG(output_word_count) OVER (PARTITION BY cluster_id), 1) AS deviation,
    ROUND(PERCENT_RANK() OVER (PARTITION BY cluster_id
                               ORDER BY output_word_count), 3)        AS pct_rank_in_cluster,
    adequacy_score
FROM instruction_corpus;


-- ---------------------------------------------------------------------
-- V9. The headline: what to drop and what survives
-- CTE feeding the single number that goes on the recommendation slide.
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_cleaning_impact AS
WITH tagged AS (
    SELECT
        row_id,
        cluster_label,
        CASE
            WHEN is_exact_duplicate = 1 OR is_near_duplicate = 1 THEN 'drop_duplicate'
            WHEN adequacy_score <= 60                            THEN 'drop_low_quality'
            ELSE 'keep'
        END AS disposition
    FROM instruction_corpus
)
SELECT
    cluster_label,
    disposition,
    COUNT(*)                                                     AS n_rows,
    ROUND(100.0 * COUNT(*)
          / SUM(COUNT(*)) OVER (PARTITION BY cluster_label), 2)  AS pct_of_cluster
FROM tagged
GROUP BY cluster_label, disposition
ORDER BY cluster_label, disposition;
