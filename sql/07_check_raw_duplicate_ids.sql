-- Check whether duplicate paper IDs already exist in the raw dataset.
-- Compare these results with the converted Parquet table.
SELECT
    COUNT(*) AS total_papers,
    COUNT(DISTINCT id) AS unique_paper_ids,
    COUNT(*) - COUNT(DISTINCT id) AS duplicate_id_occurrences
FROM ds483_test.papers_raw;
