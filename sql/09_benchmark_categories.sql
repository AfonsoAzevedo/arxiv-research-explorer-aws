-- Count papers by their first listed category in the raw JSONL dataset.
SELECT
    SPLIT_PART(categories, ' ', 1) AS primary_category,
    COUNT(*) AS paper_count
FROM ds483_test.papers_raw
GROUP BY 1
ORDER BY paper_count DESC
LIMIT 10;

-- Run the same category analysis against the Parquet dataset.
SELECT
    SPLIT_PART(categories, ' ', 1) AS primary_category,
    COUNT(*) AS paper_count
FROM ds483_test.papers_parquet
GROUP BY 1
ORDER BY paper_count DESC
LIMIT 10;
