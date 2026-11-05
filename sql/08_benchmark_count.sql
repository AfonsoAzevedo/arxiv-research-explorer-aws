-- Count all records in the raw JSONL dataset.
SELECT COUNT(*) AS total_papers
FROM ds483_test.papers_raw;

-- Run the same count against the converted Parquet dataset.
SELECT COUNT(*) AS total_papers
FROM ds483_test.papers_parquet;
