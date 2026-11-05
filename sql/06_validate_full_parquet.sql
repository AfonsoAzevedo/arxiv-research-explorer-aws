-- Validate the complete Parquet dataset against the original arXiv metadata.
-- Check record counts, unique IDs, and required metadata fields.
SELECT
    COUNT(*) AS total_papers,
    COUNT(DISTINCT id) AS unique_paper_ids,
    COUNT_IF(id IS NULL) AS missing_ids,
    COUNT_IF(title IS NULL OR TRIM(title) = '') AS missing_titles,
    COUNT_IF(abstract IS NULL OR TRIM(abstract) = '') AS missing_abstracts,
    COUNT_IF(categories IS NULL OR TRIM(categories) = '') AS missing_categories,
    COUNT_IF(versions[1].created IS NULL) AS missing_submission_dates
FROM ds483_test.papers_parquet;
