-- Validate the number of records stored in the Parquet pilot.
-- Check that paper IDs and original submission dates are readable.
SELECT
    COUNT(*) AS total_papers,
    COUNT(DISTINCT id) AS unique_paper_ids,
    COUNT_IF(id IS NULL) AS missing_ids,
    COUNT_IF(versions[1].created IS NULL) AS missing_submission_dates
FROM ds483_test.papers_parquet_pilot;
