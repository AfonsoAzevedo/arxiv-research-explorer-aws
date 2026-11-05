-- Convert 1,000 arXiv records from JSONL to compressed Parquet.
-- Keep the original metadata fields to verify schema compatibility.
CREATE TABLE ds483_test.papers_parquet_pilot
WITH (
    format = 'PARQUET',
    parquet_compression = 'SNAPPY',
    external_location = 's3://ds483-arxiv-test/processed/papers_parquet_pilot/'
) AS
SELECT
    id,
    submitter,
    authors,
    title,
    comments,
    journal_ref,
    doi,
    report_no,
    categories,
    license,
    abstract,
    versions,
    update_date,
    authors_parsed
FROM ds483_test.papers_raw
LIMIT 1000;
