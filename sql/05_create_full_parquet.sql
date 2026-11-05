-- Convert the complete arXiv metadata dataset from JSONL to Parquet.
-- Preserve all 14 columns, including nested author and version metadata.
-- Use SNAPPY compression to reduce storage and query scan sizes.
CREATE TABLE ds483_test.papers_parquet
WITH (
    format = 'PARQUET',
    parquet_compression = 'SNAPPY',
    external_location = 's3://ds483-arxiv-test/processed/papers_parquet/'
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
FROM ds483_test.papers_raw;
