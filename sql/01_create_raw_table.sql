-- Define the raw arXiv metadata table stored in Amazon S3.
CREATE EXTERNAL TABLE IF NOT EXISTS ds483_test.papers_raw (
    id STRING,
    submitter STRING,
    authors STRING,
    title STRING,
    comments STRING,
    journal_ref STRING,
    doi STRING,
    report_no STRING,
    categories STRING,
    license STRING,
    abstract STRING,
    versions ARRAY<STRUCT<version:STRING,created:STRING>>,
    update_date STRING,
    authors_parsed ARRAY<ARRAY<STRING>>
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES (
    'mapping.journal_ref' = 'journal-ref',
    'mapping.report_no' = 'report-no'
)
STORED AS INPUTFORMAT
    'org.apache.hadoop.mapred.TextInputFormat'
OUTPUTFORMAT
    'org.apache.hadoop.hive.ql.io.IgnoreKeyTextOutputFormat'
LOCATION 's3://ds483-arxiv-test/raw/'
TBLPROPERTIES (
    'classification' = 'json',
    'typeOfData' = 'file'
);
