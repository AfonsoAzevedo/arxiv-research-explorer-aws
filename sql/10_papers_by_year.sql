-- Analyze the number of arXiv papers submitted each year.
-- Use the first version to identify the original submission year.
SELECT
    YEAR(
        DATE_PARSE(
            REGEXP_REPLACE(versions[1].created, '^[A-Za-z]{3}, ', ''),
            '%e %b %Y %H:%i:%s GMT'
        )
    ) AS submission_year,
    COUNT(*) AS paper_count
FROM ds483_test.papers_parquet
GROUP BY 1
ORDER BY 1;
