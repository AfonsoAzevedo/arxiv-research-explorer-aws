-- Count the records in the full arXiv dataset.
-- Compare this result with the 3,233,620 locally validated records.
SELECT COUNT(*) AS total_papers
FROM ds483_test.papers_raw;
