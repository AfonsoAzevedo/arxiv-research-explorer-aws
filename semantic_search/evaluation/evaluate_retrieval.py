"""Evaluate TF-IDF and Titan retrieval using approved relevance judgments."""

import csv
import json
import math
import statistics
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parent
RANKINGS_PATH = EVALUATION_DIR / "matched_368_rankings.json"
JUDGMENTS_PATH = (
    EVALUATION_DIR / "DS483_approved_relevance_judgments_368.csv"
)


def dcg(grades):
    """Compute discounted cumulative gain for graded relevance."""
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, 1)
    )


def evaluate_ranking(results, judgments, ideal_grades):
    """Calculate Precision@5 and nDCG@10 for one ranking."""
    grades = [judgments[item["paper_id"]] for item in results]
    ideal_dcg = dcg(ideal_grades[:10])

    return {
        "precision_at_5": sum(
            grade >= 1 for grade in grades[:5]
        ) / 5,
        "ndcg_at_10": (
            dcg(grades[:10]) / ideal_dcg
            if ideal_dcg > 0 else 0.0
        ),
    }


def main():
    # Load the approved judgments, keyed by query and candidate paper.
    with JUDGMENTS_PATH.open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 174

    judgments = {}

    for row in rows:
        key = (row["query_id"], row["candidate_id"])
        grade = int(row["relevance_grade"])

        assert grade in (0, 1, 2)
        assert key not in judgments
        judgments[key] = grade

    rankings = json.loads(RANKINGS_PATH.read_text())
    assert rankings["corpus_size"] == 368
    assert rankings["query_count"] == 10

    results = []

    for query in rankings["queries"]:
        query_id = query["query_id"]

        # The ideal ranking uses the pooled relevance judgments.
        pool = {
            paper_id: grade
            for (qid, paper_id), grade in judgments.items()
            if qid == query_id
        }

        expected_ids = {
            item["paper_id"]
            for method in ("tfidf_top_10", "titan_top_10")
            for item in query[method]
        }

        assert set(pool) == expected_ids
        ideal_grades = sorted(pool.values(), reverse=True)

        tfidf = evaluate_ranking(
            query["tfidf_top_10"], pool, ideal_grades
        )
        titan = evaluate_ranking(
            query["titan_top_10"], pool, ideal_grades
        )

        results.append({
            "category": query["sampling_group"],
            "tfidf": tfidf,
            "titan": titan,
        })

    print("TF-IDF VS TITAN RETRIEVAL EVALUATION")
    print("Corpus: 368 papers")
    print("Queries:", len(results))
    print("Judgments:", len(judgments))
    print()

    for metric in ("precision_at_5", "ndcg_at_10"):
        lexical = [
            row["tfidf"][metric] for row in results
        ]
        semantic = [
            row["titan"][metric] for row in results
        ]

        print(metric)
        print(f"  TF-IDF: {statistics.mean(lexical):.4f}")
        print(f"  Titan:  {statistics.mean(semantic):.4f}")
        print(
            "  Titan / TF-IDF wins / ties:",
            sum(b > a for a, b in zip(lexical, semantic)),
            "/",
            sum(a > b for a, b in zip(lexical, semantic)),
            "/",
            sum(a == b for a, b in zip(lexical, semantic)),
        )


if __name__ == "__main__":
    main()
