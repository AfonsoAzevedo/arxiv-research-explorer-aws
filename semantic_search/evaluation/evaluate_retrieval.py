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
    # Read the approved relevance grades for each query and candidate paper
    with JUDGMENTS_PATH.open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    # Check that all 174 approved judgments are present
    assert len(rows) == 174

    judgments = {}

    # Match every relevance grade to its query and paper ID
    for row in rows:
        key = (row["query_id"], row["candidate_id"])
        grade = int(row["relevance_grade"])

        assert grade in (0, 1, 2)
        assert key not in judgments
        judgments[key] = grade

    # Load the saved TF-IDF and Titan rankings for the same 368 papers
    rankings = json.loads(RANKINGS_PATH.read_text())
    assert rankings["corpus_size"] == 368
    assert rankings["query_count"] == 10

    results = []

    # Evaluate both retrieval methods using the same query
    for query in rankings["queries"]:
        query_id = query["query_id"]

        # The ideal ranking uses the pooled relevance judgments.
        # Build the shared relevance pool from both methods' top results
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
        # Use the best possible ordering to normalize nDCG
        ideal_grades = sorted(pool.values(), reverse=True)

        # Calculate Precision@5 and nDCG@10 for TF-IDF
        tfidf = evaluate_ranking(
            query["tfidf_top_10"], pool, ideal_grades
        )
        # Calculate the same metrics for Titan embeddings
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

    # Average each metric across the ten evaluation queries
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
