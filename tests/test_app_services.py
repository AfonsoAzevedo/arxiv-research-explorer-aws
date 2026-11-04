"""
Test the classification and semantic search services.

These tests use the saved production classifier and cached
semantic search data without making new AWS requests.
"""

import unittest

from app.classifier_service import (
    load_classifier,
    classify_abstract,
)
from app.search_service import (
    load_search_service,
    find_related_papers,
)
from semantic_search.query_service import QueryNotCached


class TestClassifierService(unittest.TestCase):
    """Check that the saved classifier works with valid and invalid inputs."""

    @classmethod
    def setUpClass(cls):
        # Load the production classifier once for all classification tests.
        cls.model = load_classifier()

    def test_classification_prediction(self):
        # Check that the model returns a valid category and three predictions.
        abstract = (
            "We propose an image classification system using "
            "convolutional neural networks and vision transformers "
            "to detect objects in digital images."
        )

        result = classify_abstract(self.model, abstract)

        self.assertIn(result["category_code"], self.model.classes_)
        self.assertEqual(len(result["top_categories"]), 3)
        self.assertEqual(
            result["top_categories"][0]["code"],
            result["category_code"],
        )

    def test_empty_abstract(self):
        # Empty abstracts should be rejected before model inference.
        with self.assertRaises(ValueError):
            classify_abstract(self.model, "   ")


class TestSearchService(unittest.TestCase):
    """Check cached retrieval and the offline query restriction."""

    @classmethod
    def setUpClass(cls):
        # Load the existing 368-paper FAISS index once.
        cls.service = load_search_service()

    def test_cached_search(self):
        # Use an existing cached embedding without invoking Bedrock.
        query = (
            "Deep learning methods for detecting "
            "cybersecurity threats and network intrusions"
        )

        results = find_related_papers(
            self.service,
            query,
            top_k=5,
        )

        self.assertEqual(len(results), 5)

        for paper in results:
            self.assertTrue(paper["title"])
            self.assertTrue(paper["abstract"])
            self.assertTrue(paper["paper_id"])
            self.assertTrue(paper["arxiv_url"])

    def test_uncached_search(self):
        # The offline service must reject an uncached query.
        query = "DS483_AUTOMATED_OFFLINE_TEST_20261104"

        with self.assertRaises(QueryNotCached):
            find_related_papers(self.service, query)

    def test_empty_search(self):
        # Empty search input should be rejected immediately.
        with self.assertRaises(ValueError):
            find_related_papers(self.service, "   ")


if __name__ == "__main__":
    unittest.main()
