import unittest

import numpy as np

from k_llmmeans import kLLMmeans


DOCS = [
    "How to optimize SQL queries for large tables?",
    "PostgreSQL index strategy for analytics workloads",
    "What is the best way to tune a random forest model?",
    "Cross-validation tips for imbalanced classification",
]


def simple_embeddings(texts):
    rows = []
    for text in texts:
        lower = text.lower()
        rows.append(
            [
                float(any(term in lower for term in ("sql", "postgresql", "query"))),
                float(any(term in lower for term in ("forest", "classification"))),
                float(len(lower)),
            ]
        )
    return np.asarray(rows, dtype=float)


def simple_summary(texts):
    return " ".join(texts[:2])


class KLLMmeansTests(unittest.TestCase):
    def test_fit_predict_with_custom_functions(self):
        model = kLLMmeans(
            n_clusters=2,
            embedding_fn=simple_embeddings,
            summarizer_fn=simple_summary,
            max_llm_iter=2,
            random_state=0,
        )

        labels = model.fit_predict(DOCS)
        predictions = model.predict(DOCS)

        self.assertEqual(labels.shape, (len(DOCS),))
        np.testing.assert_array_equal(predictions, labels)
        self.assertEqual(len(model.summaries_), 2)
        self.assertGreaterEqual(model.n_iter_, 1)
        self.assertEqual(model.cluster_centers_.shape[0], 2)

    def test_precomputed_embeddings_are_used_and_sanitized(self):
        precomputed = simple_embeddings(DOCS)
        precomputed[0, 0] = np.nan

        def fail_if_called(texts):
            raise AssertionError(f"embedding_fn should not be called for {texts}")

        model = kLLMmeans(
            n_clusters=2,
            embedding_fn=fail_if_called,
            summarizer_fn=simple_summary,
            max_llm_iter=1,
            random_state=0,
        )

        model.fit(
            DOCS,
            precomputed_embeddings=precomputed,
            precomputed_summary_embeddings=simple_embeddings([simple_summary(DOCS[:2]), simple_summary(DOCS[2:])]),
        )

        self.assertTrue(np.isfinite(model.cluster_centers_).all())
        self.assertTrue(np.isfinite(model.summary_embeddings_).all())

    def test_requires_llm_when_no_custom_summarizer_is_configured(self):
        model = kLLMmeans(n_clusters=2, embedding_fn=simple_embeddings)

        with self.assertRaisesRegex(ValueError, "No LLM configured"):
            model.fit(DOCS)


if __name__ == "__main__":
    unittest.main()
