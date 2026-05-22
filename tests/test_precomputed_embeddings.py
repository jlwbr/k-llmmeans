import numpy as np

from conftest import DOCS, simple_embeddings, simple_summary
from k_llmmeans import kLLMmeans


def test_precomputed_embeddings_are_used_and_sanitized():
    precomputed = simple_embeddings(DOCS)
    precomputed[0, 0] = np.nan
    summary_texts = [simple_summary(DOCS[:2]), simple_summary(DOCS[2:])]

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
        precomputed_summary_embeddings=simple_embeddings(summary_texts),
    )

    assert np.isfinite(model.cluster_centers_).all()
    assert np.isfinite(model.summary_embeddings_).all()
