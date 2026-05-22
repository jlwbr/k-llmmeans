import numpy as np

from conftest import DOCS, simple_embeddings, simple_summary
from k_llmmeans import kLLMmeans


def test_fit_predict_with_custom_functions():
    model = kLLMmeans(
        n_clusters=2,
        embedding_fn=simple_embeddings,
        summarizer_fn=simple_summary,
        max_llm_iter=2,
        random_state=0,
    )

    labels = model.fit_predict(DOCS)
    predictions = model.predict(DOCS)

    assert labels.shape == (len(DOCS),)
    np.testing.assert_array_equal(predictions, labels)
    assert len(model.summaries_) == 2
    assert model.n_iter_ >= 1
    assert model.cluster_centers_.shape[0] == 2
