from types import SimpleNamespace

import numpy as np
import pytest

import k_llmmeans
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


def completion_response(content, prompt_tokens=7, completion_tokens=3):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


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


def test_requires_llm_when_no_custom_summarizer_is_configured(monkeypatch):
    monkeypatch.delenv("LITELLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    model = kLLMmeans(n_clusters=2, embedding_fn=simple_embeddings)

    with pytest.raises(ValueError, match="No LLM configured"):
        model.fit(DOCS)


def test_fit_uses_litellm_completion(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return completion_response(f"summary {len(calls)}")

    monkeypatch.setattr(k_llmmeans.litellm, "completion", fake_completion)

    model = kLLMmeans(
        n_clusters=2,
        embedding_fn=simple_embeddings,
        llm={"model": "openai/test-model", "temperature": 0.2},
        prompt="Summarize this cluster for tests.",
        text_type="Document:",
        max_llm_iter=1,
        random_state=0,
    )

    model.fit(DOCS)

    assert len(calls) == 2
    assert model.summaries_ == ["summary 1", "summary 2"]
    assert model._llm_kwargs_resolved == {
        "model": "openai/test-model",
        "temperature": 0.2,
    }
    assert all(call["model"] == "openai/test-model" for call in calls)
    assert all(call["temperature"] == 0.2 for call in calls)
    assert all(call["messages"][0]["role"] == "user" for call in calls)
    assert all("Document:" in call["messages"][0]["content"] for call in calls)
    assert all(
        "Summarize this cluster for tests." in call["messages"][0]["content"]
        for call in calls
    )
