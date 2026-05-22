import k_llmmeans
from conftest import DOCS, completion_response, simple_embeddings
from k_llmmeans import kLLMmeans


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
