import pytest

from conftest import DOCS, simple_embeddings
from k_llmmeans import kLLMmeans


def test_requires_llm_when_no_custom_summarizer_is_configured(monkeypatch):
    monkeypatch.delenv("LITELLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    model = kLLMmeans(n_clusters=2, embedding_fn=simple_embeddings)

    with pytest.raises(ValueError, match="No LLM configured"):
        model.fit(DOCS)
