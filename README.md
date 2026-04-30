# k-llmmeans

Scikit-learn compatible implementation of **k-LLMmeans** for text clustering with summary-based centroids.

This package adapts the original research code into an estimator API you can use with familiar `fit`, `predict`, and `fit_predict` workflows.

- Original implementation: [jairoadiazr/k-LLMmeans](https://github.com/jairoadiazr/k-LLMmeans)
- Paper: [Summaries as Centroids for Interpretable and Scalable Text Clustering (arXiv:2502.09667)](https://arxiv.org/abs/2502.09667)

## What This Package Provides

- `kLLMmeans` estimator implementing `BaseEstimator` + `ClusterMixin`
- scikit-learn style methods:
  - `fit(X)`
  - `predict(X)`
  - `fit_predict(X)`
- configurable document embedding function (`embedding_fn`)
- configurable cluster summarization function (`summarizer_fn`) or DSPy-backed LLM summarization
- optional precomputed embedding support for faster iterative experimentation

## Installation

```bash
pip install k-llmmeans
```

Or from source:

```bash
pip install -e .
```

## Quick Start

```python
import dspy
from k_llmmeans import kLLMmeans

# Option 1: pass an LM directly to the estimator
lm = dspy.LM("openai/gpt-5-mini")

docs = [
    "How to optimize SQL queries for large tables?",
    "What is the best way to tune a random forest model?",
    "PostgreSQL index strategy for analytics workloads",
    "Cross-validation tips for imbalanced classification",
]

model = kLLMmeans(
    n_clusters=2,
    llm=lm,
    max_llm_iter=5,
    random_state=0,
)

labels = model.fit_predict(docs)
print(labels)
print(model.summaries_)  # human-readable cluster summaries
```

## Using Custom Embeddings and Summarization

You can fully control both the embedding and summarization steps:

```python
from sentence_transformers import SentenceTransformer
from k_llmmeans import kLLMmeans

encoder = SentenceTransformer("all-MiniLM-L6-v2")

def embedding_fn(texts: list[str]):
    return encoder.encode(texts)

def summarizer_fn(cluster_texts: list[str]) -> str:
    # Replace with your own deterministic or LLM summarizer
    return " | ".join(cluster_texts[:2])

model = kLLMmeans(
    n_clusters=3,
    embedding_fn=embedding_fn,
    summarizer_fn=summarizer_fn,
)

model.fit(["text a", "text b", "text c", "text d"])
```

## API Notes

- Input `X` should be `list[str]`.
- The estimator stores standard fitted attributes such as:
  - `labels_`
  - `cluster_centers_`
  - `n_iter_`
- Additional clustering interpretability attributes:
  - `summaries_`
  - `summary_embeddings_`
  - `summaries_evolution_`
  - `centroids_evolution_`

## Citation

If you use this package in research or production work, please cite the original paper:

```bibtex
@article{diazrodriguez2025summaries,
  title={Summaries as Centroids for Interpretable and Scalable Text Clustering},
  author={Diaz-Rodriguez, Jairo},
  journal={arXiv preprint arXiv:2502.09667},
  year={2025}
}
```

Paper URL: [https://arxiv.org/abs/2502.09667](https://arxiv.org/abs/2502.09667)

## Acknowledgment

This package is a scikit-learn compatible adaptation of the original project:
[https://github.com/jairoadiazr/k-LLMmeans](https://github.com/jairoadiazr/k-LLMmeans)
