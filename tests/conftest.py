from types import SimpleNamespace

import numpy as np


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
