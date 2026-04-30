import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any, Callable, Mapping

import dspy
import numpy as np
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

__version__ = "0.1.2"
__all__ = ["kLLMmeans"]

load_dotenv()

cluster_summarizer = dspy.Predict("prompt, text_type -> summary")

def summarize_cluster_with_dspy(
    texts: list[str], prompt: str = "", text_type: str = ""
) -> str:
    cluster_text = "\n".join(t for t in texts if t.strip())
    if not cluster_text:
        return ""

    if not prompt:
        prompt = (
            "Write a single sentence that represents the following cluster "
            "concisely:\n\n" + cluster_text
        )
    else:
        prompt = f"{prompt}\n\n{cluster_text}"

    text_type = text_type or "Sentence:"

    out = cluster_summarizer(prompt=prompt, text_type=text_type)
    return (out.summary or "").strip()


class kLLMmeans(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_clusters: int = 8,
        embedding_fn: Callable[[list[str]], np.ndarray] | None = None,
        embedding_cache: Mapping[str, np.ndarray] | None = None,
        summarizer_fn: Callable[[list[str]], str] | None = None,
        prompt: str = "",
        text_type: str = "",
        llm: dspy.LM | str | None = None,
        summary_workers: int = 1,
        show_progress: bool = False,
        verbose: bool = False,
        max_llm_iter: int = 5,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.embedding_fn = embedding_fn
        self.embedding_cache = embedding_cache
        self.summarizer_fn = summarizer_fn
        self.prompt = prompt
        self.text_type = text_type
        self.llm = llm
        self.summary_workers = summary_workers
        self.show_progress = show_progress
        self.verbose = verbose
        self.max_llm_iter = max_llm_iter
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        # Mutable cache used across runs to avoid re-embedding repeated texts.
        self._embedding_cache: dict[str, np.ndarray] = dict(embedding_cache or {})

    def _build_default_embedding_fn(
        self, text_data: list[str]
    ) -> Callable[[list[str]], np.ndarray]:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        vectorizer.fit(text_data)
        self._vectorizer = vectorizer
        return lambda texts: vectorizer.transform(texts).toarray()

    def _resolve_embedding_fn(self, text_data: list[str]) -> Callable[[list[str]], np.ndarray]:
        if self.embedding_fn is not None:
            return self.embedding_fn
        return self._build_default_embedding_fn(text_data)

    def _resolve_summarizer(self) -> Callable[[list[str]], str]:
        if self.summarizer_fn is not None:
            return self.summarizer_fn
        return lambda texts: summarize_cluster_with_dspy(
            texts, prompt=self.prompt, text_type=self.text_type
        )

    def _resolve_llm(self) -> dspy.LM:
        if isinstance(self.llm, dspy.LM):
            return self.llm
        if isinstance(self.llm, str):
            return dspy.LM(self.llm, api_key=os.getenv("OPENAI_API_KEY"))
        if dspy.settings.lm is not None:
            return dspy.settings.lm
        raise ValueError(
            "No LLM configured. Pass llm='openai/gpt-5-mini' (or another model) "
            "to kLLMmeans, or configure dspy globally with dspy.configure(lm=...)."
        )

    def _sanitize_embeddings(self, arr, name: str) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float64)
        if out.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got shape {out.shape}.")
        non_finite = ~np.isfinite(out)
        if np.any(non_finite):
            if self.verbose:
                print(
                    f"[sanitize] {name}: replacing {int(non_finite.sum())} non-finite values"
                )
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        # Bound values to avoid overflow in downstream matrix ops.
        out = np.clip(out, -1e6, 1e6)
        return out

    def _normalize_embeddings(self, arr: np.ndarray, name: str) -> np.ndarray:
        out = self._sanitize_embeddings(arr, name)
        if out.shape[1] == 0:
            return out
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        # Avoid divide-by-zero; zero rows stay zero after normalization.
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        out = out / safe_norms
        return self._sanitize_embeddings(out, f"{name}_normalized")

    @staticmethod
    def _to_1d_embedding(vec, *, name: str, expected_dim: int | None = None) -> np.ndarray:
        out = np.asarray(vec, dtype=np.float64).reshape(-1)
        if expected_dim is not None and out.shape[0] != expected_dim:
            raise ValueError(
                f"{name} embedding has inconsistent dimension {out.shape[0]} "
                f"(expected {expected_dim})."
            )
        return out

    def _get_embeddings_with_precomputed(
        self,
        texts: list[str],
        embedding_fn: Callable[[list[str]], np.ndarray],
        name: str,
        precomputed: np.ndarray | Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if isinstance(precomputed, np.ndarray):
            if precomputed.shape[0] != len(texts):
                raise ValueError(
                    f"{name} precomputed array must have one row per input text."
                )
            return self._sanitize_embeddings(precomputed, name)

        precomputed_map = precomputed if isinstance(precomputed, Mapping) else None
        dim: int | None = None
        resolved: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, text in enumerate(texts):
            vec = (
                precomputed_map.get(text)
                if precomputed_map is not None
                else self._embedding_cache.get(text)
            )
            if vec is None and precomputed_map is not None:
                vec = self._embedding_cache.get(text)

            if vec is None:
                missing_indices.append(idx)
                missing_texts.append(text)
                continue

            arr = self._to_1d_embedding(vec, name=name, expected_dim=dim)
            dim = arr.shape[0] if dim is None else dim
            resolved[idx] = arr

        if missing_texts:
            computed = self._sanitize_embeddings(embedding_fn(missing_texts), name)
            if computed.shape[0] != len(missing_texts):
                raise ValueError(
                    f"{name} embedding_fn output must have one row per input text."
                )
            if dim is not None and computed.shape[1] != dim:
                raise ValueError(
                    f"{name} computed embedding dimension {computed.shape[1]} does not "
                    f"match precomputed dimension {dim}."
                )
            dim = computed.shape[1] if dim is None else dim
            for row_idx, text in enumerate(missing_texts):
                vec = self._to_1d_embedding(computed[row_idx], name=name, expected_dim=dim)
                target_idx = missing_indices[row_idx]
                resolved[target_idx] = vec
                self._embedding_cache[text] = vec.copy()
        elif dim is None:
            return self._sanitize_embeddings(np.zeros((len(texts), 0), dtype=np.float64), name)

        out = np.vstack([vec for vec in resolved if vec is not None])
        if out.shape[0] != len(texts):
            raise ValueError(f"{name} embedding resolution failed for some inputs.")
        return self._sanitize_embeddings(out, name)

    @staticmethod
    def _history_slice(lm: dspy.LM, start_idx: int) -> list[dict[str, Any]]:
        history = getattr(lm, "history", None)
        if not isinstance(history, list):
            return []
        return [h for h in history[start_idx:] if isinstance(h, dict)]

    @staticmethod
    def _sum_usage_from_history(entries: list[dict[str, Any]]) -> dict[str, int]:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for entry in entries:
            usage = entry.get("usage")
            if not isinstance(usage, dict):
                continue
            prompt_tokens += int(
                usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            )
            completion_tokens += int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )
            total_tokens += int(usage.get("total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def fit(
        self,
        X,
        y=None,
        precomputed_embeddings: np.ndarray | Mapping[str, np.ndarray] | None = None,
        precomputed_summary_embeddings: np.ndarray | Mapping[str, np.ndarray] | None = None,
    ):
        if not isinstance(X, list):
            raise TypeError("X must be a list[str].")
        if len(X) < self.n_clusters:
            raise ValueError("n_clusters cannot exceed number of texts.")
        if self.summary_workers < 1:
            raise ValueError("summary_workers must be >= 1.")

        embedding_fn = self._resolve_embedding_fn(X)
        summarizer = self._resolve_summarizer()
        resolved_lm = self._resolve_llm()

        doc_features = self._get_embeddings_with_precomputed(
            X,
            embedding_fn=embedding_fn,
            name="doc_features",
            precomputed=precomputed_embeddings,
        )
        doc_features = self._normalize_embeddings(doc_features, "doc_features")
        if doc_features.shape[0] != len(X):
            raise ValueError("Embedding output must have one row per input text.")

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            max_iter=max(1, self.max_iter // max(1, self.max_llm_iter + 1)),
            n_init=10,
            random_state=self.random_state,
        )
        labels = kmeans.fit_predict(doc_features)
        centroids = kmeans.cluster_centers_

        summaries_evolution: list[list[str]] = []
        centroids_evolution: list[np.ndarray] = []
        summaries: list[str] = [""] * self.n_clusters
        summary_embeddings = np.zeros_like(centroids)

        if self.verbose:
            print(
                f"Starting fit: n_samples={len(X)}, n_clusters={self.n_clusters}, "
                f"max_llm_iter={self.max_llm_iter}, summary_workers={self.summary_workers}"
            )

        converged = False
        n_iter = 0
        iter_range = range(1, self.max_llm_iter + 1)
        iter_range = tqdm(
            iter_range,
            disable=not self.show_progress,
            desc="kLLMmeans iterations",
        )
        for iteration in iter_range:
            n_iter = iteration
            clustered_texts: dict[int, list[str]] = {i: [] for i in range(self.n_clusters)}
            for text, cid in zip(X, labels):
                clustered_texts[int(cid)].append(text)

            def summarize_cluster_idx(i: int) -> tuple[str, dict[str, Any]]:
                cur_texts = clustered_texts[i]
                if not cur_texts:
                    return "", {
                        "cluster": i,
                        "n_texts": 0,
                        "input_chars": 0,
                        "summary_chars": 0,
                        "latency_s": 0.0,
                    }
                input_chars = sum(len(t) for t in cur_texts)
                t0 = time.perf_counter()
                with dspy.context(lm=resolved_lm):
                    summary = summarizer(cur_texts)
                latency_s = time.perf_counter() - t0
                return summary, {
                    "cluster": i,
                    "n_texts": len(cur_texts),
                    "input_chars": input_chars,
                    "summary_chars": len(summary),
                    "latency_s": latency_s,
                }

            history_before = len(getattr(resolved_lm, "history", []) or [])
            summary_start = time.perf_counter()
            if self.verbose:
                print(
                    f"[iter {iteration}] starting summarization "
                    f"for {self.n_clusters} clusters"
                )
            if self.summary_workers == 1:
                cluster_range = range(self.n_clusters)
                cluster_range = tqdm(
                    cluster_range,
                    disable=not self.show_progress,
                    desc=f"Summarizing clusters (iter {iteration})",
                    leave=False,
                )
                summaries = [""] * self.n_clusters
                cluster_stats: list[dict[str, Any] | None] = [None] * self.n_clusters
                for i in cluster_range:
                    summary, stats = summarize_cluster_idx(i)
                    summaries[i] = summary
                    cluster_stats[i] = stats
            else:
                with ThreadPoolExecutor(max_workers=self.summary_workers) as executor:
                    futures = {
                        executor.submit(summarize_cluster_idx, i): i
                        for i in range(self.n_clusters)
                    }
                    summaries = [""] * self.n_clusters
                    cluster_stats = [None] * self.n_clusters
                    progress_bar = tqdm(
                        total=self.n_clusters,
                        disable=not self.show_progress,
                        desc=f"Summarizing clusters (iter {iteration})",
                        leave=False,
                    )
                    completed = 0
                    for future in as_completed(futures):
                        i = futures[future]
                        summary, stats = future.result()
                        summaries[i] = summary
                        cluster_stats[i] = stats
                        completed += 1
                        progress_bar.update(1)
                        if self.verbose and (completed % 10 == 0 or completed == self.n_clusters):
                            elapsed = time.perf_counter() - summary_start
                            rate = completed / max(elapsed, 1e-9)
                            print(
                                f"[iter {iteration}] summarization progress: "
                                f"{completed}/{self.n_clusters} clusters "
                                f"({rate:.2f} clusters/s)"
                            )
                    progress_bar.close()
            summary_elapsed = time.perf_counter() - summary_start
            summaries_evolution.append(summaries)
            if self.verbose:
                print(f"[iter {iteration}] generated {len(summaries)} summaries")
            if self.verbose or self.show_progress:
                summaries_per_sec = len(summaries) / max(summary_elapsed, 1e-9)
                history_after_entries = self._history_slice(resolved_lm, history_before)
                usage = self._sum_usage_from_history(history_after_entries)
                msg = (
                    f"[iter {iteration}] summary step: "
                    f"{summary_elapsed:.2f}s total, {summaries_per_sec:.2f} summaries/s"
                )
                if usage["total_tokens"] > 0:
                    tok_per_sec = usage["total_tokens"] / max(summary_elapsed, 1e-9)
                    msg += (
                        ", "
                        f"tokens in/out/total="
                        f"{usage['prompt_tokens']}/"
                        f"{usage['completion_tokens']}/"
                        f"{usage['total_tokens']}"
                        f", {tok_per_sec:.1f} tok/s"
                    )
                print(msg)
                stats_rows = [
                    s for s in cluster_stats if s is not None and int(s["n_texts"]) > 0
                ]
                if stats_rows:
                    latencies = np.asarray([float(s["latency_s"]) for s in stats_rows])
                    input_chars = np.asarray([int(s["input_chars"]) for s in stats_rows])
                    print(
                        f"[iter {iteration}] cluster latency s p50/p95/max="
                        f"{np.percentile(latencies, 50):.2f}/"
                        f"{np.percentile(latencies, 95):.2f}/"
                        f"{latencies.max():.2f}"
                    )
                    print(
                        f"[iter {iteration}] input chars p50/p95/max="
                        f"{np.percentile(input_chars, 50):.0f}/"
                        f"{np.percentile(input_chars, 95):.0f}/"
                        f"{input_chars.max():.0f}"
                    )
                    slowest = sorted(
                        stats_rows, key=lambda s: float(s["latency_s"]), reverse=True
                    )[:3]
                    for s in slowest:
                        print(
                            f"[iter {iteration}] slow cluster {s['cluster']}: "
                            f"latency={float(s['latency_s']):.2f}s, "
                            f"n_texts={int(s['n_texts'])}, "
                            f"input_chars={int(s['input_chars'])}, "
                            f"summary_chars={int(s['summary_chars'])}"
                        )

            summary_embeddings = self._get_embeddings_with_precomputed(
                summaries,
                embedding_fn=embedding_fn,
                name="summary_embeddings",
                precomputed=precomputed_summary_embeddings,
            )
            summary_embeddings = self._normalize_embeddings(
                summary_embeddings, "summary_embeddings"
            )
            if self.verbose:
                print(
                    f"[iter {iteration}] summary_embeddings shape={summary_embeddings.shape}"
                )

            centroid_shift = np.linalg.norm(centroids - summary_embeddings, axis=1).sum()
            if centroid_shift < self.tol:
                if self.verbose or self.show_progress:
                    print(
                        f"Converged after {iteration} iterations "
                        f"(centroid_shift={centroid_shift:.6f})."
                    )
                converged = True
                break

            centroids = summary_embeddings
            centroids_evolution.append(centroids.copy())

            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init=centroids,
                max_iter=max(1, self.max_iter // max(1, self.max_llm_iter + 1)),
                n_init=1,
                random_state=self.random_state,
            )
            labels = kmeans.fit_predict(doc_features)
            if self.verbose:
                print(f"[iter {iteration}] completed reclustering")

        self.labels_ = labels
        self.cluster_centers_ = kmeans.cluster_centers_
        self.summaries_ = summaries
        self.summary_embeddings_ = summary_embeddings
        self.summaries_evolution_ = summaries_evolution
        self.centroids_evolution_ = centroids_evolution
        self.converged_ = converged
        self.n_iter_ = n_iter
        self.n_features_in_ = doc_features.shape[1]
        self._embedding_fn_resolved = embedding_fn
        self._llm_resolved = resolved_lm
        return self

    def predict(
        self, X, precomputed_embeddings: np.ndarray | Mapping[str, np.ndarray] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, ["cluster_centers_", "_embedding_fn_resolved"])
        if not isinstance(X, list):
            raise TypeError("X must be a list[str].")
        features = self._get_embeddings_with_precomputed(
            X,
            embedding_fn=self._embedding_fn_resolved,
            name="predict_features",
            precomputed=precomputed_embeddings,
        )
        features = self._normalize_embeddings(features, "predict_features")
        return pairwise_distances_argmin(features, self.cluster_centers_)

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X, y=y).labels_
