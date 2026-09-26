"""Reusable offline-ranking evaluation utilities.

The module deliberately has no database or web dependencies so the benchmark
can be reproduced from the prepared catalog and checked-in FAISS indexes.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


class BM25Index:
    """Small dependency-free BM25 implementation for the catalog benchmark."""

    def __init__(
        self,
        documents: Sequence[str],
        query_vocabulary: set[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.document_count = len(documents)
        self.k1 = k1
        self.b = b
        self.document_lengths = np.zeros(self.document_count, dtype="float32")
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for document_id, document in enumerate(documents):
            tokens = tokenize(document)
            self.document_lengths[document_id] = len(tokens)
            for term, frequency in Counter(tokens).items():
                if query_vocabulary is None or term in query_vocabulary:
                    postings[term].append((document_id, frequency))

        self.average_document_length = float(self.document_lengths.mean()) or 1.0
        self.postings = {
            term: (
                np.fromiter((item[0] for item in values), dtype="int64"),
                np.fromiter((item[1] for item in values), dtype="float32"),
            )
            for term, values in postings.items()
        }

    def search(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros(self.document_count, dtype="float32")
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if posting is None:
                continue
            document_ids, frequencies = posting
            document_frequency = len(document_ids)
            inverse_document_frequency = math.log(
                1.0
                + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            lengths = self.document_lengths[document_ids]
            denominator = frequencies + self.k1 * (
                1.0 - self.b + self.b * lengths / self.average_document_length
            )
            scores[document_ids] += (
                inverse_document_frequency
                * frequencies
                * (self.k1 + 1.0)
                / denominator
            )

        limit = min(top_k, self.document_count)
        ranking = np.argsort(-scores, kind="stable")[:limit]
        return scores[ranking], ranking


@dataclass(frozen=True)
class RankingMetrics:
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float
    reciprocal_rank: float
    hits: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def ranking_metrics(
    ranked_ids: Iterable[int], relevant_ids: set[int], k: int
) -> RankingMetrics:
    ranked = list(ranked_ids)[:k]
    relevance = [1 if item in relevant_ids else 0 for item in ranked]
    hits = sum(relevance)
    dcg = sum(value / math.log2(position + 2) for position, value in enumerate(relevance))
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    first_hit = next((position for position, value in enumerate(relevance, 1) if value), None)
    return RankingMetrics(
        precision_at_k=hits / k if k else 0.0,
        recall_at_k=hits / len(relevant_ids) if relevant_ids else 0.0,
        ndcg_at_k=dcg / idcg if idcg else 0.0,
        hit_rate_at_k=1.0 if hits else 0.0,
        reciprocal_rank=1.0 / first_hit if first_hit else 0.0,
        hits=hits,
    )


def unit_interval(cosine_scores: np.ndarray) -> np.ndarray:
    return np.clip((cosine_scores + 1.0) / 2.0, 0.0, 1.0)


def normalized_mean(vectors: np.ndarray) -> np.ndarray:
    mean = np.asarray(vectors, dtype="float32").mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm else mean


def bootstrap_mean_interval(
    values: Sequence[float], random: np.random.Generator, samples: int = 1_000
) -> tuple[float, float]:
    array = np.asarray(values, dtype="float64")
    if array.size == 0:
        return 0.0, 0.0
    draws = random.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
