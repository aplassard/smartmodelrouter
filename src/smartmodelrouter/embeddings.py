"""Utilities for embedding LiveBench prompts."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import HashingVectorizer

from .benchmark import _load_dataset


def _get_vectorizer() -> HashingVectorizer:
    """Return the HashingVectorizer used for embeddings."""

    # Use a small feature space so the embedding is inexpensive.
    return HashingVectorizer(n_features=128, alternate_sign=False)


def embed_problem(model: str, dataset: str, index: int) -> dict[str, Any]:
    """Return an embedding for a LiveBench problem.

    Parameters
    ----------
    model:
        Identifier for the embedding model. For the local HashingVectorizer the
        value is informational only but included in the result for consistency.
    dataset:
        One of ``"reasoning"``, ``"math"`` or ``"coding"``.
    index:
        Zero-based index of the problem in the dataset.

    Returns
    -------
    dict
        Mapping with keys ``model``, ``dataset``, ``index``, ``response`` and
        ``embedding``.
    """

    questions = _load_dataset(dataset)
    try:
        entry = questions[index]
    except IndexError as exc:  # pragma: no cover - invalid test usage
        raise IndexError(
            f"Problem index {index} out of range for dataset '{dataset}'"
        ) from exc

    prompt = entry["question"]

    vectorizer = _get_vectorizer()
    embedding_arr = vectorizer.transform([prompt]).toarray()[0]
    embedding = list(embedding_arr)
    response = {"embedding": embedding}

    return {
        "model": model,
        "dataset": dataset,
        "index": index,
        "response": response,
        "embedding": embedding,
    }


__all__ = ["embed_problem"]

