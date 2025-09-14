"""Utilities for evaluating models on LiveBench problems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from .llm import chat_completion

_DATA_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/LiveBench/LiveBench/main/data/{dataset}.json"
)


def _load_dataset(dataset: str) -> list[dict[str, Any]]:
    """Return the questions for ``dataset``.

    Attempts to download the dataset from the official LiveBench repository. If
    the network request fails, falls back to reading a bundled local copy.
    """
    url = _DATA_URL_TEMPLATE.format(dataset=dataset)
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        local_path = Path(__file__).parent / "data" / f"{dataset}.json"
        if not local_path.exists():  # pragma: no cover - developer error
            raise RuntimeError(f"Dataset '{dataset}' not available")
        with local_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)


def evaluate_model(
    model: str,
    dataset: str,
    index: int,
    runs: int = 10,
) -> dict[str, Any]:
    """Run ``runs`` evaluations of ``model`` on a LiveBench problem.

    Parameters
    ----------
    model:
        The model identifier to pass to the OpenAI-compatible endpoint.
    dataset:
        One of ``"reasoning"``, ``"math"`` or ``"coding"``.
    index:
        Zero-based index of the problem in the dataset.
    runs:
        Number of times to query the model. Defaults to 10.

    Returns
    -------
    dict
        Mapping with keys ``model``, ``runs``, ``correct``, ``responses`` and
        ``problem``.
    """
    questions = _load_dataset(dataset)
    try:
        entry = questions[index]
    except IndexError as exc:  # pragma: no cover - invalid test usage
        raise IndexError(
            f"Problem index {index} out of range for dataset '{dataset}'"
        ) from exc

    prompt = entry["question"]
    answer = entry.get("answer")

    responses: list[str] = []
    correct = 0
    for _ in range(runs):
        result = chat_completion(prompt, model=model, max_tokens=1024, temperature=0)
        message = result["message"].strip()
        responses.append(message)
        if answer is not None and message.strip().lower() == str(answer).strip().lower():
            correct += 1

    return {
        "model": model,
        "runs": runs,
        "correct": correct,
        "responses": responses,
        "problem": prompt,
    }

__all__ = ["evaluate_model"]
