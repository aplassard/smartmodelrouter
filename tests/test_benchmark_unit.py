import httpx
import pytest

from smartmodelrouter.benchmark import _load_dataset, evaluate_model


def test_load_dataset_local_fallback(monkeypatch):
    def fake_get(*args, **kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx, "get", fake_get)
    data = _load_dataset("math")
    assert data and data[0]["question"].startswith("What is 15")


def test_evaluate_model_counts_correct_responses(monkeypatch):
    def fake_load(dataset):
        return [{"question": "What is 2+2?", "answer": "4"}]

    monkeypatch.setattr("smartmodelrouter.benchmark._load_dataset", fake_load)

    responses = ["4", "3", "4"]
    calls = {"i": 0}

    def fake_chat(prompt, model, max_tokens=1024, temperature=0):
        msg = responses[calls["i"]]
        calls["i"] += 1
        return {"message": msg, "usage": {}}

    monkeypatch.setattr("smartmodelrouter.benchmark.chat_completion", fake_chat)

    result = evaluate_model("model", "math", 0, runs=3)
    assert result["model"] == "model"
    assert result["runs"] == 3
    assert result["correct"] == 2
    assert result["responses"] == responses
    assert result["problem"] == "What is 2+2?"
