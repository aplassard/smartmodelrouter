import httpx

from smartmodelrouter.benchmark import evaluate_model


def test_evaluate_model_end_to_end(monkeypatch):
    # Force local dataset usage
    def fail_get(*args, **kwargs):
        raise httpx.HTTPError("no network")

    monkeypatch.setattr("smartmodelrouter.benchmark.httpx.get", fail_get)

    responses = iter(["2", "wrong"])

    def fake_chat(prompt, model, max_tokens=1024, temperature=0):
        return {"message": next(responses), "usage": {}}

    monkeypatch.setattr("smartmodelrouter.benchmark.chat_completion", fake_chat)

    result = evaluate_model("test-model", "reasoning", 0, runs=2)
    assert result["model"] == "test-model"
    assert result["runs"] == 2
    assert result["correct"] == 1
    assert len(result["responses"]) == 2
    assert "Alice" in result["problem"]
