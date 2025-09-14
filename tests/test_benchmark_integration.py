import os

import pytest
from dotenv import load_dotenv

from smartmodelrouter.benchmark import evaluate_model


@pytest.mark.integration
def test_evaluate_model_integration(monkeypatch):
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY must be set for integration test")

    sample = [{"question": "What is 2+2?", "answer": "4"}]
    monkeypatch.setattr("smartmodelrouter.benchmark._load_dataset", lambda _dataset: sample)

    result = evaluate_model("qwen/qwen3-30b-a3b", "math", 0, runs=1)
    assert result["model"] == "qwen/qwen3-30b-a3b"
    assert result["runs"] == 1
    assert len(result["responses"]) == 1
    assert result["problem"] == "What is 2+2?"
    assert isinstance(result["responses"][0], str)
    assert result["responses"][0]
