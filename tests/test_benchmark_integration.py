import os

import httpx
import pytest
from dotenv import load_dotenv

from smartmodelrouter.benchmark import evaluate_model


@pytest.mark.integration
def test_evaluate_model_integration(monkeypatch):
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY must be set for integration test")

    def fail_get(*_args, **_kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr("smartmodelrouter.benchmark.httpx.get", fail_get)

    result = evaluate_model("qwen/qwen3-30b-a3b", "math", 0, runs=1)
    assert result["model"] == "qwen/qwen3-30b-a3b"
    assert result["runs"] == 1
    assert len(result["responses"]) == 1
    assert result["problem"] == "What is 15 divided by 3?"
    assert isinstance(result["responses"][0], str)
    assert result["responses"][0]
