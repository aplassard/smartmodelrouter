import httpx
import pytest

from smartmodelrouter.embeddings import embed_problem


@pytest.mark.integration
def test_embed_problem_integration(monkeypatch):
    def fail_get(*_args, **_kwargs):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr("smartmodelrouter.benchmark.httpx.get", fail_get)

    result = embed_problem("hashing-embed", "math", 0)

    assert result["model"] == "hashing-embed"
    assert result["dataset"] == "math"
    assert result["index"] == 0
    assert isinstance(result["embedding"], list)
    assert result["embedding"]
    assert result["response"] == {"embedding": result["embedding"]}
