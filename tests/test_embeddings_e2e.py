import httpx

from smartmodelrouter.embeddings import embed_problem


def test_embed_problem_end_to_end(monkeypatch):
    def fail_get(*args, **kwargs):
        raise httpx.HTTPError("no network")

    monkeypatch.setattr("smartmodelrouter.benchmark.httpx.get", fail_get)

    captured = {}

    class DummyVectorizer:
        def transform(self, texts):
            captured["input"] = texts[0]

            class Arr:
                def toarray(self):
                    return [[0.5, 0.5]]

            return Arr()

    monkeypatch.setattr(
        "smartmodelrouter.embeddings._get_vectorizer", lambda: DummyVectorizer()
    )

    result = embed_problem("test-embed", "reasoning", 0)
    assert result["model"] == "test-embed"
    assert result["dataset"] == "reasoning"
    assert result["index"] == 0
    assert result["embedding"] == [0.5, 0.5]
    assert result["response"] == {"embedding": [0.5, 0.5]}
    assert captured["input"].startswith("If Alice")
