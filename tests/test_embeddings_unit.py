import pytest

from smartmodelrouter.embeddings import embed_problem


def test_embed_problem_returns_embedding(monkeypatch):
    def fake_load(dataset):
        return [{"question": "What is 2+2?"}]

    monkeypatch.setattr("smartmodelrouter.embeddings._load_dataset", fake_load)

    class DummyVectorizer:
        def transform(self, texts):
            assert texts == ["What is 2+2?"]

            class Arr:
                def toarray(self):
                    return [[0.1, 0.2]]

            return Arr()

    monkeypatch.setattr(
        "smartmodelrouter.embeddings._get_vectorizer", lambda: DummyVectorizer()
    )

    result = embed_problem("emb-model", "math", 0)
    assert result["model"] == "emb-model"
    assert result["dataset"] == "math"
    assert result["index"] == 0
    assert result["embedding"] == [0.1, 0.2]
    assert result["response"] == {"embedding": [0.1, 0.2]}
