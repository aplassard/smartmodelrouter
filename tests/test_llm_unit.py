import os
from pathlib import Path

import pytest
from openai import APIConnectionError
import httpx

from smartmodelrouter.llm import _ensure_env, chat_completion


def test_ensure_env_loads_dotenv(monkeypatch, tmp_path: Path) -> None:
    """_ensure_env loads credentials from a local .env when variables are missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=dummy-key\nOPENAI_BASE_URL=https://example.com\n"
    )
    monkeypatch.setattr("dotenv.main.find_dotenv", lambda *args, **kwargs: str(env_file))
    _ensure_env()
    assert os.environ["OPENAI_API_KEY"] == "dummy-key"
    assert os.environ["OPENAI_BASE_URL"] == "https://example.com"


def test_chat_completion_parses_response(monkeypatch) -> None:
    """chat_completion returns the message and usage from the client."""

    captured: dict | None = None

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.chat = self.Chat()

        class Chat:
            def __init__(self):
                self.completions = self.Completions()

            class Completions:
                def create(self, **kwargs):
                    nonlocal captured
                    captured = kwargs

                    class Msg:
                        content = "hi"

                    class Choice:
                        message = Msg()

                    class Completion:
                        choices = [Choice()]
                        usage = {"prompt_tokens": 1}

                    return Completion()

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com")
    monkeypatch.setattr("smartmodelrouter.llm.OpenAI", DummyClient)

    result = chat_completion("hi", model="openai/gpt-5-nano", max_tokens=1024)
    assert result["message"] == "hi"
    assert result["usage"] == {"prompt_tokens": 1}
    assert "response" in result
    assert captured is not None
    assert captured["extra_headers"]["HTTP-Referer"] == "https://github.com/aplassard/smartmodelrouter"
    assert captured["extra_headers"]["X-Title"] == "smartmodelrouter"


def test_chat_completion_rejects_invalid_base_url(monkeypatch) -> None:
    """chat_completion raises when OPENAI_BASE_URL lacks a scheme."""

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "example.com")
    with pytest.raises(RuntimeError, match="Invalid OPENAI_BASE_URL"):
        chat_completion("hi", model="openai/gpt-5-nano", max_tokens=1024)


def test_chat_completion_missing_choices_retries(monkeypatch) -> None:
    """chat_completion retries when no choices are returned."""

    calls = {"count": 0}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.chat = self.Chat()

        class Chat:
            def __init__(self):
                self.completions = self.Completions()

            class Completions:
                def create(self, **kwargs):
                    calls["count"] += 1
                    class Completion:
                        choices = None
                        usage = {}

                    return Completion()

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com")
    monkeypatch.setattr("smartmodelrouter.llm.OpenAI", DummyClient)

    with pytest.raises(RuntimeError, match="Completion returned no choices"):
        chat_completion("hi", model="openai/gpt-5-nano", max_tokens=1024)
    assert calls["count"] == 3


def test_chat_completion_api_error_retries(monkeypatch) -> None:
    """chat_completion retries on APIConnectionError from the client."""

    calls = {"count": 0}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.chat = self.Chat()

        class Chat:
            def __init__(self):
                self.completions = self.Completions()

            class Completions:
                def create(self, **kwargs):
                    calls["count"] += 1
                    if calls["count"] < 3:
                        raise APIConnectionError(request=httpx.Request("POST", "https://example.com"))

                    class Msg:
                        content = "hi"

                    class Choice:
                        message = Msg()

                    class Completion:
                        choices = [Choice()]
                        usage = {}

                    return Completion()

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com")
    monkeypatch.setattr("smartmodelrouter.llm.OpenAI", DummyClient)

    result = chat_completion("hi", model="openai/gpt-5-nano", max_tokens=1024)
    assert result["message"] == "hi"
    assert calls["count"] == 3
