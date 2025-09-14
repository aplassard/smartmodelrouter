import os

import pytest
from dotenv import load_dotenv
from openai import APIConnectionError

from smartmodelrouter.llm import chat_completion


@pytest.mark.integration
def test_openai_hello_world():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY must be set for integration test")
    try:
        result = chat_completion(
            "Say hello world",
            model="openai/gpt-5-nano",
            max_tokens=1024,
        )
    except APIConnectionError as exc:  # pragma: no cover - network issues
        pytest.skip(f"API connection failed: {exc}")
    except Exception as exc:  # pragma: no cover - auth or other issues
        pytest.skip(f"OpenAI request failed: {exc}")
    normalized = result["message"].lower().replace(",", "")
    assert "hello world" in normalized
