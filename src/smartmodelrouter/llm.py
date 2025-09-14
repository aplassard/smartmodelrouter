"""OpenAI-compatible LLM client utilities."""

from __future__ import annotations

import os
import time
from json import JSONDecodeError
from urllib.parse import urlparse

import atexit
import threading

import httpx
from openai import APIConnectionError, OpenAI

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-5-nano")


def _ensure_env() -> None:
    """Load API credentials from ``.env`` when missing.

    ``OPENAI_API_KEY`` must be provided; ``OPENAI_BASE_URL`` is optional and
    falls back to the OpenAI default endpoint when absent.
    """
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        from dotenv import load_dotenv

        load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")


_client_lock = threading.Lock()
_client: OpenAI | None = None
_client_params: tuple[str, str | None] | None = None
_client_ctor: type[OpenAI] | None = None


def _get_client() -> OpenAI:
    """Return a shared OpenAI client instance."""
    global _client, _client_params, _client_ctor
    current_params = (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"))
    current_ctor = OpenAI
    if (
        _client is None
        or _client_params != current_params
        or _client_ctor is not current_ctor
    ):
        with _client_lock:
            current_params = (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"))
            current_ctor = OpenAI
            if (
                _client is None
                or _client_params != current_params
                or _client_ctor is not current_ctor
            ):
                _ensure_env()
                api_key = os.environ["OPENAI_API_KEY"]
                base_url = os.getenv("OPENAI_BASE_URL")
                client_kwargs = {"api_key": api_key}
                if base_url:
                    parsed = urlparse(base_url)
                    if not parsed.scheme or not parsed.netloc:
                        raise RuntimeError(
                            f"Invalid OPENAI_BASE_URL: {base_url!r}. Include scheme, e.g. 'https://api.openai.com/v1'."
                        )
                    client_kwargs["base_url"] = base_url
                if _client is not None and hasattr(_client, "close"):
                    _client.close()
                _client = current_ctor(**client_kwargs)
                _client_params = (api_key, base_url)
                _client_ctor = current_ctor
    return _client


def _close_client() -> None:
    """Close the shared OpenAI client on interpreter shutdown."""
    global _client
    if _client is not None:
        if hasattr(_client, "close"):
            _client.close()
        _client = None


atexit.register(_close_client)


def chat_completion(
    prompt: str,
    model: str | None = None,
    max_tokens: int = 10_240,
    temperature: float = 0.7,
) -> dict:
    """Return the assistant message and token usage details.

    The returned dictionary contains the assistant ``message`` along with ``usage``
    statistics (prompt, cache, reasoning and completion tokens) and ``cost`` for
    each token type when pricing information is available for ``model``. The
    ``temperature`` controls sampling diversity and defaults to ``0.7``.
    """
    client = _get_client()
    target_model = model or MODEL_NAME
    # ``openai`` occasionally returns malformed JSON or encounters transient
    # network issues.  These manifest as ``JSONDecodeError`` or ``httpx``
    # exceptions bubbling out of ``client.chat.completions.create``.  Instead of
    # failing immediately, attempt a few simple retries with exponential
    # backoff.  If all retries fail, surface a more helpful ``RuntimeError`` so
    # callers don't see an opaque JSON decoding stack trace.
    completion = None
    message_content = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"max_output_tokens": max_tokens},
                extra_headers={
                    "HTTP-Referer": "https://github.com/aplassard/smartmodelrouter",
                    "X-Title": "smartmodelrouter",
                },
            )
            if not getattr(completion, "choices", None):
                if attempt == 2:
                    raise RuntimeError("Completion returned no choices")
                time.sleep(2**attempt)
                continue
            first = completion.choices[0]
            message_content = getattr(getattr(first, "message", None), "content", None)
            if message_content is None:
                if attempt == 2:
                    raise RuntimeError("Completion returned no message content")
                time.sleep(2**attempt)
                continue
            break
        except (JSONDecodeError, httpx.HTTPError, APIConnectionError) as exc:  # pragma: no cover - network
            if attempt == 2:
                raise RuntimeError("Failed to retrieve completion") from exc
            time.sleep(2**attempt)
    assert completion is not None and message_content is not None  # for type checkers
    usage = getattr(completion, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return {
        "message": message_content,
        "usage": usage,
        "response": completion,
    }
