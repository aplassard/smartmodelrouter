# smartmodelrouter

Utilities for routing chat completion requests to configured LLM providers.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# install runtime and dev dependencies
uv sync --dev
```

The repository contains a preconfigured `.env` with `OPENAI_API_KEY` and
`OPENAI_BASE_URL` for the [OpenRouter](https://openrouter.ai) API. **Do not
modify or commit changes to `.env`.**

## Testing

Run the full test suite with:

```bash
uv run pytest
```

Integration tests make live API calls and are marked with the `integration`
marker. To execute them explicitly, run:

```bash
uv run pytest -m integration
```
