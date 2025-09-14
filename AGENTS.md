# Developer Instructions

- Use [uv](https://github.com/astral-sh/uv) for dependency management and to run commands.
- Install project and development dependencies with `uv sync --dev`.
- Run the test suite with `uv run pytest`. Integration tests hit external APIs and are marked `integration`; run them with `uv run pytest -m integration` when the required environment variables are set.
- The `.env` file contains `OPENAI_API_KEY` and `OPENAI_BASE_URL` for local development. **Do not modify or commit changes to `.env`.**
- Keep the test suite passing before committing.
