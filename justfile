lint:
    uv run ruff check src/

serve:
    uv run python -m stems_mcp --mode http

serve-http:
    uv run python -m stems_mcp --mode http

test:
    uv run pytest -x -q
