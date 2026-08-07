set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

lint:
    uv run ruff check src/

serve:
    uv run python -m stems_mcp --mode http

serve-http:
    uv run python -m stems_mcp --mode http

test:
    uv run pytest -x -q

# Bootstrap: install dev deps + pre-commit hook
bootstrap:
    uv sync --group dev
    uv run pre-commit install
    Write-Host "Pre-commit hooks installed." -ForegroundColor Green