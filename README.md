# MightyCode

A terminal-native AI coding agent built with Python, FastAPI, and Textual.

## Project Structure

```
mightycode/
├── pyproject.toml              # Workspace root
├── packages/
│   ├── shared/                 # Pydantic models & contracts
│   │   └── src/mightycode_shared/
│   ├── cli/                    # Textual terminal UI
│   │   └── src/mightycode_cli/
│   └── server/                 # FastAPI backend
│       └── src/mightycode_server/
└── .github/workflows/ci.yml   # CI pipeline
```

## Quick Start

```bash
# Install all workspace packages
uv sync --all-packages

# Run the CLI
uv run mightycode

# Run the server
uv run uvicorn mightycode_server.main:app

# Run tests
uv run pytest -v

# Lint
uv run ruff check .
uv run pyright packages/
```

## Phase 1

Scaffold with workspace wiring, shared Pydantic contracts, stub CLI and
server entry points, and CI pipeline.
