# stems-mcp

FastMCP 3.4+ server for audio stem separation using HTDemucs v4 via ONNX Runtime.

Separates audio into **vocals, drums, bass, and other** (4 stems) or **vocals + other** (2 stems).

## Requirements

- Python 3.12+
- CUDA-capable GPU recommended (RTX 4090 tested; runs on CPU but slower)
- ~200 MB free disk for model cache

## Quick start

```powershell
uv sync --extra gpu   # or --extra cpu (pick exactly one)
uv run stems-mcp
```

### HTTP mode

```powershell
uv run stems-mcp --mode http
# Server on http://127.0.0.1:11126
```

## Tools

| Tool | Operations | Description |
|------|-----------|-------------|
| `stems_separate` | `separate`, `separate_vocals`, `status` | Portmanteau for all stem operations |

## Ports

| Service | Port |
|---------|------|
| Backend (MCP HTTP + REST) | 11126 |
| Frontend (reserved) | 11127 |

## Model

Downloads `htdemucs.onnx` from HuggingFace on first use. Cached at `~/.cache/stems-mcp/` by default (override with `STEMS_MODEL_DIR`).
