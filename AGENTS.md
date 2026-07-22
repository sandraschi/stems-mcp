# stems-mcp

Audio stem separation server using HTDemucs v4 via ONNX Runtime.

## Quick start
```powershell
uv sync
uv run stems-mcp
```

## Ports
- Backend: 11126 (MCP HTTP / REST API)
- Frontend: 11127 (webapp)

## Commands
```powershell
uv run stems-mcp              # stdio (default)
uv run stems-mcp --mode http  # HTTP on MCP_PORT (default 11126)
```

## Model
Downloads htdemucs.onnx (~200MB) from HuggingFace on first use to ~/.cache/stems-mcp/.
