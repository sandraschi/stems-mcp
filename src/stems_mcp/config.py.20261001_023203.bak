import os
from pathlib import Path

BACKEND_PORT = int(os.environ.get("MCP_PORT", "11126"))
FRONTEND_PORT = int(os.environ.get("WEB_PORT", "11127"))
MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MODEL_CACHE = Path(os.environ.get("STEMS_MODEL_DIR", str(Path.home() / ".cache" / "stems-mcp")))
MODEL_URL = "https://huggingface.co/dhunstack/demucs-onnx/resolve/main/htdemucs.onnx"
MODEL_FILENAME = "htdemucs.onnx"
