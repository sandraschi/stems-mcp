import os
from pathlib import Path

BACKEND_PORT = int(os.environ.get("MCP_PORT", "11126"))
FRONTEND_PORT = int(os.environ.get("WEB_PORT", "11127"))
MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MODEL_CACHE = Path(os.environ.get("STEMS_MODEL_DIR", str(Path.home() / ".cache" / "stems-mcp")))
# Single-file 4-stem htdemucs export (MIT) from the StemSplit `demucs-onnx` project.
# Contract: input "mix" [1, 2, 343980] float32 @ 44.1 kHz -> output "stems" [1, 4, 2, 343980]
# in order drums, bass, other, vocals. Pinned by size + SHA-256 (the HF LFS oid).
MODEL_URL = "https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs.onnx"
MODEL_FILENAME = "htdemucs.onnx"
MODEL_SIZE = 316446953
MODEL_SHA256 = "68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74"
