import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP

from stems_mcp.config import BACKEND_PORT, FRONTEND_PORT

mcp = FastMCP(
    "stems-mcp",
    description="Audio stem separation using HTDemucs v4 via ONNX Runtime",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from stems_mcp.tools import register_all_tools
    register_all_tools()
    yield


app = FastAPI(title="stems-mcp", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{FRONTEND_PORT}",
        f"http://127.0.0.1:{FRONTEND_PORT}",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
    ],
    allow_origin_regex=r"https?://(?:[a-zA-Z0-9-]+\.ts\.net|.*?\.tail-[a-f0-9]+\.ts\.net|tauri\.localhost|localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|100\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d+)?$|^tauri://localhost$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_start_time = time.time()


@app.get("/api/health")
async def health():
    from stems_mcp.tools.stems import _get_session_info
    session_info = _get_session_info()
    return {
        "status": "ok",
        "server": "stems-mcp",
        "version": "0.1.0",
        "uptime_seconds": int(time.time() - _start_time),
        "tool_count": 3,
        "providers": {
            "onnx": session_info,
        },
    }


@app.get("/api/v1/diagnostics")
async def diagnostics():
    from stems_mcp.tools.stems import _get_session_info, _model_path
    model_path = _model_path()
    return {
        "status": "ok",
        "server": "stems-mcp",
        "version": "0.1.0",
        "uptime_seconds": int(time.time() - _start_time),
        "tool_count": 3,
        "tools": [
            {"name": "stems_separate", "operations": ["separate", "separate_vocals", "status"]},
        ],
        "system": {"windows": True},
        "model": {
            "cached": model_path.exists(),
            "path": str(model_path) if model_path.exists() else None,
        },
        "providers": _get_session_info(),
        "errors": [],
    }


mcp.mount("/api", app)
