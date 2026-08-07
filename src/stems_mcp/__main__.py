import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="stems-mcp: audio stem separation")
    parser.add_argument("--mode", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    args, _ = parser.parse_known_args()

    # Strip --mode from sys.argv so transport runners don't crash on unknown flags
    cleaned = [a for a in sys.argv[1:] if not a.startswith("--mode") and a not in ("--host", "--port")]
    sys.argv = [sys.argv[0]] + cleaned

    from stems_mcp.server import BACKEND_PORT, app, mcp

    if args.mode == "http":
        import uvicorn

        port = args.port or BACKEND_PORT
        uvicorn.run(app, host=args.host, port=port, log_level="info")
    else:
        mcp.run_stdio_async()


if __name__ == "__main__":
    main()
