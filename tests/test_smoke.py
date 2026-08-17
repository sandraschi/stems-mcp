"""Minimal smoke test so fleet CI can run pytest."""

def test_import() -> None:
    import stems_mcp  # noqa: F401
