"""_load_session must fall back to CPU when a listed GPU provider cannot actually run."""

import onnxruntime
import pytest

from stems_mcp.tools import stems as stems_mod


class _Meta:
    name = "mix"


class _FakeSession:
    def __init__(self, path, providers):
        self.providers = providers

    def get_inputs(self):
        return [_Meta()]

    def run(self, names, feed):
        if self.providers[0] == "CUDAExecutionProvider":
            raise RuntimeError("cuDNN is unavailable: LoadLibrary failed for cudnn64_9.dll")
        return []


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(stems_mod, "_session", None)
    monkeypatch.setattr(stems_mod, "_ensure_model", lambda: "model.onnx")
    monkeypatch.setattr(onnxruntime, "InferenceSession", _FakeSession)
    monkeypatch.setattr(
        onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )


def test_broken_cuda_falls_back_to_cpu(patched, capsys):
    session = stems_mod._load_session()
    assert session.providers == ["CPUExecutionProvider"]
    assert "falling back to CPU" in capsys.readouterr().err
