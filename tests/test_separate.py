"""Pipeline tests for _separate_audio using a fake ONNX session (no model download)."""

import numpy as np
import soundfile as sf

from stems_mcp.tools import stems as stems_mod


class _Meta:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Mimics the htdemucs graph contract: mix [1,2,N] -> stems [1,4,2,N]; stems sum to the mix."""

    def __init__(self):
        self.shapes = []

    def get_inputs(self):
        return [_Meta("mix")]

    def get_outputs(self):
        return [_Meta("stems")]

    def run(self, names, feed):
        mix = feed["mix"]
        self.shapes.append(mix.shape)
        assert mix.shape == (1, 2, stems_mod.SEGMENT_SAMPLES)
        shares = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)[:, None, None]
        return [(mix[0][None] * shares)[None]]


def _write_tone(path, sr, seconds, channels):
    t = np.arange(int(sr * seconds)) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    sf.write(path, data, sr)


def test_multichunk_resampled_stems_sum_back_to_input(tmp_path, monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(stems_mod, "_load_session", lambda: fake)
    src = tmp_path / "in.wav"
    _write_tone(src, 32000, 20.0, channels=1)  # mono, non-44.1k, > 2 segments

    result = stems_mod._separate_audio(str(src), str(tmp_path / "out"), stems=4)

    assert result["success"], result
    assert result["metadata"]["stems"] == ["drums", "bass", "other", "vocals"]
    assert result["metadata"]["chunks"] == len(fake.shapes) >= 3
    parts = {k: sf.read(v, always_2d=True)[0] for k, v in result["output_files"].items()}
    assert parts["vocals"].shape[1] == 2
    assert abs(len(parts["vocals"]) - 20 * stems_mod.SAMPLE_RATE) <= 2
    mixed = sum(parts.values())
    ref = np.sin(2 * np.pi * 220 * np.arange(len(mixed)) / stems_mod.SAMPLE_RATE) * 0.5
    assert np.max(np.abs(mixed[:, 0] - ref)) < 1e-2
    # Stem order follows the model: vocals carry the 0.4 share, drums the 0.1 share.
    assert np.std(parts["vocals"]) > np.std(parts["drums"]) * 3


def test_separate_vocals_writes_vocals_and_instrumental(tmp_path, monkeypatch):
    monkeypatch.setattr(stems_mod, "_load_session", lambda: FakeSession())
    src = tmp_path / "in.wav"
    _write_tone(src, 44100, 3.0, channels=2)

    result = stems_mod._separate_audio(str(src), str(tmp_path / "out"), stems=2)

    assert result["success"], result
    assert set(result["output_files"]) == {"vocals", "instrumental"}
