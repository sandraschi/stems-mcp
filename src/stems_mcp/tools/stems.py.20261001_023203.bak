import os
import threading
from pathlib import Path
from typing import Annotated

import numpy as np
import soundfile as sf

from stems_mcp.config import MODEL_CACHE, MODEL_FILENAME, MODEL_URL
from stems_mcp.server import mcp

# ---------------------------------------------------------------------------
# Lazy model globals (thread-safe, single-load)
# ---------------------------------------------------------------------------
_session = None
_session_lock = threading.Lock()


def _model_path() -> Path:
    return MODEL_CACHE / MODEL_FILENAME


def _ensure_model() -> str:
    """Download the ONNX model from HuggingFace if not cached. Returns local path."""
    model_path = _model_path()
    if model_path.exists():
        return str(model_path)

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    import httpx

    print(f"Downloading HTDemucs model from HuggingFace ({MODEL_URL}) ...")
    with httpx.stream("GET", MODEL_URL, follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(model_path, "wb") as f:
            for chunk in r.iter_bytes(8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(
                        f"\r  Downloading ... {downloaded / 1_000_000:.1f} / {total / 1_000_000:.1f} MB ({pct:.0f}%)",
                        end="",
                    )
                else:
                    print(f"\r  Downloading ... {downloaded / 1_000_000:.1f} MB", end="")
        print()

    return str(model_path)


def _get_session_info() -> dict:
    """Return info about the ONNX session without loading it."""
    import onnxruntime

    providers = onnxruntime.get_available_providers()
    return {
        "available_providers": providers,
        "cuda_available": "CUDAExecutionProvider" in providers,
        "model_cached": _model_path().exists(),
    }


def _load_session() -> tuple:
    """Load (or retrieve) the ONNX InferenceSession."""
    global _session
    if _session is not None:
        return _session

    with _session_lock:
        if _session is not None:
            return _session

        import onnxruntime

        model_path = _ensure_model()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = onnxruntime.get_available_providers()
        sess_providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        _session = onnxruntime.InferenceSession(
            model_path,
            providers=sess_providers,
        )
        return _session


def _separate_audio(
    input_path: str,
    output_dir: str,
    stems: int = 4,
) -> dict:
    """Run HTDemucs inference on a single audio file.

    Parameters
    ----------
    input_path : str
        Path to input audio file (any format soundfile can read).
    output_dir : str
        Directory to write stem WAV files.
    stems : int
        4 = vocals, drums, bass, other. 2 = vocals, other (for separate_vocals).

    Returns
    -------
    dict with success, message, output_files, metadata.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session = _load_session()

    input_name = session.get_inputs()[0].name
    sample_rate = 44100

    # Read audio
    try:
        audio, orig_sr = sf.read(input_path, always_2d=True)
    except Exception as e:
        return {"success": False, "error": f"Cannot read audio file: {e}"}

    # Mix to mono if multi-channel
    if audio.shape[1] > 1:
        audio = np.mean(audio, axis=1, keepdims=True)
    audio = audio.T  # shape: (1, T)

    # Resample to 44.1 kHz if needed
    if orig_sr != sample_rate:
        import scipy.signal

        ratio = sample_rate / orig_sr
        new_len = int(len(audio[0]) * ratio)
        audio_resampled = np.zeros((1, new_len), dtype=np.float32)
        audio_resampled[0] = scipy.signal.resample(audio[0], new_len)
        audio = audio_resampled

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    # Pad to multiple of model's chunk size (typically 264600 samples = 6s)
    chunk_size = 264600
    pad_len = chunk_size - (audio.shape[1] % chunk_size) if audio.shape[1] % chunk_size != 0 else 0
    if pad_len > 0:
        audio = np.pad(audio, ((0, 0), (0, pad_len)), mode="constant")

    # Run inference
    try:
        outputs = session.run(None, {input_name: audio.astype(np.float32)})
    except Exception as e:
        return {"success": False, "error": f"ONNX inference failed: {e}"}

    # outputs shape: (n_sources, 1, T) for 4-stems or (2, 1, T) for 2-stems
    n_sources = outputs[0].shape[0]
    stem_names = {
        4: ["vocals", "drums", "bass", "other"],
        2: ["vocals", "other"],
    }.get(n_sources, [f"stem_{i}" for i in range(n_sources)])

    # Trim padding
    trim_end = audio.shape[1] - pad_len if pad_len > 0 else audio.shape[1]

    # De-normalize and write each stem
    output_files = {}
    for i, name in enumerate(stem_names):
        stem_audio = outputs[0][i, 0, :trim_end]
        if peak > 0:
            stem_audio = stem_audio * peak
        out_path = out / f"{name}.wav"
        sf.write(str(out_path), stem_audio, sample_rate)
        output_files[name] = str(out_path)

    return {
        "success": True,
        "message": f"Separated {n_sources} stems to {out}",
        "output_files": output_files,
        "metadata": {
            "sample_rate": sample_rate,
            "input_file": input_path,
            "stems": stem_names,
        },
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def stems_separate(
    operation: Annotated[str, "Operation to perform: separate (4 stems), separate_vocals (2 stems), or status"],
    input_path: Annotated[str | None, "Path to input audio file (required for separate/separate_vocals)"] = None,
    output_dir: Annotated[
        str | None, "Directory to write stem WAV files (required for separate/separate_vocals)"
    ] = None,
) -> dict:
    """Separate audio into stems using the HTDemucs v4 model.

    [RATIONALE] Consolidated portmanteau for all stem operations to keep the tool list
    compact while exposing separate, separate_vocals, and status under one namespace.

    ## Return Format
    {"success": bool, "message": str, "output_files": dict[str, str], "metadata": dict}

    ## Examples
    stems_separate(operation="separate", input_path="song.mp3", output_dir="stems/")
    stems_separate(operation="separate_vocals", input_path="song.mp3", output_dir="stems/")
    stems_separate(operation="status")

    ## Errors
    - Input file not found or unreadable by soundfile
    - ONNX model download failure (network)
    - ONNX inference failure (memory, shape mismatch)
    """
    if operation == "status":
        info = _get_session_info()
        model_path = _model_path()
        return {
            "success": True,
            "message": "Model status",
            "model_cached": model_path.exists(),
            "model_path": str(model_path) if model_path.exists() else None,
            "providers": info["available_providers"],
            "cuda": info["cuda_available"],
        }

    if not input_path:
        return {"success": False, "error": "input_path is required for separate operations"}
    if not output_dir:
        return {"success": False, "error": "output_dir is required for separate operations"}
    if not os.path.isfile(input_path):
        return {"success": False, "error": f"Input file not found: {input_path}"}

    try:
        _load_session()
    except Exception as e:
        return {"success": False, "error": f"Failed to load ONNX model: {e}"}

    if operation == "separate":
        return _separate_audio(input_path, output_dir, stems=4)
    elif operation == "separate_vocals":
        result = _separate_audio(input_path, output_dir, stems=4)
        if result.get("success"):
            keep = {"vocals", "other"}
            result["output_files"] = {k: v for k, v in result["output_files"].items() if k in keep}
            result["message"] = f"Extracted vocals + other to {output_dir}"
        return result
    else:
        return {"success": False, "error": f"Unknown operation: {operation}"}
