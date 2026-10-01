import hashlib
import os
import sys
import threading
from pathlib import Path
from typing import Annotated

import numpy as np
import soundfile as sf

from stems_mcp.config import MODEL_CACHE, MODEL_FILENAME, MODEL_SHA256, MODEL_SIZE, MODEL_URL
from stems_mcp.server import mcp

SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 343980  # 7.8 s, fixed by the exported graph
SOURCES = ("drums", "bass", "other", "vocals")  # model output order

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

    part_path = model_path.with_suffix(model_path.suffix + ".part")
    digest = hashlib.sha256()
    print(f"Downloading HTDemucs model from HuggingFace ({MODEL_URL}) ...")
    try:
        with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            downloaded = 0
            with open(part_path, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    print(f"\r  Downloading ... {downloaded / 1_000_000:.1f} / {MODEL_SIZE / 1_000_000:.1f} MB", end="")
            print()
        if downloaded != MODEL_SIZE or digest.hexdigest() != MODEL_SHA256:
            raise RuntimeError(
                f"Model integrity check failed (got {downloaded} bytes, sha256 {digest.hexdigest()}); "
                f"expected {MODEL_SIZE} bytes, sha256 {MODEL_SHA256}"
            )
        part_path.replace(model_path)
    finally:
        part_path.unlink(missing_ok=True)

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

        session = onnxruntime.InferenceSession(
            model_path,
            providers=sess_providers,
        )
        if sess_providers[0] != "CPUExecutionProvider":
            # A provider can be listed yet fail at run time (e.g. cuDNN DLL missing); probe once, else use CPU.
            try:
                session.run(None, {session.get_inputs()[0].name: np.zeros((1, 2, SEGMENT_SAMPLES), np.float32)})
            except Exception as e:
                print(
                    f"{sess_providers[0]} unusable ({str(e).splitlines()[0][:200]}); falling back to CPU",
                    file=sys.stderr,
                )
                session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        _session = session
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
        4 = drums, bass, other, vocals. 2 = vocals + instrumental (drums + bass + other summed).

    Returns
    -------
    dict with success, message, output_files, metadata.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session = _load_session()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Read audio as (channels, samples) float32, forced to stereo at 44.1 kHz
    try:
        audio, orig_sr = sf.read(input_path, dtype="float32", always_2d=True)
    except Exception as e:
        return {"success": False, "error": f"Cannot read audio file: {e}"}
    audio = audio.T
    audio = np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio[:2]

    if orig_sr != SAMPLE_RATE:
        from math import gcd

        import scipy.signal

        g = gcd(SAMPLE_RATE, int(orig_sr))
        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE // g, int(orig_sr) // g, axis=1).astype(np.float32)

    # Overlap-add over fixed 7.8 s segments (25% overlap, linear crossfade)
    total = audio.shape[1]
    overlap = SEGMENT_SAMPLES // 4
    stride = SEGMENT_SAMPLES - overlap
    n_chunks = max(1, (total + stride - 1) // stride)
    window = np.ones(SEGMENT_SAMPLES, dtype=np.float32)
    fade = np.linspace(0, 1, overlap, dtype=np.float32)
    window[:overlap] = fade
    window[-overlap:] = fade[::-1]

    acc = np.zeros((len(SOURCES), 2, total), dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    try:
        for i in range(n_chunks):
            start = i * stride
            end = min(start + SEGMENT_SAMPLES, total)
            chunk = audio[:, start:end]
            if chunk.shape[1] < SEGMENT_SAMPLES:
                chunk = np.pad(chunk, ((0, 0), (0, SEGMENT_SAMPLES - chunk.shape[1])))
            result = session.run([output_name], {input_name: chunk[np.newaxis].astype(np.float32)})[0][0]
            w = window[: end - start]
            acc[:, :, start:end] += result[:, :, : end - start] * w
            weight[start:end] += w
    except Exception as e:
        return {"success": False, "error": f"ONNX inference failed: {e}"}
    acc /= np.maximum(weight, 1e-8)

    named = dict(zip(SOURCES, acc, strict=True))
    if stems == 2:
        named = {"vocals": named["vocals"], "instrumental": named["drums"] + named["bass"] + named["other"]}

    output_files = {}
    for name, data in named.items():
        out_path = out / f"{name}.wav"
        sf.write(str(out_path), data.T, SAMPLE_RATE, subtype="PCM_24")
        output_files[name] = str(out_path)

    return {
        "success": True,
        "message": f"Separated {len(named)} stems to {out}",
        "output_files": output_files,
        "metadata": {
            "sample_rate": SAMPLE_RATE,
            "input_file": input_path,
            "stems": list(named),
            "chunks": n_chunks,
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
    """Separate audio into stems using the HTDemucs v4 model (stereo 44.1 kHz WAV output).

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
        return _separate_audio(input_path, output_dir, stems=2)
    else:
        return {"success": False, "error": f"Unknown operation: {operation}"}
