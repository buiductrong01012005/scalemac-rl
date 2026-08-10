from __future__ import annotations

import hashlib
import io
import json
import os
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "PYTHONHASHSEED",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_mapping_sha256(mapping: Mapping[str, Any]) -> str:
    """Hash tensor content independent of torch checkpoint ZIP metadata."""
    digest = hashlib.sha256()
    for key in sorted(mapping):
        value = mapping[key]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def checkpoint_model_sha256(path: str | Path) -> str:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"checkpoint does not contain model_state_dict: {path}")
    return tensor_mapping_sha256(payload["model_state_dict"])


def numpy_global_rng_sha256() -> str:
    state = np.random.get_state()
    buffer = io.BytesIO()
    np.save(buffer, state[1], allow_pickle=False)
    payload = (
        str(state[0]).encode("utf-8")
        + buffer.getvalue()
        + str(state[2:]).encode("utf-8")
    )
    return _sha256_bytes(payload)


def torch_cpu_rng_sha256() -> str:
    return _sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes())


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def collect_runtime_fingerprint() -> dict[str, Any]:
    try:
        deterministic_debug_mode: str | int = torch.get_deterministic_debug_mode()
    except (AttributeError, RuntimeError):
        deterministic_debug_mode = "unavailable"
    try:
        interop_threads: int | str = torch.get_num_interop_threads()
    except RuntimeError:
        interop_threads = "unavailable"

    config_text = torch.__config__.show()
    numpy_config = io.StringIO()
    try:
        with np.printoptions():
            old_stdout = None
            import contextlib
            with contextlib.redirect_stdout(numpy_config):
                np.show_config()
    except Exception as exc:  # pragma: no cover - defensive metadata only
        numpy_config.write(f"unavailable: {exc}")

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": os.path.realpath(os.sys.executable),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_model": _cpu_model(),
        },
        "packages": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
        },
        "torch_runtime": {
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": interop_threads,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "deterministic_debug_mode": deterministic_debug_mode,
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "mkl_available": bool(torch.backends.mkl.is_available()),
            "openmp_available": bool(torch.backends.openmp.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "thread_environment": {name: os.environ.get(name) for name in _THREAD_ENV_VARS},
        "torch_config": config_text,
        "numpy_config": numpy_config.getvalue(),
    }


def source_fingerprint(root: str | Path) -> dict[str, str]:
    root = Path(root)
    relative_paths = (
        "scalemac_rl/scripts/train_ppo.py",
        "scalemac_rl/env.py",
        "scalemac_rl/models.py",
        "scalemac_rl/reward_study.py",
        "scalemac_rl/rl_evaluation.py",
    )
    output: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        if path.is_file():
            output[relative] = sha256_file(path)
    return output


def write_runtime_metadata(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
