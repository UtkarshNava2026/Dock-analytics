"""Resolve ``torch.device`` from config: CUDA when requested, no silent CPU fallback."""

from __future__ import annotations

import torch


def resolve_inference_device(requested: str, *, context: str = "model") -> torch.device:
    """Return ``torch.device`` for YAML ``device`` strings.

    If the config requests CUDA (string starts with ``cuda``, case-insensitive) but
    ``torch.cuda.is_available()`` is false, raise ``RuntimeError`` instead of using CPU.
    """
    raw = str(requested).strip()
    if not raw:
        raw = "cuda"
    lower = raw.lower()
    if lower == "cpu":
        return torch.device("cpu")
    if lower.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{context}: device is {requested!r} (CUDA) but torch.cuda.is_available() "
                "is False. Install a CUDA-capable PyTorch build or set device to 'cpu' in config."
            )
        return torch.device(raw)
    return torch.device(raw)
