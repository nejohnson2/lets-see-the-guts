"""Device auto-detection for MPS/CUDA/CPU."""

import logging

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


def get_model_dtype(device: torch.device) -> torch.dtype:
    """Return appropriate dtype for model loading.

    MPS does not support bfloat16, so we use float32 there.
    CUDA uses bfloat16 for efficiency.
    """
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32
