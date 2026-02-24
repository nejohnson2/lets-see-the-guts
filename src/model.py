"""Model loading with eager attention for activation capture."""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_ID

logger = logging.getLogger(__name__)


def load_model(
    device: torch.device, dtype: torch.dtype
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load SmolLM2-1.7B with eager attention for weight capture.

    Uses attn_implementation="eager" because SDPA and Flash Attention
    do not return attention weights. Since our prompts are short (<128
    tokens), the performance penalty is negligible.
    """
    logger.info("Loading tokenizer for %s...", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    logger.info("Loading model %s (dtype=%s, device=%s)...", MODEL_ID, dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer
