"""Logit lens: project each layer's hidden state to token predictions.

The logit lens technique takes the intermediate hidden state at each layer
and projects it through the model's final RMSNorm and unembedding matrix
(the transposed embedding matrix, since weights are tied). This reveals
what token the model would predict if it stopped processing at that layer.

Reference: https://www.lesswrong.com/posts/AcKRB8wDds238WgfZ/interpreting-gpt-the-logit-lens
"""

import logging
from pathlib import Path

import numpy as np

from src.config import LOGIT_LENS_TOP_K, RMS_NORM_EPS

logger = logging.getLogger(__name__)


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS) -> np.ndarray:
    """Apply RMSNorm: x * weight / sqrt(mean(x^2) + eps).

    Args:
        x: Hidden states, shape [..., hidden_dim].
        weight: Norm weight, shape [hidden_dim].
        eps: Epsilon for numerical stability.
    """
    variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    x_normed = x.astype(np.float32) / np.sqrt(variance + eps)
    return x_normed * weight


def compute_logit_lens(
    residual_stream: np.ndarray,
    norm_weight: np.ndarray,
    embed_weight: np.ndarray,
    top_k: int = LOGIT_LENS_TOP_K,
) -> dict:
    """Compute logit lens predictions for each layer and position.

    For each layer l and position p:
      1. h = residual_stream[l, p, :]         # hidden state
      2. h_normed = rms_norm(h, norm_weight)   # apply final RMSNorm
      3. logits = h_normed @ embed_weight.T    # project to vocab
      4. probs = softmax(logits)               # get probabilities
      5. top_k tokens and their probabilities

    Args:
        residual_stream: [num_layers+1, seq_len, hidden_dim] float16.
        norm_weight: [hidden_dim] float32 — final RMSNorm weight.
        embed_weight: [vocab_size, hidden_dim] float32 — embedding matrix.
        top_k: Number of top predictions to save per position per layer.

    Returns:
        Dict with:
          'top_k_indices': [num_layers+1, seq_len, top_k] int32
          'top_k_probs': [num_layers+1, seq_len, top_k] float32
    """
    num_layers_plus_one, seq_len, hidden_dim = residual_stream.shape
    logger.info(
        "Computing logit lens: %d layers × %d positions",
        num_layers_plus_one,
        seq_len,
    )

    top_k_indices = np.zeros(
        (num_layers_plus_one, seq_len, top_k), dtype=np.int32
    )
    top_k_probs = np.zeros(
        (num_layers_plus_one, seq_len, top_k), dtype=np.float32
    )

    for layer_idx in range(num_layers_plus_one):
        # Apply RMSNorm
        h = residual_stream[layer_idx]  # [seq_len, hidden_dim]
        h_normed = _rms_norm(h, norm_weight)  # [seq_len, hidden_dim] float32

        # Project to vocab
        logits = h_normed @ embed_weight.T  # [seq_len, vocab_size]

        # Softmax (numerically stable)
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Top-k per position
        for pos in range(seq_len):
            top_indices = np.argpartition(probs[pos], -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(probs[pos][top_indices])[::-1]]
            top_k_indices[layer_idx, pos] = top_indices
            top_k_probs[layer_idx, pos] = probs[pos][top_indices]

    return {
        "top_k_indices": top_k_indices,
        "top_k_probs": top_k_probs,
    }


def save_logit_lens(prompt_dir: Path, logit_lens_data: dict) -> None:
    """Save logit lens results to disk."""
    np.savez(
        prompt_dir / "logit_lens_topk.npz",
        top_k_indices=logit_lens_data["top_k_indices"],
        top_k_probs=logit_lens_data["top_k_probs"],
    )
    logger.info("Saved logit lens data to %s", prompt_dir / "logit_lens_topk.npz")
