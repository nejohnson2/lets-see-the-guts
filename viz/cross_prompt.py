"""Visualization 6: Cross-Prompt Comparison.

Side-by-side comparison of key metrics across all prompts.

WHAT THIS REVEALS:
Different prompts cause the model to behave differently. By overlaying
metrics from multiple prompts, you can see:

- Do factual prompts ("The capital of France is") cause sharper attention
  than creative prompts ("Once upon a time")?
- Do certain prompts cause higher MLP activity (more computation)?
- Is the model more "confident" (higher prediction probability) for
  some types of prompts than others?
- Do all prompts show similar norm growth, or do some grow faster?

This comparison reveals the model's internal "effort" allocation —
how it adapts its processing strategy to different inputs.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz.common import (
    PROMPT_COLORS,
    PROMPT_MARKERS,
    add_explanation,
    layer_labels,
    load_activations,
    save_figure,
    get_prompt_dirs,
)

logger = logging.getLogger(__name__)


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float32))))


def generate(activations_dir: Path, output_dir: Path) -> None:
    """Generate cross-prompt comparison plots."""
    prompt_dirs = get_prompt_dirs(activations_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(prompt_dirs) < 2:
        logger.info("Need at least 2 prompts for cross-prompt comparison, skipping")
        return

    # Load all prompts
    all_data = []
    for d in prompt_dirs:
        data = load_activations(d)
        if "residual_stream" in data and "metadata" in data:
            all_data.append(data)

    if len(all_data) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # === Panel 1: Residual norm growth (last token) ===
    ax = axes[0, 0]
    for i, data in enumerate(all_data):
        residual = data["residual_stream"]
        norms = np.linalg.norm(residual.astype(np.float32), axis=-1)
        # Use the last token position
        last_tok_norms = norms[:, -1]
        color = PROMPT_COLORS[i % len(PROMPT_COLORS)]
        marker = PROMPT_MARKERS[i % len(PROMPT_MARKERS)]
        label = f"\"{data['metadata']['prompt']}\""
        ax.plot(
            last_tok_norms,
            marker=marker,
            markersize=3,
            linewidth=1.5,
            color=color,
            label=label,
            alpha=0.8,
        )

    num_layers = all_data[0]["residual_stream"].shape[0] - 1
    labels = layer_labels(num_layers)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("L2 Norm")
    ax.set_title("Residual Stream Norm (Last Token)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # === Panel 2: Average attention entropy per layer ===
    ax = axes[0, 1]
    for i, data in enumerate(all_data):
        if "attention_weights" not in data:
            continue
        attn = data["attention_weights"]  # [L, H, S, S]
        # Compute mean entropy across all heads and positions
        entropies = []
        for layer_idx in range(attn.shape[0]):
            layer_attn = np.clip(attn[layer_idx].astype(np.float32), 1e-10, 1.0)
            entropy = -np.sum(layer_attn * np.log2(layer_attn), axis=-1)
            entropies.append(entropy.mean())

        color = PROMPT_COLORS[i % len(PROMPT_COLORS)]
        marker = PROMPT_MARKERS[i % len(PROMPT_MARKERS)]
        ax.plot(
            entropies,
            marker=marker,
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.8,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Attention Entropy (bits)")
    ax.set_title("Attention Entropy per Layer")
    ax.grid(True, alpha=0.3)

    # === Panel 3: MLP sparsity per layer ===
    ax = axes[1, 0]
    for i, data in enumerate(all_data):
        if "mlp_gate_pre_act" not in data:
            continue
        gate = _silu(data["mlp_gate_pre_act"].astype(np.float32))
        sparsity = np.mean(np.abs(gate) < 0.01, axis=(1, 2)) * 100

        color = PROMPT_COLORS[i % len(PROMPT_COLORS)]
        marker = PROMPT_MARKERS[i % len(PROMPT_MARKERS)]
        ax.plot(
            sparsity,
            marker=marker,
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.8,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("% Features Near Zero")
    ax.set_title("MLP Gate Sparsity per Layer")
    ax.grid(True, alpha=0.3)

    # === Panel 4: Top-1 prediction confidence (last position) ===
    ax = axes[1, 1]
    for i, data in enumerate(all_data):
        if "logit_lens_top_k_probs" not in data:
            continue
        probs = data["logit_lens_top_k_probs"]  # [L+1, S, K]
        top1_last = probs[:, -1, 0]  # Top-1 prob at last position

        color = PROMPT_COLORS[i % len(PROMPT_COLORS)]
        marker = PROMPT_MARKERS[i % len(PROMPT_MARKERS)]
        ax.plot(
            top1_last,
            marker=marker,
            markersize=3,
            linewidth=1.5,
            color=color,
            alpha=0.8,
        )

    labels = layer_labels(num_layers)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-1 Probability")
    ax.set_title("Prediction Confidence (Last Position)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Prompt Comparison", fontsize=14)

    explanation = (
        "Comparing all prompts on four key metrics. TOP-LEFT: How hidden state magnitude grows "
        "(steeper = more layer contribution). TOP-RIGHT: How focused vs. distributed attention is. "
        "BOTTOM-LEFT: MLP selectivity (higher = more features suppressed). "
        "BOTTOM-RIGHT: How quickly the model becomes confident in its next-token prediction."
    )
    add_explanation(fig, explanation, y=0.01)

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    save_figure(fig, output_dir / "cross_prompt_comparison.png")
    logger.info("Saved cross-prompt comparison")
