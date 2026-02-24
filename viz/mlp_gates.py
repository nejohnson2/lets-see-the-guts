"""Visualization 4: MLP Gate Activation Analysis.

Shows what happens inside the feed-forward (MLP) layers, specifically
how the gating mechanism selects which features to activate.

WHAT THIS REVEALS:
Each transformer layer has two main components: attention and MLP.
The MLP in SmolLM2 uses a "gated" architecture (SwiGLU variant):

    MLP(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

The gate_proj decides WHICH of the 8192 features to activate, using
the SiLU activation function (smooth approximation of ReLU). Features
with gate values near zero are effectively suppressed.

This visualization captures the gate_proj output (before SiLU) and
applies SiLU to show the actual gating values.

Key observations:
- MLP activations are surprisingly SPARSE: most of the 8192 features
  are near zero for any given token. The model is very selective.
- Different tokens activate different feature sets.
- The sparsity pattern changes across layers, revealing that each
  layer computes different things.
- Some features may consistently activate for related concepts,
  hinting at the model's learned feature decomposition.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz.common import (
    PROMPT_COLORS,
    add_explanation,
    layer_labels,
    load_activations,
    save_figure,
    token_labels,
)

logger = logging.getLogger(__name__)


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float32))))


def generate(activations_dir: Path, output_dir: Path) -> None:
    """Generate MLP gate analysis plots."""
    from viz.common import get_prompt_dirs

    prompt_dirs = get_prompt_dirs(activations_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_dir in prompt_dirs:
        data = load_activations(prompt_dir)
        if "mlp_gate_pre_act" not in data:
            logger.warning("No mlp_gate_pre_act in %s, skipping", prompt_dir)
            continue

        gate_raw = data["mlp_gate_pre_act"]  # [num_layers, seq_len, 8192]
        meta = data["metadata"]
        tokens = token_labels(meta["token_strings"])
        num_layers, seq_len, intermediate_size = gate_raw.shape

        # Apply SiLU to get actual gate values
        gate_activated = _silu(gate_raw.astype(np.float32))

        # === Figure 1: Sparsity Analysis ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Fraction of near-zero activations per layer
        threshold = 0.01
        sparsity = np.mean(np.abs(gate_activated) < threshold, axis=(1, 2))

        ax = axes[0]
        bars = ax.bar(range(num_layers), sparsity * 100, color="#2196F3", alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("% of Gate Values Near Zero")
        ax.set_title("MLP Gate Sparsity by Layer")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate mean sparsity
        mean_sparsity = sparsity.mean() * 100
        ax.axhline(mean_sparsity, color="red", linestyle="--", alpha=0.5)
        ax.text(
            num_layers - 1,
            mean_sparsity + 2,
            f"Mean: {mean_sparsity:.1f}%",
            ha="right",
            color="red",
            fontsize=9,
        )

        # Right: Average gate magnitude per layer per token
        avg_magnitude = np.mean(np.abs(gate_activated), axis=-1)  # [num_layers, seq_len]

        ax = axes[1]
        im = ax.imshow(
            avg_magnitude.T,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Token")
        ax.set_yticks(np.arange(seq_len))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_title("Mean |Gate Activation| per Layer × Token")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Mean |activation|")

        fig.suptitle(
            f"MLP Gate Analysis — \"{meta['prompt']}\"",
            fontsize=13,
        )

        explanation = (
            "LEFT: Sparsity — what fraction of the 8,192 MLP features are near zero (suppressed) "
            "at each layer. High sparsity means the MLP is very selective. "
            "RIGHT: Average gate magnitude per token per layer — brighter cells mean that token "
            "triggers stronger MLP activity at that layer."
        )
        add_explanation(fig, explanation, y=0.01)

        fig.tight_layout(rect=[0, 0.07, 1, 0.93])
        save_figure(fig, output_dir / f"{meta['prompt_id']}_mlp_gates.png")

        # === Figure 2: Top Activated Features ===
        fig2, ax2 = plt.subplots(figsize=(14, 6))

        # For the last token position, show top 50 features across layers
        last_tok_gates = gate_activated[:, -1, :]  # [num_layers, 8192]

        # Find features with highest average activation across layers
        mean_activation = np.mean(np.abs(last_tok_gates), axis=0)
        top_features = np.argsort(mean_activation)[-50:][::-1]

        im2 = ax2.imshow(
            last_tok_gates[:, top_features].T,
            aspect="auto",
            cmap="RdBu_r",
            interpolation="nearest",
        )
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Feature Index (top 50 by mean activation)")
        ax2.set_yticks(np.arange(0, 50, 5))
        ax2.set_yticklabels([str(top_features[i]) for i in range(0, 50, 5)], fontsize=8)
        ax2.set_title(
            f"Top 50 MLP Features for Last Token — \"{meta['prompt']}\"\n"
            f"(Last token: \"{tokens[-1]}\")"
        )
        fig2.colorbar(im2, ax=ax2, label="Gate activation (post-SiLU)")

        explanation2 = (
            "Each row is one of the top 50 most-activated MLP features for the final token position. "
            "Blue = negative (suppressed), Red = positive (activated). Each column is a layer. "
            "This reveals which features the model uses at each processing stage to build its prediction."
        )
        add_explanation(fig2, explanation2)

        fig2.tight_layout(rect=[0, 0.08, 1, 1])
        save_figure(fig2, output_dir / f"{meta['prompt_id']}_mlp_top_features.png")

        logger.info("Saved MLP gate analysis for '%s'", meta["prompt"])
