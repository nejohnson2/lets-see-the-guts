"""Visualization 1: Activation Magnitude Heatmap.

A 2D heatmap where each cell shows the L2 norm of the hidden state
at a specific (layer, token position) coordinate.

WHAT THIS REVEALS:
This gives you a bird's-eye view of how the model processes each token
across all layers simultaneously. Bright spots indicate tokens that the
model is "paying more attention to" at that layer — their representations
are being amplified or significantly modified.

Patterns to look for:
- Vertical bright stripes: A token that's important across many layers
  (often content words like nouns and verbs).
- Horizontal gradients: Norm generally increases layer by layer (expected
  due to residual connections adding to the stream).
- Dark spots in bright rows: A token whose representation was briefly
  suppressed at a certain layer, then recovered.
- Differences between prompts: Factual prompts may show different
  patterns than creative or simple prompts.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz.common import (
    add_explanation,
    layer_labels,
    load_activations,
    save_figure,
    token_labels,
)

logger = logging.getLogger(__name__)


def generate(activations_dir: Path, output_dir: Path) -> None:
    """Generate activation heatmaps for each prompt."""
    from viz.common import get_prompt_dirs

    prompt_dirs = get_prompt_dirs(activations_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_dir in prompt_dirs:
        data = load_activations(prompt_dir)
        if "residual_stream" not in data:
            continue

        residual = data["residual_stream"]  # [num_layers+1, seq_len, hidden_dim]
        meta = data["metadata"]
        tokens = token_labels(meta["token_strings"])
        num_layers_plus_one, seq_len, _ = residual.shape

        # Compute L2 norms: [num_layers+1, seq_len]
        norms = np.linalg.norm(residual.astype(np.float32), axis=-1)

        fig, ax = plt.subplots(figsize=(max(8, seq_len * 1.2), 10))

        im = ax.imshow(
            norms,
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
        )

        # Labels
        labels = layer_labels(num_layers_plus_one - 1, include_embed=True)
        ax.set_yticks(np.arange(num_layers_plus_one))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks(np.arange(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")
        ax.set_title(f"Activation Magnitude (L2 Norm) — \"{meta['prompt']}\"")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="L2 Norm")

        explanation = (
            "Each cell shows the magnitude of the hidden state vector at that (layer, token) position. "
            "Brighter = larger magnitude. The model builds up representations layer by layer — "
            "bright vertical stripes indicate tokens the model considers important throughout processing."
        )
        add_explanation(fig, explanation)

        fig.tight_layout(rect=[0, 0.08, 1, 1])
        save_figure(fig, output_dir / f"{meta['prompt_id']}_activation_heatmap.png")
        logger.info("Saved activation heatmap for '%s'", meta["prompt"])
