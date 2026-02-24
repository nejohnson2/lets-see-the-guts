"""Visualization 3: Residual Stream Norm Evolution.

Shows how the magnitude (L2 norm) of each token's hidden state grows
as it passes through the layers of the network.

WHAT THIS REVEALS:
The "residual stream" is the main highway of information in a transformer.
Each layer ADDS its output to the residual stream (that's the "residual"
in "residual connection"). So the hidden state at layer L is:

    h_L = h_0 + attn_1(h_0) + mlp_1(h_0) + attn_2(h_1) + mlp_2(h_1) + ...

Because each layer adds to the stream, the L2 norm (magnitude) of the
hidden state typically grows through the network. The RATE of growth
tells you how much each layer is contributing — a steep jump means that
layer is making a big change to the representation.

Different tokens often show different growth patterns. For example,
content words ("France", "soup") may grow differently than function
words ("the", "is") because the model processes them differently.
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


def generate(activations_dir: Path, output_dir: Path) -> None:
    """Generate residual norm plots for each prompt."""
    from viz.common import get_prompt_dirs

    prompt_dirs = get_prompt_dirs(activations_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_dir in prompt_dirs:
        data = load_activations(prompt_dir)
        if "residual_stream" not in data:
            logger.warning("No residual_stream data in %s, skipping", prompt_dir)
            continue

        residual = data["residual_stream"]  # [num_layers+1, seq_len, hidden_dim]
        meta = data["metadata"]
        tokens = token_labels(meta["token_strings"])
        num_layers_plus_one, seq_len, _ = residual.shape

        # Compute L2 norms: [num_layers+1, seq_len]
        norms = np.linalg.norm(residual.astype(np.float32), axis=-1)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        labels = layer_labels(num_layers_plus_one - 1, include_embed=True)
        x = np.arange(num_layers_plus_one)

        for tok_idx in range(seq_len):
            color = plt.cm.viridis(tok_idx / max(seq_len - 1, 1))
            ax.plot(
                x,
                norms[:, tok_idx],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=f'"{tokens[tok_idx]}"',
                alpha=0.8,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("L2 Norm of Hidden State")
        ax.set_title(f"Residual Stream Norm Growth — \"{meta['prompt']}\"")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        explanation = (
            "Each line shows one token's hidden state magnitude as it flows through the network. "
            "The residual stream grows because each layer ADDS to it. Steep jumps = high-impact layers. "
            "Different tokens grow at different rates based on their role in the input."
        )
        add_explanation(fig, explanation)

        fig.tight_layout(rect=[0, 0.08, 1, 1])
        save_figure(fig, output_dir / f"{meta['prompt_id']}_residual_norms.png")
        logger.info("Saved residual norms plot for '%s'", meta["prompt"])
