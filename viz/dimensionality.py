"""Visualization 7: Dimensionality Reduction (PCA) of Hidden States.

Projects the high-dimensional hidden states (2048-d) into 2D using
PCA to reveal the geometry of the model's representations.

WHAT THIS REVEALS:
Each token at each layer has a 2048-dimensional hidden state vector.
PCA finds the two directions of greatest variation and projects onto
them, giving us a "map" of the representation space.

What to look for:

- LAYER TRAJECTORY: Points colored by layer depth (blue=early, red=late)
  show how the representation transforms. If early and late layers are
  far apart, the model is making large representational changes. If they
  cluster together, the representation is relatively stable.

- CLUSTERING BY TOKEN: If tokens that are semantically similar cluster
  together (even across layers), the model has learned meaningful
  representations.

- PHASE TRANSITIONS: Sometimes you'll see a sudden jump between
  adjacent layers — the representation changes dramatically. This
  often corresponds to the layer where the model "figures out" the
  answer (visible in the logit lens too).

- EXPLAINED VARIANCE: The percentage shown on each axis tells you
  how much of the total variation that axis captures. High values
  mean the 2D projection is faithful to the true high-dimensional
  geometry.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from viz.common import (
    add_explanation,
    layer_labels,
    load_activations,
    save_figure,
    token_labels,
)

logger = logging.getLogger(__name__)


def generate(activations_dir: Path, output_dir: Path) -> None:
    """Generate PCA dimensionality reduction plots."""
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
        num_layers_plus_one, seq_len, hidden_dim = residual.shape

        # Reshape to [num_layers+1 * seq_len, hidden_dim] for PCA
        all_states = residual.astype(np.float32).reshape(-1, hidden_dim)

        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_states)  # [N, 2]
        projected = projected.reshape(num_layers_plus_one, seq_len, 2)

        explained = pca.explained_variance_ratio_

        # === Figure 1: All tokens, colored by layer ===
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Color by layer depth
        ax = axes[0]
        for layer_idx in range(num_layers_plus_one):
            color = plt.cm.coolwarm(layer_idx / (num_layers_plus_one - 1))
            ax.scatter(
                projected[layer_idx, :, 0],
                projected[layer_idx, :, 1],
                c=[color],
                s=20,
                alpha=0.7,
                label=f"L{layer_idx - 1}" if layer_idx > 0 else "Embed",
            )

        ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
        ax.set_title("Hidden States Colored by Layer Depth\n(blue=early, red=late)")
        ax.grid(True, alpha=0.2)

        # Add a colorbar instead of legend (too many layers)
        sm = plt.cm.ScalarMappable(
            cmap="coolwarm",
            norm=plt.Normalize(vmin=0, vmax=num_layers_plus_one - 1),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, label="Layer index")

        # Right: Trajectory of each token through layers
        ax = axes[1]
        for tok_idx in range(seq_len):
            trajectory = projected[:, tok_idx, :]  # [num_layers+1, 2]
            color = plt.cm.tab10(tok_idx % 10)

            # Draw trajectory as connected line
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                linewidth=1,
                alpha=0.5,
            )

            # Mark start (embed) and end (final layer)
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                marker="o",
                s=60,
                c=[color],
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                marker="*",
                s=120,
                c=[color],
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label=f'"{tokens[tok_idx]}"',
            )

        ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
        ax.set_title("Token Trajectories Through Layers\n(circle=embed, star=final layer)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.2)

        fig.suptitle(
            f"PCA of Hidden States — \"{meta['prompt']}\"",
            fontsize=13,
        )

        explanation = (
            "LEFT: Every hidden state projected to 2D, colored by layer (blue=early, red=late). "
            "Spread between colors shows how much the representation changes across layers. "
            "RIGHT: Each token's trajectory through layers (circle=start, star=end). "
            "Long trajectories mean the token's representation changed dramatically during processing."
        )
        add_explanation(fig, explanation, y=0.01)

        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        save_figure(fig, output_dir / f"{meta['prompt_id']}_pca.png")
        logger.info("Saved PCA plot for '%s'", meta["prompt"])
