"""Visualization 8: 3D Activation Space.

Projects hidden states into 3D using PCA and renders multiple views
to reveal the spatial geometry of the model's representations.

WHAT THIS REVEALS:
The 2D PCA captures only the top two principal components. Adding a
third component often reveals structure that was hidden — clusters
that overlapped in 2D may separate in 3D, and trajectories that
appeared to cross may actually pass over/under each other.

Three figures per prompt:

1. LAYER CLOUD — All hidden states as a 3D point cloud, colored by
   layer depth. This shows the overall "shape" of the activation space.
   Early layers (blue) and late layers (red) typically occupy different
   regions. The transition between them reveals how the model reshapes
   its representation of the input.

2. TOKEN TRAJECTORIES — Each token's path through the 24 layers,
   drawn as a 3D curve. Long, winding paths mean the model is
   significantly transforming that token's representation. Short,
   tight paths mean the token's meaning was established early and
   barely changed. Where paths converge, the model is building a
   shared representation; where they diverge, it's differentiating
   tokens.

3. LAYER SLICES — The same 3D space shown from three orthogonal
   viewing angles (front, top, side), giving you a complete picture
   without needing to interactively rotate.

The third principal component often captures 5-15% of the variance,
which is enough to reveal meaningful structure missed by 2D views.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
    """Generate 3D activation space visualizations."""
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

        # PCA to 3D
        all_states = residual.astype(np.float32).reshape(-1, hidden_dim)
        pca = PCA(n_components=3)
        projected = pca.fit_transform(all_states)
        projected = projected.reshape(num_layers_plus_one, seq_len, 3)
        explained = pca.explained_variance_ratio_

        axis_labels = [
            f"PC1 ({explained[0]:.1%})",
            f"PC2 ({explained[1]:.1%})",
            f"PC3 ({explained[2]:.1%})",
        ]

        _plot_layer_cloud(projected, num_layers_plus_one, meta, axis_labels, output_dir)
        _plot_token_trajectories(projected, num_layers_plus_one, seq_len, tokens, meta, axis_labels, output_dir)
        _plot_layer_slices(projected, num_layers_plus_one, seq_len, tokens, meta, axis_labels, output_dir)

        logger.info("Saved 3D activation plots for '%s'", meta["prompt"])


def _plot_layer_cloud(projected, num_layers_plus_one, meta, axis_labels, output_dir):
    """Figure 1: 3D point cloud colored by layer depth."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    for layer_idx in range(num_layers_plus_one):
        t = layer_idx / (num_layers_plus_one - 1)
        color = plt.cm.coolwarm(t)
        ax.scatter(
            projected[layer_idx, :, 0],
            projected[layer_idx, :, 1],
            projected[layer_idx, :, 2],
            c=[color],
            s=25,
            alpha=0.7,
            depthshade=True,
        )

    ax.set_xlabel(axis_labels[0], fontsize=9, labelpad=8)
    ax.set_ylabel(axis_labels[1], fontsize=9, labelpad=8)
    ax.set_zlabel(axis_labels[2], fontsize=9, labelpad=8)
    ax.set_title(
        f"3D Activation Space — \"{meta['prompt']}\"\n"
        f"All hidden states colored by layer (blue=early, red=late)",
        fontsize=12,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm",
        norm=plt.Normalize(vmin=0, vmax=num_layers_plus_one - 1),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, label="Layer index")

    ax.view_init(elev=25, azim=135)
    ax.tick_params(labelsize=7)

    explanation = (
        "Each point is one token's hidden state at one layer, projected from 2048-d to 3D via PCA. "
        "The spatial separation between blue (early) and red (late) clusters shows how much "
        "the model transforms its representations across layers. The third axis often reveals "
        "structure invisible in 2D projections."
    )
    add_explanation(fig, explanation, y=0.02)

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_figure(fig, output_dir / f"{meta['prompt_id']}_3d_layer_cloud.png")


def _plot_token_trajectories(projected, num_layers_plus_one, seq_len, tokens, meta, axis_labels, output_dir):
    """Figure 2: 3D token trajectories through layers."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    for tok_idx in range(seq_len):
        trajectory = projected[:, tok_idx, :]  # [num_layers+1, 3]
        color = plt.cm.tab10(tok_idx % 10)

        # Draw the trajectory as a line with gradient alpha
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=color,
            linewidth=1.5,
            alpha=0.6,
        )

        # Layer-by-layer points along trajectory, fading from small to large
        sizes = np.linspace(8, 30, num_layers_plus_one)
        for layer_idx in range(num_layers_plus_one):
            ax.scatter(
                trajectory[layer_idx, 0],
                trajectory[layer_idx, 1],
                trajectory[layer_idx, 2],
                c=[color],
                s=sizes[layer_idx],
                alpha=0.4,
                depthshade=True,
            )

        # Mark start and end
        ax.scatter(
            *trajectory[0],
            marker="o",
            s=80,
            c=[color],
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
            depthshade=False,
        )
        ax.scatter(
            *trajectory[-1],
            marker="*",
            s=200,
            c=[color],
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
            depthshade=False,
            label=f'"{tokens[tok_idx]}"',
        )

    ax.set_xlabel(axis_labels[0], fontsize=9, labelpad=8)
    ax.set_ylabel(axis_labels[1], fontsize=9, labelpad=8)
    ax.set_zlabel(axis_labels[2], fontsize=9, labelpad=8)
    ax.set_title(
        f"Token Trajectories in 3D — \"{meta['prompt']}\"\n"
        f"(circle = embedding, star = final layer, dot size grows with depth)",
        fontsize=12,
    )
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.95))
    ax.view_init(elev=20, azim=135)
    ax.tick_params(labelsize=7)

    explanation = (
        "Each colored path traces one token's hidden state as it flows through all 24 layers. "
        "Dots grow larger at deeper layers. Long, winding paths = the model is heavily transforming "
        "that token. Converging paths = the model is merging token representations. "
        "Sudden jumps between adjacent layers may indicate where the model 'figures out' the answer."
    )
    add_explanation(fig, explanation, y=0.02)

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_figure(fig, output_dir / f"{meta['prompt_id']}_3d_trajectories.png")


def _plot_layer_slices(projected, num_layers_plus_one, seq_len, tokens, meta, axis_labels, output_dir):
    """Figure 3: Three orthogonal views of the 3D space."""
    fig = plt.figure(figsize=(18, 6))

    views = [
        ("Front (PC1 vs PC2)", 0, 90),    # Looking along PC3
        ("Top (PC1 vs PC3)", 90, 90),      # Looking down along PC2
        ("Side (PC2 vs PC3)", 0, 0),       # Looking along PC1
    ]

    for panel_idx, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, panel_idx + 1, projection="3d")

        for tok_idx in range(seq_len):
            trajectory = projected[:, tok_idx, :]
            color = plt.cm.tab10(tok_idx % 10)

            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=color,
                linewidth=1.2,
                alpha=0.5,
            )

            # Start and end markers
            ax.scatter(
                *trajectory[0],
                marker="o",
                s=50,
                c=[color],
                edgecolors="black",
                linewidths=0.5,
                depthshade=False,
            )
            ax.scatter(
                *trajectory[-1],
                marker="*",
                s=120,
                c=[color],
                edgecolors="black",
                linewidths=0.5,
                depthshade=False,
                label=f'"{tokens[tok_idx]}"' if panel_idx == 0 else None,
            )

        ax.set_xlabel(axis_labels[0], fontsize=8, labelpad=5)
        ax.set_ylabel(axis_labels[1], fontsize=8, labelpad=5)
        ax.set_zlabel(axis_labels[2], fontsize=8, labelpad=5)
        ax.set_title(view_name, fontsize=10)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=6)

    fig.legend(
        *fig.axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=min(seq_len, 6),
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Three Views of Activation Space — \"{meta['prompt']}\"",
        fontsize=13,
    )

    explanation = (
        "The same 3D token trajectories viewed from three orthogonal angles. "
        "Structures that overlap in one view often separate in another — "
        "check all three to understand the true geometry of the activation space."
    )
    add_explanation(fig, explanation, y=0.01)

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    save_figure(fig, output_dir / f"{meta['prompt_id']}_3d_views.png")
