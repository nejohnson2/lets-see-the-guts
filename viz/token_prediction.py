"""Visualization 5: Token Prediction Evolution (Logit Lens).

Shows how the model's "best guess" for the next token changes as
information flows through the layers.

WHAT THIS REVEALS:
The "logit lens" is a powerful interpretability technique. At each layer,
we take the intermediate hidden state and project it through the model's
final output head (RMSNorm → unembedding matrix) to see what token the
model would predict if it stopped processing at that layer.

This reveals the MODEL'S REASONING PROCESS layer by layer:

- Early layers (L0-L5): Predictions are usually generic high-frequency
  tokens ("the", "a", ".") because the model hasn't yet computed
  meaningful representations.

- Middle layers (L6-L17): Predictions start to shift toward semantically
  relevant tokens. For "The capital of France is", you might see
  country names and cities appearing.

- Late layers (L18-L23): The correct prediction crystallizes. "Paris"
  should appear and gain confidence in the final layers.

The LAYER where the correct answer first appears tells you which layer
"knows" the answer. Some facts are retrieved early (well-memorized),
while others require deep processing (complex reasoning).

This is one of the most fascinating views into how a model actually works.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    """Generate logit lens visualizations."""
    from viz.common import get_prompt_dirs

    prompt_dirs = get_prompt_dirs(activations_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for decoding token IDs
    from transformers import AutoTokenizer
    from src.config import MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    for prompt_dir in prompt_dirs:
        data = load_activations(prompt_dir)
        if "logit_lens_top_k_indices" not in data:
            logger.warning("No logit lens data in %s, skipping", prompt_dir)
            continue

        top_k_indices = data["logit_lens_top_k_indices"]  # [L+1, S, K]
        top_k_probs = data["logit_lens_top_k_probs"]  # [L+1, S, K]
        meta = data["metadata"]
        tokens = token_labels(meta["token_strings"])
        num_layers_plus_one, seq_len, top_k = top_k_indices.shape

        # Focus on the last token position (the prediction position)
        # and also show all positions in a summary view

        # === Figure 1: Last-position prediction evolution ===
        fig, ax = plt.subplots(figsize=(10, 12))

        labels = layer_labels(num_layers_plus_one - 1, include_embed=True)
        last_pos = seq_len - 1

        # Show top 5 predictions at each layer for the last position
        show_k = 5
        cell_height = 1.0
        cell_width = 3.0

        for layer_idx in range(num_layers_plus_one):
            y = num_layers_plus_one - 1 - layer_idx  # Flip so early layers at bottom

            for k in range(show_k):
                token_id = top_k_indices[layer_idx, last_pos, k]
                prob = top_k_probs[layer_idx, last_pos, k]
                token_str = tokenizer.decode([token_id]).strip()
                if not token_str:
                    token_str = repr(tokenizer.decode([token_id]))

                x = k * cell_width

                # Color by probability
                color = plt.cm.YlOrRd(prob)
                rect = plt.Rectangle(
                    (x, y - 0.4),
                    cell_width - 0.1,
                    0.8,
                    facecolor=color,
                    edgecolor="gray",
                    linewidth=0.5,
                    alpha=0.8,
                )
                ax.add_patch(rect)

                # Token text
                display_text = f'"{token_str}"' if len(token_str) <= 12 else f'"{token_str[:10]}.."'
                ax.text(
                    x + cell_width / 2 - 0.05,
                    y + 0.05,
                    display_text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold" if k == 0 else "normal",
                )
                ax.text(
                    x + cell_width / 2 - 0.05,
                    y - 0.2,
                    f"{prob:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#555555",
                )

        # Y-axis: layer labels
        ax.set_yticks(range(num_layers_plus_one))
        ax.set_yticklabels(list(reversed(labels)), fontsize=8)
        ax.set_ylabel("Layer (bottom = early, top = late)")

        # X-axis
        ax.set_xticks([k * cell_width + cell_width / 2 for k in range(show_k)])
        ax.set_xticklabels([f"Rank {k + 1}" for k in range(show_k)])

        ax.set_xlim(-0.2, show_k * cell_width)
        ax.set_ylim(-0.6, num_layers_plus_one - 0.4)

        ax.set_title(
            f"Logit Lens — What does the model predict at each layer?\n"
            f"Prompt: \"{meta['prompt']}\" | Predicting token after \"{tokens[-1]}\"",
            fontsize=11,
        )

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap="YlOrRd", norm=mcolors.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, label="Probability")

        explanation = (
            "Each row is a layer (bottom=embedding, top=final layer). Each cell shows "
            "what token the model would predict if processing stopped at that layer. "
            "Watch how the top prediction changes from generic words (early) to the "
            "specific answer (late). The layer where the right answer appears reveals "
            "WHERE in the network that knowledge lives."
        )
        add_explanation(fig, explanation, y=0.01)

        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        save_figure(fig, output_dir / f"{meta['prompt_id']}_logit_lens.png")

        # === Figure 2: All positions summary ===
        # Show top-1 prediction confidence across all layers and positions
        fig2, ax2 = plt.subplots(figsize=(max(8, seq_len * 1.5), 10))

        top1_probs = top_k_probs[:, :, 0]  # [L+1, S] — probability of top prediction

        im = ax2.imshow(
            top1_probs,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        # Overlay top-1 token text
        for layer_idx in range(num_layers_plus_one):
            for pos in range(seq_len):
                token_id = top_k_indices[layer_idx, pos, 0]
                token_str = tokenizer.decode([token_id]).strip()
                if len(token_str) > 6:
                    token_str = token_str[:5] + ".."
                prob = top1_probs[layer_idx, pos]
                text_color = "white" if prob > 0.5 else "black"
                ax2.text(
                    pos,
                    layer_idx,
                    token_str,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )

        ax2.set_yticks(np.arange(num_layers_plus_one))
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xticks(np.arange(seq_len))
        ax2.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax2.set_xlabel("Token Position (input)")
        ax2.set_ylabel("Layer")
        ax2.set_title(
            f"Top-1 Prediction at Every (Layer, Position) — \"{meta['prompt']}\"",
            fontsize=11,
        )
        fig2.colorbar(im, ax=ax2, shrink=0.8, label="Top-1 Probability")

        explanation2 = (
            "Each cell shows the model's single best prediction at that (layer, position). "
            "Text = predicted token, color = confidence. At each position, the model predicts "
            "the NEXT token. Bright cells = high confidence. Watch how predictions sharpen in later layers."
        )
        add_explanation(fig2, explanation2)

        fig2.tight_layout(rect=[0, 0.08, 1, 1])
        save_figure(fig2, output_dir / f"{meta['prompt_id']}_logit_lens_all_positions.png")

        logger.info("Saved logit lens plots for '%s'", meta["prompt"])
