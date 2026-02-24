"""Visualization 2: Attention Pattern Matrices.

Shows the attention weight matrices for selected heads at early, middle,
and late layers.

WHAT THIS REVEALS:
Attention is how tokens "talk to" each other. Each attention head produces
a matrix where entry [i, j] represents how much token i attends to token j
when computing its updated representation.

The model has 32 attention heads per layer × 24 layers = 768 heads total.
Each head learns a different "communication pattern." Common patterns:

- DIAGONAL: Token attends mainly to itself ("self-attention"). Common in
  early layers where the model is still building local representations.

- COLUMN (vertical stripe): All tokens attend to one specific token. This
  often happens with semantically important words or punctuation. The BOS
  (beginning of sequence) token often receives high attention as a "sink."

- LOWER TRIANGLE: Each token attends to all previous tokens roughly
  equally. This is a "uniform lookback" pattern.

- BANDED: Tokens attend to nearby neighbors. This captures local/syntactic
  relationships (adjacent words).

- SPARSE/SPECIFIC: Only certain token pairs show high attention. These
  heads have learned specific linguistic relationships (subject-verb,
  adjective-noun, etc.).

Early layers tend to show positional patterns (diagonal, banded).
Later layers show more semantic patterns (sparse, specific).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz.common import (
    add_explanation,
    load_activations,
    save_figure,
    token_labels,
)

logger = logging.getLogger(__name__)

# Which layers and heads to visualize
# We pick early, middle, and late layers to show progression
LAYER_INDICES = [0, 6, 12, 18, 23]  # 5 layers across the network
HEADS_PER_LAYER = 4  # Show 4 heads per layer


def _select_interesting_heads(
    attn_weights: np.ndarray, layer_idx: int, n_heads: int = HEADS_PER_LAYER
) -> list[int]:
    """Select heads with diverse attention patterns.

    Picks heads based on entropy: the most focused (lowest entropy)
    and most distributed (highest entropy) heads, plus extremes.
    """
    # attn_weights for this layer: [num_heads, seq_len, key_len]
    layer_attn = attn_weights[layer_idx]  # [32, seq_len, key_len]
    num_heads = layer_attn.shape[0]

    # Compute entropy of attention distribution for each head
    # Average across query positions
    entropies = []
    for h in range(num_heads):
        # Clip to avoid log(0)
        attn = np.clip(layer_attn[h].astype(np.float32), 1e-10, 1.0)
        entropy = -np.sum(attn * np.log2(attn), axis=-1).mean()
        entropies.append(entropy)

    entropies = np.array(entropies)
    sorted_heads = np.argsort(entropies)

    # Pick: lowest entropy, highest entropy, and 2 from the middle
    selected = []
    selected.append(sorted_heads[0])  # Most focused
    selected.append(sorted_heads[-1])  # Most distributed
    # Spread the remaining across the range
    step = max(1, len(sorted_heads) // (n_heads - 1))
    for idx in range(1, len(sorted_heads) - 1):
        if len(selected) >= n_heads:
            break
        if idx % step == 0 and sorted_heads[idx] not in selected:
            selected.append(sorted_heads[idx])

    # Pad if needed
    while len(selected) < n_heads:
        for h in range(num_heads):
            if h not in selected:
                selected.append(h)
                break

    return sorted(selected[:n_heads])


def generate(activations_dir: Path, output_dir: Path) -> None:
    """Generate attention pattern visualizations."""
    from viz.common import get_prompt_dirs

    prompt_dirs = get_prompt_dirs(activations_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_dir in prompt_dirs:
        data = load_activations(prompt_dir)
        if "attention_weights" not in data:
            logger.warning("No attention_weights in %s, skipping", prompt_dir)
            continue

        attn = data["attention_weights"]  # [num_layers, num_heads, seq_len, key_len]
        meta = data["metadata"]
        tokens = token_labels(meta["token_strings"])
        num_layers = attn.shape[0]

        # Filter layer indices to those that exist
        layers_to_show = [l for l in LAYER_INDICES if l < num_layers]

        fig, axes = plt.subplots(
            len(layers_to_show),
            HEADS_PER_LAYER,
            figsize=(HEADS_PER_LAYER * 3.5, len(layers_to_show) * 3.5),
            squeeze=False,
        )

        for row, layer_idx in enumerate(layers_to_show):
            heads = _select_interesting_heads(attn, layer_idx)

            for col, head_idx in enumerate(heads):
                ax = axes[row, col]
                head_attn = attn[layer_idx, head_idx].astype(np.float32)

                im = ax.imshow(
                    head_attn,
                    cmap="Blues",
                    vmin=0,
                    vmax=head_attn.max(),
                    interpolation="nearest",
                )

                seq_len = head_attn.shape[0]
                ax.set_xticks(np.arange(seq_len))
                ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(np.arange(seq_len))
                ax.set_yticklabels(tokens, fontsize=7)

                # Compute entropy for subtitle
                attn_clipped = np.clip(head_attn, 1e-10, 1.0)
                entropy = -np.sum(
                    attn_clipped * np.log2(attn_clipped), axis=-1
                ).mean()
                ax.set_title(
                    f"L{layer_idx} H{head_idx}\nentropy={entropy:.2f}",
                    fontsize=9,
                )

                if col == 0:
                    ax.set_ylabel(f"Layer {layer_idx}\n(query)", fontsize=9)
                if row == len(layers_to_show) - 1:
                    ax.set_xlabel("key", fontsize=9)

        fig.suptitle(
            f"Attention Patterns — \"{meta['prompt']}\"\n"
            f"Rows: layers (early→late) | Columns: selected heads (by entropy diversity)",
            fontsize=13,
        )

        explanation = (
            "Each matrix shows how much each token (row=query) attends to every other token (column=key). "
            "Bright = high attention. Low entropy = focused on few tokens; high entropy = distributed broadly. "
            "Early layers often show positional patterns; later layers capture semantic relationships."
        )
        add_explanation(fig, explanation, y=0.01)

        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        save_figure(fig, output_dir / f"{meta['prompt_id']}_attention_patterns.png")
        logger.info("Saved attention patterns for '%s'", meta["prompt"])
