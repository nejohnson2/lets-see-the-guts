"""Shared visualization utilities and constants."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Consistent style
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)

# Color palette for prompts
PROMPT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
PROMPT_MARKERS = ["o", "s", "^", "D", "v"]


def clean_token_label(token: str) -> str:
    """Clean a token string for display in plots.

    Handles common tokenizer artifacts like:
    - Metaspace characters (Ġ, ▁) used for word boundaries
    - Special tokens (<s>, </s>, <|endoftext|>)
    - Newlines and tabs
    """
    replacements = {
        "Ġ": " ",
        "▁": " ",
        "Ċ": "\\n",
        "ĉ": "\\t",
        "<s>": "[BOS]",
        "</s>": "[EOS]",
        "<|endoftext|>": "[EOS]",
    }
    for old, new in replacements.items():
        token = token.replace(old, new)
    return token


def token_labels(token_strings: list[str]) -> list[str]:
    """Convert a list of token strings to clean display labels."""
    return [clean_token_label(t) for t in token_strings]


def layer_labels(num_layers: int, include_embed: bool = True) -> list[str]:
    """Generate layer labels: ['Embed', 'L0', 'L1', ..., 'L23']."""
    labels = []
    if include_embed:
        labels.append("Embed")
    labels.extend([f"L{i}" for i in range(num_layers)])
    return labels


def load_activations(prompt_dir: Path) -> dict:
    """Load all saved activations and metadata for a prompt."""
    data = {}

    meta_path = prompt_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)

    for npy_file in prompt_dir.glob("*.npy"):
        data[npy_file.stem] = np.load(npy_file)

    npz_file = prompt_dir / "logit_lens_topk.npz"
    if npz_file.exists():
        npz = np.load(npz_file)
        data["logit_lens_top_k_indices"] = npz["top_k_indices"]
        data["logit_lens_top_k_probs"] = npz["top_k_probs"]

    return data


def add_explanation(fig, text: str, y: float = 0.02, fontsize: int = 9):
    """Add explanatory text at the bottom of a figure."""
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        style="italic",
        color="#444444",
        wrap=True,
        transform=fig.transFigure,
    )


def save_figure(fig, path: Path, dpi: int = 150):
    """Save figure with consistent settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def get_prompt_dirs(activations_dir: Path) -> list[Path]:
    """Find all prompt directories in the activations folder."""
    dirs = sorted(
        [d for d in activations_dir.iterdir() if d.is_dir() and d.name.startswith("prompt_")]
    )
    return dirs
