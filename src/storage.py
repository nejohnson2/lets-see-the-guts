"""Save and load activation data as numpy arrays."""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.hooks import ActivationStore

logger = logging.getLogger(__name__)


def _stack_to_numpy(tensors: list[torch.Tensor]) -> np.ndarray:
    """Stack a list of tensors into a single numpy array."""
    return torch.stack(tensors).numpy()


def save_prompt_activations(
    prompt_dir: Path,
    store: ActivationStore,
    metadata: dict,
) -> None:
    """Save all captured activations for a single prompt.

    Args:
        prompt_dir: Directory to save into (created if needed).
        store: ActivationStore with captured data.
        metadata: Dict with prompt text, token IDs, token strings, etc.
    """
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    with open(prompt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Residual stream: [num_layers+1, seq_len, hidden_dim]
    if store.residual_stream:
        np.save(
            prompt_dir / "residual_stream.npy",
            _stack_to_numpy(store.residual_stream),
        )

    # Attention weights: [num_layers, num_heads, seq_len, key_len]
    if store.attention_weights:
        np.save(
            prompt_dir / "attention_weights.npy",
            _stack_to_numpy(store.attention_weights),
        )

    # MLP gate pre-activations: [num_layers, seq_len, intermediate_size]
    if store.mlp_gate_pre_act:
        np.save(
            prompt_dir / "mlp_gate_pre_act.npy",
            _stack_to_numpy(store.mlp_gate_pre_act),
        )

    # MLP outputs: [num_layers, seq_len, hidden_dim]
    if store.mlp_outputs:
        np.save(
            prompt_dir / "mlp_outputs.npy",
            _stack_to_numpy(store.mlp_outputs),
        )

    logger.info("Saved prefill activations to %s", prompt_dir)


def save_generation_step(
    gen_dir: Path,
    step: int,
    store: ActivationStore,
) -> None:
    """Save activations from a single generation step.

    Generation steps have different shapes than prefill (single token
    query attending to growing key sequence), so they're stored separately.
    """
    gen_dir.mkdir(parents=True, exist_ok=True)

    if store.residual_stream:
        np.save(
            gen_dir / f"step_{step:03d}_residual.npy",
            _stack_to_numpy(store.residual_stream),
        )

    if store.attention_weights:
        # Variable shape: [num_layers, num_heads, 1, prompt_len + step]
        np.save(
            gen_dir / f"step_{step:03d}_attention.npy",
            _stack_to_numpy(store.attention_weights),
        )


def save_generation_metadata(gen_dir: Path, generated_tokens: list[dict]) -> None:
    """Save generation metadata (token IDs, strings, step info)."""
    gen_dir.mkdir(parents=True, exist_ok=True)
    with open(gen_dir / "generated_tokens.json", "w") as f:
        json.dump(generated_tokens, f, indent=2)


def save_model_weights(weights_dir: Path, model) -> None:
    """Save model weights needed for logit lens recomputation.

    Saves the final RMSNorm weight and the (tied) embedding matrix.
    """
    weights_dir.mkdir(parents=True, exist_ok=True)

    norm_weight = model.model.norm.weight.detach().cpu().float().numpy()
    np.save(weights_dir / "final_norm_weight.npy", norm_weight)

    embed_weight = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    np.save(weights_dir / "embed_tokens_weight.npy", embed_weight)

    logger.info("Saved model weights for logit lens to %s", weights_dir)


def load_prompt_activations(prompt_dir: Path) -> dict:
    """Load all saved activations for a single prompt.

    Returns a dict with keys matching the saved filenames (without extension).
    """
    data = {}

    meta_path = prompt_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data["metadata"] = json.load(f)

    for npy_file in prompt_dir.glob("*.npy"):
        data[npy_file.stem] = np.load(npy_file)

    # Load generation data if present
    gen_dir = prompt_dir / "generation"
    if gen_dir.exists():
        gen_data = {"steps": {}}
        gen_meta = gen_dir / "generated_tokens.json"
        if gen_meta.exists():
            with open(gen_meta) as f:
                gen_data["tokens"] = json.load(f)

        for npy_file in sorted(gen_dir.glob("*.npy")):
            gen_data["steps"][npy_file.stem] = np.load(npy_file)

        data["generation"] = gen_data

    return data


def load_model_weights(weights_dir: Path) -> dict:
    """Load saved model weights for logit lens."""
    return {
        "final_norm_weight": np.load(weights_dir / "final_norm_weight.npy"),
        "embed_tokens_weight": np.load(weights_dir / "embed_tokens_weight.npy"),
    }
