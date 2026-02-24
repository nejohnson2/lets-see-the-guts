"""Hook manager and activation store for capturing model internals.

Uses PyTorch forward hooks to intercept activations at every layer
without modifying the model. Activations are immediately detached,
moved to CPU, and converted to float16 to minimize GPU/MPS memory usage.
"""

import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class ActivationStore:
    """Stores captured activations from a single forward pass.

    All tensors are stored as CPU float16 with batch dimension squeezed.
    """

    # Output of embed_tokens, then output of each decoder layer
    # List of [seq_len, hidden_dim] tensors — length = num_layers + 1
    residual_stream: list[torch.Tensor] = field(default_factory=list)

    # Attention weights per layer
    # List of [num_heads, seq_len, key_len] tensors — length = num_layers
    attention_weights: list[torch.Tensor] = field(default_factory=list)

    # Raw gate_proj output per layer (pre-SiLU)
    # List of [seq_len, intermediate_size] tensors — length = num_layers
    mlp_gate_pre_act: list[torch.Tensor] = field(default_factory=list)

    # Full MLP block output per layer
    # List of [seq_len, hidden_dim] tensors — length = num_layers
    mlp_outputs: list[torch.Tensor] = field(default_factory=list)

    def clear(self):
        """Clear all stored activations."""
        self.residual_stream.clear()
        self.attention_weights.clear()
        self.mlp_gate_pre_act.clear()
        self.mlp_outputs.clear()


def _to_cpu_f16(tensor: torch.Tensor) -> torch.Tensor:
    """Detach, move to CPU, convert to float16, remove batch dim."""
    return tensor.detach().cpu().to(torch.float16).squeeze(0)


class HookManager:
    """Register and manage forward hooks on the model.

    Use as a context manager to ensure hooks are always removed:

        store = ActivationStore()
        with HookManager(model, store):
            model(input_ids, output_attentions=True)
        # store now contains all activations
    """

    def __init__(self, model, store: ActivationStore):
        self.model = model
        self.store = store
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def __enter__(self):
        self._register_all_hooks()
        return self

    def __exit__(self, *args):
        self._remove_all_hooks()

    def _remove_all_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _register_all_hooks(self):
        # 1. Embedding output → first entry in residual stream
        self._handles.append(
            self.model.model.embed_tokens.register_forward_hook(self._embedding_hook)
        )

        # 2. Per-layer hooks
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Decoder layer output → residual stream
            self._handles.append(
                layer.register_forward_hook(self._make_residual_hook(layer_idx))
            )
            # Self-attention → attention weights
            self._handles.append(
                layer.self_attn.register_forward_hook(
                    self._make_attention_hook(layer_idx)
                )
            )
            # gate_proj (Linear) → raw gate activations before SiLU
            self._handles.append(
                layer.mlp.gate_proj.register_forward_hook(
                    self._make_gate_hook(layer_idx)
                )
            )
            # MLP block → full MLP output
            self._handles.append(
                layer.mlp.register_forward_hook(
                    self._make_mlp_output_hook(layer_idx)
                )
            )

        logger.debug(
            "Registered %d hooks across %d layers",
            len(self._handles),
            len(self.model.model.layers),
        )

    def _embedding_hook(self, module, input, output):
        self.store.residual_stream.append(_to_cpu_f16(output))

    def _make_residual_hook(self, layer_idx: int):
        def hook(module, input, output):
            # LlamaDecoderLayer returns a tuple: (hidden_states, ...)
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.store.residual_stream.append(_to_cpu_f16(hidden_states))

        return hook

    def _make_attention_hook(self, layer_idx: int):
        def hook(module, input, output):
            # LlamaAttention.forward returns (attn_output, attn_weights, past_kv)
            # attn_weights is at index 1 when output_attentions=True
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                self.store.attention_weights.append(_to_cpu_f16(output[1]))
            else:
                logger.warning(
                    "Layer %d: attention weights not available. "
                    "Ensure output_attentions=True in model forward call.",
                    layer_idx,
                )

        return hook

    def _make_gate_hook(self, layer_idx: int):
        def hook(module, input, output):
            self.store.mlp_gate_pre_act.append(_to_cpu_f16(output))

        return hook

    def _make_mlp_output_hook(self, layer_idx: int):
        def hook(module, input, output):
            self.store.mlp_outputs.append(_to_cpu_f16(output))

        return hook
