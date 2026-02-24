"""Activation capture orchestrator for prefill and generation phases."""

import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hooks import ActivationStore, HookManager
from src.storage import (
    save_generation_metadata,
    save_generation_step,
    save_prompt_activations,
)

logger = logging.getLogger(__name__)


def capture_prefill(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
) -> tuple[ActivationStore, object, torch.Tensor, torch.Tensor]:
    """Capture all activations during prompt encoding (prefill).

    Returns:
        store: ActivationStore with all captured activations.
        past_kv: KV cache for generation phase.
        input_ids: Tokenized input tensor.
        first_token: The first generated token (argmax of final logits).
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    logger.info(
        "Prefill: '%s' → %d tokens: %s",
        prompt,
        len(tokens),
        tokens,
    )

    store = ActivationStore()

    with torch.no_grad():
        with HookManager(model, store):
            outputs = model(
                input_ids,
                output_attentions=True,
                use_cache=True,
            )

    logger.info(
        "Captured: %d residual states, %d attention maps, %d gate activations, %d MLP outputs",
        len(store.residual_stream),
        len(store.attention_weights),
        len(store.mlp_gate_pre_act),
        len(store.mlp_outputs),
    )

    # Get first generated token from prefill logits
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return store, outputs.past_key_values, input_ids, first_token


def capture_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    past_kv,
    first_token: torch.Tensor,
    max_tokens: int,
    gen_dir,
) -> list[dict]:
    """Capture activations during token-by-token generation.

    Uses a manual generation loop (not model.generate()) for full
    control over per-step activation capture.

    Args:
        past_kv: KV cache from the prefill phase.
        first_token: First token to generate (from prefill logits).
        max_tokens: Maximum tokens to generate.
        gen_dir: Directory to save per-step activations.

    Returns:
        List of dicts with token_id, token_str, and step index.
    """
    generated_tokens = []
    current_kv = past_kv
    next_token = first_token

    for step in tqdm(range(max_tokens), desc="Generating", leave=False):
        step_store = ActivationStore()

        with torch.no_grad():
            with HookManager(model, step_store):
                outputs = model(
                    next_token,
                    past_key_values=current_kv,
                    output_attentions=True,
                    use_cache=True,
                )

        current_kv = outputs.past_key_values

        token_id = next_token.item()
        token_str = tokenizer.decode(token_id)
        generated_tokens.append(
            {"step": step, "token_id": token_id, "token_str": token_str}
        )

        # Save this step's activations
        save_generation_step(gen_dir, step, step_store)

        # Get next token
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            logger.info("Generation stopped at EOS after %d tokens", step + 1)
            break

    generated_text = tokenizer.decode(
        [t["token_id"] for t in generated_tokens], skip_special_tokens=True
    )
    logger.info("Generated %d tokens: '%s'", len(generated_tokens), generated_text)

    return generated_tokens


def capture_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_config: dict,
    prompt_dir,
    device: torch.device,
) -> None:
    """Full capture pipeline for a single prompt.

    1. Prefill: encode the full prompt, capture all layer activations
    2. Generation: generate tokens one by one, capture per-step activations
    3. Save everything to disk
    """
    prompt = prompt_config["text"]
    max_gen = prompt_config.get("max_gen_tokens", 30)

    # Phase 1: Prefill
    store, past_kv, input_ids, first_token = capture_prefill(
        model, tokenizer, prompt, device
    )

    # Build metadata
    token_ids = input_ids[0].tolist()
    metadata = {
        "prompt": prompt,
        "prompt_id": prompt_config["id"],
        "description": prompt_config.get("description", ""),
        "token_ids": token_ids,
        "token_strings": tokenizer.convert_ids_to_tokens(token_ids),
        "num_tokens": len(token_ids),
    }

    # Save prefill activations
    save_prompt_activations(prompt_dir, store, metadata)

    # Phase 2: Generation
    gen_dir = prompt_dir / "generation"
    generated = capture_generation(
        model, tokenizer, past_kv, first_token, max_gen, gen_dir
    )

    # Save generation metadata and update prompt metadata
    save_generation_metadata(gen_dir, generated)
    metadata["generated_tokens"] = generated
    metadata["generated_text"] = tokenizer.decode(
        [t["token_id"] for t in generated], skip_special_tokens=True
    )

    # Re-save metadata with generation info included
    save_prompt_activations(prompt_dir, store, metadata)
