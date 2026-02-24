"""Capture internal activations from SmolLM2-1.7B for visualization.

Usage:
    python run_capture.py                        # All prompts
    python run_capture.py --prompt 0             # Single prompt by index
    python run_capture.py --max-gen-tokens 10    # Override generation length
    python run_capture.py --prompts-file custom.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from src.capture import capture_prompt
from src.config import ACTIVATIONS_DIR, MAX_GENERATION_TOKENS, PROMPTS_FILE
from src.device import get_device, get_model_dtype
from src.logit_lens import compute_logit_lens, save_logit_lens
from src.model import load_model
from src.storage import save_model_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_prompts(prompts_file: Path) -> list[dict]:
    """Load prompt configurations from YAML file."""
    with open(prompts_file) as f:
        data = yaml.safe_load(f)
    return data["prompts"]


def main():
    parser = argparse.ArgumentParser(description="Capture LLM activations")
    parser.add_argument(
        "--prompt",
        type=int,
        default=None,
        help="Index of a single prompt to capture (default: all)",
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        default=None,
        help=f"Override max generation tokens (default: per-prompt setting or {MAX_GENERATION_TOKENS})",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=PROMPTS_FILE,
        help="Path to prompts YAML file",
    )
    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.prompts_file)
    if args.prompt is not None:
        if args.prompt >= len(prompts):
            logger.error("Prompt index %d out of range (have %d prompts)", args.prompt, len(prompts))
            sys.exit(1)
        prompts = [prompts[args.prompt]]

    if args.max_gen_tokens is not None:
        for p in prompts:
            p["max_gen_tokens"] = args.max_gen_tokens

    logger.info("Will capture %d prompt(s)", len(prompts))

    # Setup device and model
    device = get_device()
    dtype = get_model_dtype(device)
    model, tokenizer = load_model(device, dtype)

    # Save model weights for logit lens (only once)
    weights_dir = ACTIVATIONS_DIR / "model_weights"
    save_model_weights(weights_dir, model)

    # Capture each prompt
    for i, prompt_config in enumerate(tqdm(prompts, desc="Prompts")):
        prompt_dir = ACTIVATIONS_DIR / f"prompt_{prompt_config['id']}"
        logger.info(
            "=== Prompt %d/%d: '%s' ===",
            i + 1,
            len(prompts),
            prompt_config["text"],
        )

        capture_prompt(model, tokenizer, prompt_config, prompt_dir, device)

        # Compute logit lens from saved residual stream
        residual_path = prompt_dir / "residual_stream.npy"
        if residual_path.exists():
            residual_stream = np.load(residual_path)
            from src.storage import load_model_weights

            weights = load_model_weights(weights_dir)
            logit_lens_data = compute_logit_lens(
                residual_stream,
                weights["final_norm_weight"],
                weights["embed_tokens_weight"],
            )
            save_logit_lens(prompt_dir, logit_lens_data)

    logger.info("Capture complete. Data saved to %s", ACTIVATIONS_DIR)


if __name__ == "__main__":
    main()
