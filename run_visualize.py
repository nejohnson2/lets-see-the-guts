"""Generate all visualizations from saved activation data.

Usage:
    python run_visualize.py                      # All visualizations, all prompts
    python run_visualize.py --viz attention       # Single visualization type
    python run_visualize.py --prompt 0            # Single prompt (by index)
    python run_visualize.py --dpi 300             # Publication quality
"""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from src.config import ACTIVATIONS_DIR, FIGURES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Registry of visualization modules
VIZ_MODULES = {
    "heatmap": ("viz.activation_heatmap", "01_activation_heatmap"),
    "attention": ("viz.attention_patterns", "02_attention_patterns"),
    "residual": ("viz.residual_norms", "03_residual_norms"),
    "mlp": ("viz.mlp_gates", "04_mlp_gates"),
    "logit_lens": ("viz.token_prediction", "05_token_prediction"),
    "cross_prompt": ("viz.cross_prompt", "06_cross_prompt"),
    "pca": ("viz.dimensionality", "07_dimensionality"),
    "3d": ("viz.activation_3d", "08_activation_3d"),
}


def main():
    parser = argparse.ArgumentParser(description="Generate LLM activation visualizations")
    parser.add_argument(
        "--viz",
        type=str,
        default=None,
        choices=list(VIZ_MODULES.keys()),
        help="Generate only this visualization type (default: all)",
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=None,
        help="Index of a single prompt to visualize (default: all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures (default: 150, use 300 for publication)",
    )
    args = parser.parse_args()

    # Verify data exists
    if not ACTIVATIONS_DIR.exists():
        logger.error(
            "No activation data found at %s. Run run_capture.py first.", ACTIVATIONS_DIR
        )
        sys.exit(1)

    # Determine which prompt directories to use
    from viz.common import get_prompt_dirs

    all_prompt_dirs = get_prompt_dirs(ACTIVATIONS_DIR)
    if not all_prompt_dirs:
        logger.error("No prompt directories found in %s", ACTIVATIONS_DIR)
        sys.exit(1)

    if args.prompt is not None:
        if args.prompt >= len(all_prompt_dirs):
            logger.error(
                "Prompt index %d out of range (have %d prompts)",
                args.prompt,
                len(all_prompt_dirs),
            )
            sys.exit(1)
        activations_dir = ACTIVATIONS_DIR
    else:
        activations_dir = ACTIVATIONS_DIR

    # Determine which visualizations to run
    if args.viz:
        viz_to_run = {args.viz: VIZ_MODULES[args.viz]}
    else:
        viz_to_run = VIZ_MODULES

    # Override DPI if specified
    import matplotlib.pyplot as plt
    plt.rcParams["savefig.dpi"] = args.dpi

    # Run each visualization
    import importlib

    for viz_name, (module_path, output_subdir) in tqdm(
        viz_to_run.items(), desc="Visualizations"
    ):
        logger.info("Generating: %s", viz_name)
        try:
            module = importlib.import_module(module_path)
            output_dir = FIGURES_DIR / output_subdir
            module.generate(activations_dir, output_dir)
        except Exception:
            logger.exception("Failed to generate %s", viz_name)

    logger.info("All visualizations saved to %s", FIGURES_DIR)

    # Print summary
    total_figures = sum(1 for _ in FIGURES_DIR.rglob("*.png"))
    logger.info("Generated %d figures total", total_figures)


if __name__ == "__main__":
    main()
