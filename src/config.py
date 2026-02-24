"""Central configuration for the lets-see-the-guts project."""

from pathlib import Path

# Model
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
FIGURES_DIR = OUTPUT_DIR / "figures"
PROMPTS_FILE = PROJECT_ROOT / "prompts.yaml"

# Model architecture (from SmolLM2-1.7B config.json)
NUM_LAYERS = 24
NUM_HEADS = 32
HIDDEN_DIM = 2048
INTERMEDIATE_SIZE = 8192
VOCAB_SIZE = 49152
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 64
RMS_NORM_EPS = 1e-5

# Capture settings
STORAGE_DTYPE = "float16"
LOGIT_LENS_TOP_K = 20
MAX_GENERATION_TOKENS = 50
