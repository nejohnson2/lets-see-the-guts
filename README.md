# lets-see-the-guts

Capture and visualize the internal activations of an LLM during inference. This project hooks into every layer of **SmolLM2-1.7B** (a modern Llama-style transformer) as it processes prompts and generates text, saves all activation data as numpy arrays, and produces annotated matplotlib visualizations that explain what's happening inside the model.

## Why This Exists

Large language models are black boxes — you feed in text and get text out. But internally, information flows through dozens of transformer layers, each with attention heads that decide which tokens to focus on and MLP blocks that transform representations. This project makes that invisible processing visible by:

1. **Capturing activations** at every layer during both prompt encoding and token generation
2. **Saving raw data** so you can explore the internals yourself
3. **Generating annotated visualizations** that explain what each view reveals about the model

## Model

**SmolLM2-1.7B** (`HuggingFaceTB/SmolLM2-1.7B`) — a 2024 Llama-architecture model small enough to run on a laptop but architecturally identical to larger LLMs:

| Parameter | Value |
|-----------|-------|
| Layers | 24 |
| Attention Heads | 32 |
| Hidden Dimension | 2,048 |
| MLP Intermediate Size | 8,192 |
| Vocabulary | 49,152 tokens |
| Architecture | Llama (RoPE, SiLU, RMSNorm, gated MLP) |

The model is loaded with `attn_implementation="eager"` because SDPA/Flash Attention do not expose attention weight matrices.

## Setup

Requires Python 3.11+.

```bash
# Create virtual environment and install dependencies
make setup

# First run downloads the model (~3.4 GB)
```

### Dependencies

- `torch` — model execution and hook API
- `transformers` — model loading from HuggingFace
- `numpy` — activation storage and manipulation
- `matplotlib` — all visualizations
- `scikit-learn` — PCA for dimensionality reduction
- `PyYAML` — prompt configuration
- `tqdm` — progress tracking

## Usage

### Quick Test (Single Prompt)

```bash
make dev
```

Captures activations for the first prompt ("Hello world") with 10 generation tokens, then generates all visualizations. Good for verifying everything works.

### Full Pipeline

```bash
make all
```

Runs capture for all 5 prompts, then generates all 7 visualization types. Takes several minutes depending on hardware.

### Individual Steps

```bash
# Capture only
python run_capture.py

# Capture a single prompt
python run_capture.py --prompt 2

# Override generation length
python run_capture.py --max-gen-tokens 20

# Visualize only (requires prior capture)
python run_visualize.py

# Generate a single visualization type
python run_visualize.py --viz attention

# High-DPI for publication
python run_visualize.py --dpi 300
```

### Custom Prompts

Edit `prompts.yaml` or create your own:

```yaml
prompts:
  - id: "my_prompt"
    text: "The meaning of life is"
    description: "Philosophical completion"
    max_gen_tokens: 40
```

```bash
python run_capture.py --prompts-file my_prompts.yaml
```

## What Gets Captured

During a forward pass, PyTorch forward hooks intercept activations at every layer. All data is saved as float16 numpy arrays in `output/activations/`.

### Prefill Phase (Prompt Encoding)

The full prompt is processed in a single forward pass. Captured data:

| File | Shape | Description |
|------|-------|-------------|
| `residual_stream.npy` | [25, S, 2048] | Hidden state at each layer boundary (embed + 24 layers) |
| `attention_weights.npy` | [24, 32, S, S] | Attention matrix for every head at every layer |
| `mlp_gate_pre_act.npy` | [24, S, 8192] | MLP gate values before SiLU activation |
| `mlp_outputs.npy` | [24, S, 2048] | Output of each MLP block |
| `logit_lens_topk.npz` | [25, S, 20] | Top-20 predicted tokens at each layer (logit lens) |
| `metadata.json` | — | Prompt text, token IDs, token strings |

Where S = number of tokens in the prompt.

### Generation Phase (Token-by-Token)

Each generated token gets its own set of activation files in `generation/`:

| File | Shape | Description |
|------|-------|-------------|
| `step_NNN_residual.npy` | [25, 1, 2048] | Residual stream for the single new token |
| `step_NNN_attention.npy` | [24, 32, 1, S+N] | Attention from new token to all prior tokens |
| `generated_tokens.json` | — | Token IDs, strings, and step indices |

## Visualizations

All figures are saved to `output/figures/` with detailed annotations explaining what you're looking at.

### Key Concept: L2 Norm

Many of the visualizations below use the **L2 norm** (Euclidean norm) to summarize a hidden state vector as a single number. The L2 norm is the "length" of a vector — the straight-line distance from the origin to that point in space:

```
L2 norm = sqrt(h₁² + h₂² + ... + hₙ²)
```

Each hidden state in SmolLM2 is a 2048-dimensional vector. Its L2 norm collapses those 2048 values into one number representing the overall **magnitude** of the representation. A hidden state with norm 50 has a much "stronger" or more amplified representation than one with norm 10, even though both live in the same 2048-d space. When the norm grows across layers, it means successive layers are adding substantial contributions to the residual stream — the representation is being built up.

### 1. Activation Heatmap

**File:** `output/figures/01_activation_heatmap/`

A 2D heatmap where each cell shows the L2 norm of the hidden state at a (layer, token) coordinate. Brighter = larger magnitude. Reveals which tokens the model considers important at each processing stage.

**What to look for:** Vertical bright stripes (tokens important across many layers), horizontal gradients (norm growth through the network), differences between content words and function words.

### 2. Attention Patterns

**File:** `output/figures/02_attention_patterns/`

Attention weight matrices for selected heads at early, middle, and late layers. Each matrix shows how much each token (row) attends to every other token (column). Heads are selected for entropy diversity — from most focused to most distributed.

**What to look for:** Diagonal patterns (self-attention), vertical stripes (all tokens attending to one important token), banded patterns (local/syntactic attention), sparse patterns (learned semantic relationships). Early layers tend toward positional patterns; late layers toward semantic.

### 3. Residual Stream Norms

**File:** `output/figures/03_residual_norms/`

Line plot of hidden state L2 norm vs. layer index, one line per token. Shows how the magnitude of each token's representation grows as layers add to the residual stream.

**What to look for:** Growth rate (steep jumps = high-impact layers), differences between tokens (content words may grow differently than function words), overall pattern shape.

### 4. MLP Gate Analysis

**File:** `output/figures/04_mlp_gates/`

Two figures per prompt:
- **Sparsity analysis:** What fraction of the 8,192 MLP features are near zero at each layer. Reveals how selective the MLP is.
- **Top features:** Heatmap of the 50 most-activated MLP features for the last token across all layers.

**What to look for:** High sparsity (most features suppressed — the model is very selective), which layers are most active, whether different tokens trigger different feature sets.

### 5. Token Prediction Evolution (Logit Lens)

**File:** `output/figures/05_token_prediction/`

The "logit lens" technique: at each layer, project the hidden state through the final output head to see what token the model would predict if processing stopped there.

Two figures per prompt:
- **Last-position detail:** Top-5 predictions at every layer for the final token position (the actual next-token prediction).
- **All-positions summary:** Top-1 prediction at every (layer, position) cell.

**What to look for:** Early layers predict generic high-frequency tokens ("the", "a"). Later layers refine toward the correct answer. The layer where the right answer first appears reveals where that knowledge lives in the network. Factual prompts ("The capital of France is") show the most dramatic prediction shifts.

### 6. Cross-Prompt Comparison

**File:** `output/figures/06_cross_prompt/`

Side-by-side comparison of all prompts on four metrics:
- Residual stream norm growth (last token)
- Average attention entropy per layer
- MLP gate sparsity per layer
- Prediction confidence per layer (logit lens)

**What to look for:** Do factual prompts cause sharper attention than creative prompts? Do some prompts require more MLP computation? How quickly does each prompt reach high prediction confidence?

### 7. PCA Dimensionality Reduction

**File:** `output/figures/07_dimensionality/`

PCA projects the 2048-dimensional hidden states into 2D:
- **Left panel:** All hidden states colored by layer depth (blue=early, red=late).
- **Right panel:** Each token's trajectory through layers (circle=embedding, star=final layer).

**What to look for:** Distance between early and late layer clusters (large = big representational change), token trajectories (long paths = significant transformation), phase transitions (sudden jumps between adjacent layers).

## Working with Raw Data

All activation data is standard numpy arrays. Load them for your own analysis:

```python
import numpy as np
import json

# Load prefill activations
prompt_dir = "output/activations/prompt_capital_france"

residual = np.load(f"{prompt_dir}/residual_stream.npy")
# Shape: [25, num_tokens, 2048] — 25 = embed + 24 layers
print(f"Residual stream shape: {residual.shape}")

attention = np.load(f"{prompt_dir}/attention_weights.npy")
# Shape: [24, 32, num_tokens, num_tokens] — 24 layers, 32 heads
print(f"Attention shape: {attention.shape}")

mlp_gates = np.load(f"{prompt_dir}/mlp_gate_pre_act.npy")
# Shape: [24, num_tokens, 8192] — raw gate values (apply SiLU yourself)
print(f"MLP gates shape: {mlp_gates.shape}")

# Logit lens
logit_lens = np.load(f"{prompt_dir}/logit_lens_topk.npz")
top_indices = logit_lens["top_k_indices"]  # [25, num_tokens, 20]
top_probs = logit_lens["top_k_probs"]      # [25, num_tokens, 20]

# Metadata (prompt text, token strings, generated text)
with open(f"{prompt_dir}/metadata.json") as f:
    meta = json.load(f)
print(f"Tokens: {meta['token_strings']}")
print(f"Generated: {meta.get('generated_text', 'N/A')}")

# Model weights for recomputing logit lens
norm_weight = np.load("output/activations/model_weights/final_norm_weight.npy")
embed_weight = np.load("output/activations/model_weights/embed_tokens_weight.npy")
```

### Applying SiLU to Gate Activations

The saved `mlp_gate_pre_act.npy` contains the raw linear output before the SiLU activation. To get the actual gate values used by the model:

```python
def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float32))))

gate_activated = silu(mlp_gates)
```

### Recomputing Logit Lens

If you want different top-k or want to examine specific token predictions:

```python
from src.logit_lens import compute_logit_lens

result = compute_logit_lens(residual, norm_weight, embed_weight, top_k=50)
```

## Project Structure

```
lets-see-the-guts/
├── run_capture.py              # Entry point: capture activations
├── run_visualize.py            # Entry point: generate visualizations
├── prompts.yaml                # Prompt definitions
├── Makefile                    # setup, capture, visualize, dev, clean
├── requirements.txt
├── src/
│   ├── config.py               # Model ID, paths, architecture constants
│   ├── device.py               # MPS/CUDA/CPU auto-detection
│   ├── model.py                # Load model with eager attention
│   ├── hooks.py                # HookManager + ActivationStore
│   ├── capture.py              # Prefill + generation capture orchestrator
│   ├── logit_lens.py           # Hidden state → token prediction projection
│   └── storage.py              # Save/load numpy arrays and metadata
├── viz/
│   ├── common.py               # Shared plotting utilities
│   ├── activation_heatmap.py   # Viz 1: L2 norm heatmap
│   ├── attention_patterns.py   # Viz 2: Per-head attention matrices
│   ├── residual_norms.py       # Viz 3: Norm growth line plots
│   ├── mlp_gates.py            # Viz 4: Gate sparsity analysis
│   ├── token_prediction.py     # Viz 5: Logit lens prediction evolution
│   ├── cross_prompt.py         # Viz 6: Cross-prompt comparison
│   └── dimensionality.py       # Viz 7: PCA of hidden states
└── output/                     # Generated at runtime (gitignored)
    ├── activations/            # Numpy arrays (~8 MB per prompt)
    └── figures/                # PNG visualizations
```

## Architecture Notes

### How Hooks Work

PyTorch's `register_forward_hook(callback)` attaches a function to a module that fires every time data flows through it. The `HookManager` registers hooks on:

- `model.model.embed_tokens` — captures the initial token embeddings
- `model.model.layers[i]` — captures the residual stream after each layer
- `model.model.layers[i].self_attn` — captures attention weight matrices
- `model.model.layers[i].mlp.gate_proj` — captures MLP gate activations
- `model.model.layers[i].mlp` — captures full MLP output

Each hook callback immediately detaches the tensor from the computation graph, moves it to CPU, and converts to float16. This prevents activation storage from competing with the model for GPU/MPS memory.

### Two-Phase Capture

1. **Prefill:** The full prompt is processed in one forward pass. All tokens attend to all previous tokens simultaneously. This produces the familiar square attention matrices.

2. **Generation:** Tokens are generated one at a time using a manual loop (not `model.generate()`). Each step uses the KV cache from previous steps, so the new token's attention is a `[1, S+N]` vector (one query attending to all prior keys). This gives us per-step activation snapshots.

### Device Compatibility

- **MPS (Apple Silicon):** Model loaded in float32 (~6.8 GB). MPS does not support bfloat16.
- **CUDA:** Model loaded in bfloat16 (~3.4 GB).
- **CPU:** Model loaded in float32. Slower but works.

All activations are saved as float16 numpy arrays regardless of device.

## Default Prompts

| Prompt | Purpose |
|--------|---------|
| "Hello world" | Simple greeting — baseline language patterns |
| "Eat soup" | Action phrase — verb-noun relationships |
| "The capital of France is" | Factual completion — knowledge retrieval |
| "Once upon a time" | Narrative start — creative continuation |
| "To solve this math problem" | Reasoning — analytical processing patterns |

## Troubleshooting

**First run is slow:** The model (~3.4 GB) is downloaded from HuggingFace on first run. Subsequent runs use the local cache.

**Out of memory:** SmolLM2-1.7B needs ~7 GB in float32 (MPS) or ~4 GB in bfloat16 (CUDA), plus ~2 GB overhead. If memory is tight, run one prompt at a time with `--prompt 0`.

**No attention weights captured:** Ensure `attn_implementation="eager"` is set in model loading. SDPA and Flash Attention do not return attention weights.

**`make clean`:** Deletes all captured data and figures. You'll need to re-run capture.
