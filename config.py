"""
BlackboxNLP 2026 — Central Configuration
=========================================
All project-wide constants, model configs, paths, and hyperparameters.
Import this module everywhere instead of hard-coding values.

Usage:
    from config import CONFIG, MODELS, CONCEPTS, PATHS
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).parent.resolve()

@dataclass(frozen=True)
class ProjectPaths:
    """All project directory paths, auto-created on first access."""
    root: Path = PROJECT_ROOT

    # Data
    data: Path = PROJECT_ROOT / "data"
    prompt_pairs: Path = PROJECT_ROOT / "data" / "prompt_pairs"
    truthfulqa: Path = PROJECT_ROOT / "data" / "truthfulqa"
    advbench: Path = PROJECT_ROOT / "data" / "advbench"

    # Results
    results: Path = PROJECT_ROOT / "results"
    activations: Path = PROJECT_ROOT / "results" / "activations"
    directions: Path = PROJECT_ROOT / "results" / "directions"
    figures: Path = PROJECT_ROOT / "results" / "figures"

    # Paper
    paper: Path = PROJECT_ROOT / "paper"

    def ensure_all(self):
        """Create all directories if they don't exist."""
        for attr_name in vars(self):
            p = getattr(self, attr_name)
            if isinstance(p, Path) and attr_name != "root":
                p.mkdir(parents=True, exist_ok=True)


PATHS = ProjectPaths()


# ============================================================
# Model Configurations
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model."""
    name: str                          # Human-readable name
    hf_id: str                         # HuggingFace model ID
    transformer_lens_name: str         # TransformerLens model name (if supported)
    n_layers: int                      # Number of transformer layers
    d_model: int                       # Hidden dimension size
    is_instruct: bool = False          # Whether this is an instruction-tuned model
    quantization: Optional[str] = None # None, "4bit", or "8bit"
    max_seq_len: int = 512             # Max sequence length for extraction


# Pilot model (Phase 1)
GEMMA_2_2B_BASE = ModelConfig(
    name="Gemma 2 2B Base",
    hf_id="google/gemma-2-2b",
    transformer_lens_name="gemma-2-2b",
    n_layers=26,
    d_model=2304,
    is_instruct=False,
)

GEMMA_2_2B_INSTRUCT = ModelConfig(
    name="Gemma 2 2B Instruct",
    hf_id="google/gemma-2-2b-it",
    transformer_lens_name="gemma-2-2b-it",
    n_layers=26,
    d_model=2304,
    is_instruct=True,
)

# Primary study model (Phase 2)
QWEN_25_3B_INSTRUCT = ModelConfig(
    name="Qwen 2.5 3B Instruct",
    hf_id="Qwen/Qwen2.5-3B-Instruct",
    transformer_lens_name="Qwen/Qwen2.5-3B-Instruct",
    n_layers=36,
    d_model=2048,
    is_instruct=True,
    quantization="4bit",  # Use 4-bit to fit in Kaggle GPU memory
)

QWEN_25_3B_BASE = ModelConfig(
    name="Qwen 2.5 3B Base",
    hf_id="Qwen/Qwen2.5-3B",
    transformer_lens_name="Qwen/Qwen2.5-3B",
    n_layers=36,
    d_model=2048,
    is_instruct=False,
    quantization="4bit",
)

# ------------------------------------------------------------
# Replication sweep (Phase 3)
#
# Two orthogonal axes, so that a finding can be attributed to scale or to
# family rather than to "the model we happened to run":
#
#   Scale axis  — Qwen 2.5 at 0.5B / 1.5B / 3B / 7B, base and instruct.
#     Holds architecture, tokenizer, and pretraining recipe fixed and varies
#     only capacity. This is what the pilot non-replication (Section 5.5)
#     leaves open, so it is the axis that matters most.
#
#   Family axis — Gemma 2, Llama 3.2, Qwen 2.5 at comparable scale (~2-3B).
#     Holds capacity roughly fixed and varies architecture and data.
#
# Every entry is a base/instruct pair: the Base-vs-Instruct dispersion
# comparison is run within each pair, never across families.
# ------------------------------------------------------------

QWEN_25_05B_BASE = ModelConfig(
    name="Qwen 2.5 0.5B Base",
    hf_id="Qwen/Qwen2.5-0.5B",
    transformer_lens_name="Qwen/Qwen2.5-0.5B",
    n_layers=24,
    d_model=896,
    is_instruct=False,
)

QWEN_25_05B_INSTRUCT = ModelConfig(
    name="Qwen 2.5 0.5B Instruct",
    hf_id="Qwen/Qwen2.5-0.5B-Instruct",
    transformer_lens_name="Qwen/Qwen2.5-0.5B-Instruct",
    n_layers=24,
    d_model=896,
    is_instruct=True,
)

QWEN_25_15B_BASE = ModelConfig(
    name="Qwen 2.5 1.5B Base",
    hf_id="Qwen/Qwen2.5-1.5B",
    transformer_lens_name="Qwen/Qwen2.5-1.5B",
    n_layers=28,
    d_model=1536,
    is_instruct=False,
)

QWEN_25_15B_INSTRUCT = ModelConfig(
    name="Qwen 2.5 1.5B Instruct",
    hf_id="Qwen/Qwen2.5-1.5B-Instruct",
    transformer_lens_name="Qwen/Qwen2.5-1.5B-Instruct",
    n_layers=28,
    d_model=1536,
    is_instruct=True,
)

QWEN_25_7B_BASE = ModelConfig(
    name="Qwen 2.5 7B Base",
    hf_id="Qwen/Qwen2.5-7B",
    transformer_lens_name="Qwen/Qwen2.5-7B",
    n_layers=28,
    d_model=3584,
    is_instruct=False,
    quantization="4bit",
)

QWEN_25_7B_INSTRUCT = ModelConfig(
    name="Qwen 2.5 7B Instruct",
    hf_id="Qwen/Qwen2.5-7B-Instruct",
    transformer_lens_name="Qwen/Qwen2.5-7B-Instruct",
    n_layers=28,
    d_model=3584,
    is_instruct=True,
    quantization="4bit",
)

LLAMA_32_1B_BASE = ModelConfig(
    name="Llama 3.2 1B Base",
    hf_id="meta-llama/Llama-3.2-1B",
    transformer_lens_name="meta-llama/Llama-3.2-1B",
    n_layers=16,
    d_model=2048,
    is_instruct=False,
)

LLAMA_32_1B_INSTRUCT = ModelConfig(
    name="Llama 3.2 1B Instruct",
    hf_id="meta-llama/Llama-3.2-1B-Instruct",
    transformer_lens_name="meta-llama/Llama-3.2-1B-Instruct",
    n_layers=16,
    d_model=2048,
    is_instruct=True,
)

LLAMA_32_3B_BASE = ModelConfig(
    name="Llama 3.2 3B Base",
    hf_id="meta-llama/Llama-3.2-3B",
    transformer_lens_name="meta-llama/Llama-3.2-3B",
    n_layers=28,
    d_model=3072,
    is_instruct=False,
    quantization="4bit",
)

LLAMA_32_3B_INSTRUCT = ModelConfig(
    name="Llama 3.2 3B Instruct",
    hf_id="meta-llama/Llama-3.2-3B-Instruct",
    transformer_lens_name="meta-llama/Llama-3.2-3B-Instruct",
    n_layers=28,
    d_model=3072,
    is_instruct=True,
    quantization="4bit",
)

MODELS: Dict[str, ModelConfig] = {
    "gemma-2-2b": GEMMA_2_2B_BASE,
    "gemma-2-2b-it": GEMMA_2_2B_INSTRUCT,
    "qwen-2.5-3b-instruct": QWEN_25_3B_INSTRUCT,
    "qwen-2.5-3b": QWEN_25_3B_BASE,
    # Replication sweep
    "qwen-2.5-0.5b": QWEN_25_05B_BASE,
    "qwen-2.5-0.5b-instruct": QWEN_25_05B_INSTRUCT,
    "qwen-2.5-1.5b": QWEN_25_15B_BASE,
    "qwen-2.5-1.5b-instruct": QWEN_25_15B_INSTRUCT,
    "qwen-2.5-7b": QWEN_25_7B_BASE,
    "qwen-2.5-7b-instruct": QWEN_25_7B_INSTRUCT,
    "llama-3.2-1b": LLAMA_32_1B_BASE,
    "llama-3.2-1b-instruct": LLAMA_32_1B_INSTRUCT,
    "llama-3.2-3b": LLAMA_32_3B_BASE,
    "llama-3.2-3b-instruct": LLAMA_32_3B_INSTRUCT,
}

# Base -> Instruct pairs for the alignment comparison. Keys are base model
# names; the comparison is only ever run within a pair.
BASE_INSTRUCT_PAIRS: List[Tuple[str, str]] = [
    ("gemma-2-2b", "gemma-2-2b-it"),
    ("qwen-2.5-0.5b", "qwen-2.5-0.5b-instruct"),
    ("qwen-2.5-1.5b", "qwen-2.5-1.5b-instruct"),
    ("qwen-2.5-3b", "qwen-2.5-3b-instruct"),
    ("qwen-2.5-7b", "qwen-2.5-7b-instruct"),
    ("llama-3.2-1b", "llama-3.2-1b-instruct"),
    ("llama-3.2-3b", "llama-3.2-3b-instruct"),
]

# The two sweep axes, referenced by the replication driver and the paper.
SCALE_LADDER: List[str] = [
    "qwen-2.5-0.5b", "qwen-2.5-1.5b", "qwen-2.5-3b", "qwen-2.5-7b",
]
FAMILY_SET: List[str] = [
    "gemma-2-2b", "llama-3.2-3b", "qwen-2.5-3b",
]

# Default model for pilot runs
DEFAULT_MODEL = "gemma-2-2b-it"


# ============================================================
# Concept & Domain Definitions
# ============================================================

@dataclass(frozen=True)
class ConceptConfig:
    """Configuration for a single concept (e.g., refusal, honesty)."""
    name: str
    description: str                    # Operational definition
    domains: List[str]                  # Domain names
    target_pairs_per_domain: int = 200  # Target number of contrastive pairs

CONCEPTS: Dict[str, ConceptConfig] = {
    "refusal": ConceptConfig(
        name="refusal",
        description=(
            "Refusal is the model's behavior of declining to assist with a request "
            "that it classifies as harmful, dangerous, or policy-violating. A 'positive' "
            "example is a compliant response; a 'negative' example is a refusal response. "
            "The concept direction points from refusal toward compliance."
        ),
        domains=["violence", "illegal_activity", "medical_legal", "privacy"],
    ),
    "honesty": ConceptConfig(
        name="honesty",
        description=(
            "Honesty (truthfulness) is the model's tendency to provide factually correct "
            "and calibrated information vs. generating plausible-sounding falsehoods. "
            "A 'positive' example is a truthful response; a 'negative' example is a "
            "confident falsehood. The concept direction points from falsehood toward truth."
        ),
        domains=["factual_trivia", "math", "politics_opinion", "personal_advice"],
    ),
    "harmlessness": ConceptConfig(
        name="harmlessness",
        description=(
            "Harmlessness is the model's tendency to avoid generating content that could "
            "cause real-world harm, even when not explicitly asked to do so. Stretch goal "
            "concept — only pursue if Phase 2 bandwidth allows."
        ),
        domains=[],  # TBD — to be defined by Person A
        target_pairs_per_domain=100,
    ),
}


# ============================================================
# Extraction Hyperparameters
# ============================================================

@dataclass(frozen=True)
class ExtractionConfig:
    """Hyperparameters for activation extraction."""
    # Which layers to extract from (None = all layers)
    target_layers: Optional[List[int]] = None

    # Hook point: where in the residual stream to capture
    # Options: "resid_pre", "resid_post", "resid_mid"
    hook_point: str = "resid_post"

    # Token position to extract activations from
    # "last" = last token of the prompt (most common for causal LMs)
    token_position: str = "last"

    # Batch size for forward passes (adjust for GPU memory)
    batch_size: int = 4

    # Random seed for reproducibility
    seed: int = 42

    # Whether to use mixed precision
    use_fp16: bool = True


EXTRACTION = ExtractionConfig()


# ============================================================
# Analysis Hyperparameters
# ============================================================

@dataclass(frozen=True)
class AnalysisConfig:
    """Hyperparameters for direction computation and analysis."""
    # Number of bootstrap resamples for confidence intervals
    n_bootstrap: int = 1000

    # Significance level for confidence intervals
    ci_alpha: float = 0.05

    # Random seed
    seed: int = 42

    # Steering coefficient range to sweep
    steering_coeffs: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    )

    # Layer(s) to apply steering at (will be tuned)
    steering_layers: Optional[List[int]] = None

    # Number of PCA components to keep for visualization
    n_pca_components: int = 50  # Pre-UMAP dimensionality reduction

    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2


ANALYSIS = AnalysisConfig()


# ============================================================
# Visualization Settings
# ============================================================

@dataclass(frozen=True)
class VisualizationConfig:
    """Settings for publication-quality figures."""
    dpi: int = 300
    figure_format: str = "pdf"  # For ACL LaTeX
    font_family: str = "serif"  # ACL template uses serif fonts
    font_size: int = 10
    title_size: int = 12
    label_size: int = 10

    # Colorblind-safe palette (Wong palette)
    colors: List[str] = field(default_factory=lambda: [
        "#0072B2",  # Blue
        "#D55E00",  # Vermillion
        "#009E73",  # Bluish green
        "#CC79A7",  # Reddish purple
        "#F0E442",  # Yellow
        "#56B4E9",  # Sky blue
        "#E69F00",  # Orange
    ])

    # Figure sizes (width, height) in inches — ACL column width is ~3.25 in
    single_column: tuple = (3.25, 2.5)
    double_column: tuple = (6.75, 4.0)
    square: tuple = (3.25, 3.25)


VISUALIZATION = VisualizationConfig()


# ============================================================
# Naming Conventions
# ============================================================

def activation_filename(model: str, concept: str, domain: str, layer: int) -> str:
    """Generate standardized filename for cached activations.

    Format: {model}_{concept}_{domain}_layer{layer:03d}.pt
    Example: qwen25_3b_instruct_refusal_violence_layer012.pt
    """
    model_short = model.replace("-", "_").replace(".", "")
    return f"{model_short}_{concept}_{domain}_layer{layer:03d}.pt"


def direction_filename(model: str, concept: str, domain: str, layer: int) -> str:
    """Generate standardized filename for direction vectors.

    Format: dir_{model}_{concept}_{domain}_layer{layer:03d}.pt
    Use domain="global" for the global (all-domain) direction.
    """
    model_short = model.replace("-", "_").replace(".", "")
    return f"dir_{model_short}_{concept}_{domain}_layer{layer:03d}.pt"


# ============================================================
# Aggregate Config
# ============================================================

CONFIG = {
    "paths": PATHS,
    "models": MODELS,
    "concepts": CONCEPTS,
    "extraction": EXTRACTION,
    "analysis": ANALYSIS,
    "visualization": VISUALIZATION,
    "default_model": DEFAULT_MODEL,
}
