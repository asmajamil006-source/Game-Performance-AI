"""Central configuration for the AI Game Performance Predictor.

Everything that is a *decision* (thresholds, feature sets, documented
assumptions) lives here so it is visible, reviewable and easy to defend in a
viva.  Nothing in this file is learned from data; learned values live in the
trained model bundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (relative to the project folder, never machine specific)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "game_performance_5000.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "fps_model_bundle.joblib"

# Bump when preprocessing / training logic changes so stale artifacts are rebuilt.
BUNDLE_SCHEMA_VERSION = 3

# --------------------------------------------------------------------------
# Reproducibility and evaluation protocol
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20          # held-out test set, never used for model selection
CV_FOLDS = 5              # cross-validation on the training split only
INTERVAL_COVERAGE = 0.80  # nominal coverage of the prediction interval
PERMUTATION_REPEATS = 30
SHAP_BACKGROUND_SIZE = 150

# --------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------
TARGET = "FPS"
REQUIRED_COLUMNS = [
    "CPU", "CPU_Score", "GPU", "GPU_Score", "RAM_GB", "VRAM_GB", "Game",
    "Resolution", "FPS", "Quality", "Will_Run", "Storage_Type", "PSU_Watts",
    "OS", "Ray_Tracing", "Bottleneck_Pct", "Load_Time_Sec", "Driver_Version",
]
TEXT_COLUMNS = ["CPU", "GPU", "Game", "Resolution", "Quality", "Storage_Type", "OS"]
NUMERIC_COLUMNS = [
    "CPU_Score", "GPU_Score", "RAM_GB", "VRAM_GB", "FPS", "Will_Run",
    "PSU_Watts", "Ray_Tracing", "Bottleneck_Pct", "Load_Time_Sec", "Driver_Version",
]
# Columns that must be strictly positive to be physically meaningful.
POSITIVE_COLUMNS = ["CPU_Score", "GPU_Score", "RAM_GB", "VRAM_GB", "FPS", "PSU_Watts"]
MAX_PLAUSIBLE_FPS = 1000   # anything above is a data-entry error, not a measurement
QUALITY_LEVELS = ["Low", "Medium", "High", "Ultra"]


@dataclass(frozen=True)
class FeatureSet:
    """A named group of model input features (post feature-engineering names)."""
    key: str
    label: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...] = field(default_factory=tuple)

    @property
    def columns(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)


# Two feature sets are compared for every algorithm.  "core" contains only the
# hardware capability scores; "full" adds every user-facing configuration
# option.  Selection between them is made on validation results (see
# model_engine.select_best_model), not by assumption.
FEATURE_SETS: dict[str, FeatureSet] = {
    "core": FeatureSet("core", "Core hardware (GPU + CPU score)", ("GPU_Score", "CPU_Score")),
    "full": FeatureSet(
        "full", "Full configuration",
        ("GPU_Score", "CPU_Score", "RAM_GB", "VRAM_GB", "Ray_Tracing", "Resolution_MP"),
        ("Game",),
    ),
}

# Every column that is knowable *before* a game is launched.  Used for the
# feature-screening experiment (permutation importance + residual tests).
SCREENING_NUMERIC = [
    "GPU_Score", "CPU_Score", "RAM_GB", "VRAM_GB", "Ray_Tracing",
    "Resolution_MP", "PSU_Watts", "Driver_Version",
]
SCREENING_CATEGORICAL = ["Game", "Storage_Type", "OS"]
# Informational-only columns tested for association but never used as inputs.
INFORMATIONAL_NUMERIC = ["Bottleneck_Pct", "Load_Time_Sec"]
INFORMATIONAL_CATEGORICAL = ["Quality"]

# --------------------------------------------------------------------------
# Documented feature decisions (static reasoning; the *evidence* shown next to
# it in the UI is computed from the data at training time).
# --------------------------------------------------------------------------
FEATURE_DECISIONS: list[dict[str, str]] = [
    {"column": "FPS", "role": "Target", "reason": "Measured frames per second; the quantity being predicted."},
    {"column": "GPU_Score", "role": "Input", "reason": "Numeric GPU benchmark score. Replaces the GPU name (one-to-one mapping verified)."},
    {"column": "CPU_Score", "role": "Input", "reason": "Numeric CPU benchmark score. Replaces the CPU name (one-to-one mapping verified)."},
    {"column": "GPU", "role": "Identifier", "reason": "Hardware name used for dropdowns; encoded by its score so no false ordering is created."},
    {"column": "CPU", "role": "Identifier", "reason": "Hardware name used for dropdowns; encoded by its score so no false ordering is created."},
    {"column": "RAM_GB", "role": "Input (screened)", "reason": "Candidate input available before launch."},
    {"column": "VRAM_GB", "role": "Input (screened)", "reason": "Candidate input. In this dataset VRAM is not tied to the GPU model (e.g. a GTX 1660 appears with 12 GB), a sign of synthetic generation."},
    {"column": "Game", "role": "Input (screened)", "reason": "Nominal category, one-hot encoded. Label encoding is avoided because it would impose a false numeric order on game titles."},
    {"column": "Resolution", "role": "Derived input", "reason": "Converted to megapixels (Resolution_MP), a physically meaningful numeric quantity, instead of an arbitrary 0-3 code."},
    {"column": "Ray_Tracing", "role": "Input (screened)", "reason": "Binary in-game setting available before launch."},
    {"column": "Storage_Type", "role": "Input (screened)", "reason": "Nominal category, one-hot encoded; tested but expected to affect load time rather than FPS."},
    {"column": "OS", "role": "Input (screened)", "reason": "Nominal category, one-hot encoded."},
    {"column": "PSU_Watts", "role": "Input (screened)", "reason": "Power supply rating; tested for association."},
    {"column": "Driver_Version", "role": "Input (screened)", "reason": "Near-unique continuous value (4,197 distinct values); tested, behaves like noise."},
    {"column": "Bottleneck_Pct", "role": "Derived (not an input)", "reason": "Exactly equals |GPU-CPU| / max(GPU, CPU) x 100, so it is recomputed from the two scores rather than used as an independent feature."},
    {"column": "Load_Time_Sec", "role": "Excluded", "reason": "Measured while/after the game runs, so it is not known at prediction time (leakage risk)."},
    {"column": "Quality", "role": "Excluded", "reason": "Output of the recommender, not an input. A learnability experiment shows it carries no signal in this dataset."},
    {"column": "Will_Run", "role": "Excluded", "reason": "Constant column (every row = 1). A classifier would have a single class and cannot learn anything."},
]

# --------------------------------------------------------------------------
# Playability tiers: a *derived target* defined from predicted FPS.
# Thresholds follow common gaming conventions (30 = minimum acceptable motion,
# 60 = smooth, 90+ = high-refresh).  They are definitions, not learned values.
# --------------------------------------------------------------------------
PERFORMANCE_TIERS: list[tuple[str, float, str, str]] = [
    # (label, min_fps, colour, description)
    ("Excellent", 90.0, "#059669", "High-refresh gameplay with plenty of headroom."),
    ("Very Good", 60.0, "#2563EB", "Smooth 60+ FPS gameplay."),
    ("Playable", 45.0, "#0891B2", "Comfortably playable; occasional dips below 60 FPS."),
    ("Low Performance", 30.0, "#D97706", "Playable at the 30 FPS minimum but noticeably choppy."),
    ("Not Recommended", 0.0, "#DC2626", "Below 30 FPS; gameplay will feel unplayable."),
]
TIER_LABELS_ASCENDING = [t[0] for t in reversed(PERFORMANCE_TIERS)]

# Hardware balance: pairs whose scores differ by less than this are "balanced".
# Presentation convention, not a learned value.
BALANCED_THRESHOLD_PCT = 10.0

# Confidence labels, based on the probability that the actual FPS falls on the
# same side of the user's target FPS as the prediction does.
RELIABILITY_HIGH = 0.75
RELIABILITY_MODERATE = 0.55

# --------------------------------------------------------------------------
# Graphics-quality rule engine (RULE-BASED, labelled as such in the UI).
# The dataset's Quality column has no learnable signal, so the recommendation
# is a transparent hybrid: the ML model supplies predicted FPS, and explicit
# rules decide how much of the frame-rate headroom can be spent on quality.
# All numbers below are documented *assumptions*, not data-derived values.
# --------------------------------------------------------------------------
# Assumed FPS cost of each preset relative to the target: a preset is
# recommended only if predicted FPS >= target x multiplier.
QUALITY_HEADROOM_MULTIPLIER = {"Low": 0.0, "Medium": 1.00, "High": 1.25, "Ultra": 1.50}
# General-purpose VRAM guidelines (GB) for texture-heavy presets at a
# ~1080p-class pixel budget; scaled by resolution in recommendations.py.
VRAM_GUIDELINE_GB = {"Low": 0, "Medium": 4, "High": 6, "Ultra": 8}
# Extra VRAM guideline when rendering well above/below 1080p.
VRAM_RES_ADJUST_GB = {"low": -2, "mid": 0, "high": 2}   # <1.5 MP, 1.5-2.6 MP, >2.6 MP
RAM_GUIDELINE_GB = {"Low": 0, "Medium": 8, "High": 16, "Ultra": 16}
DEFAULT_TARGET_FPS = 60
TARGET_FPS_OPTIONS = [30, 45, 60, 90]

# What-if analysis: only suggest an upgrade if the model predicts at least this gain.
MIN_MEANINGFUL_FPS_GAIN = 3.0

# Impact labels for explanations (absolute FPS contribution).
IMPACT_THRESHOLDS_FPS = [(5.0, "High"), (2.0, "Medium"), (0.5, "Low")]

# --------------------------------------------------------------------------
# UI palette (single source of truth for CSS and charts)
# --------------------------------------------------------------------------
PALETTE = {
    "primary": "#2563EB", "primary_dark": "#1E3A8A", "accent": "#0D9488",
    "ink": "#0F172A", "muted": "#64748B", "line": "#E2E8F0", "bg": "#F6F8FB",
    "card": "#FFFFFF", "good": "#059669", "warn": "#D97706", "bad": "#DC2626",
    "soft_blue": "#DBEAFE", "soft_grey": "#CBD5E1",
}
