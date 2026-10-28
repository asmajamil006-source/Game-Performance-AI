"""Data loading, cleaning, validation, auditing and preprocessing.

The same cleaning and feature-engineering code path is used for training and
for the analytics pages, so the app never shows statistics from a different
version of the data than the model was trained on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as cfg
from utils import DataValidationError


# --------------------------------------------------------------------------
# Cleaning report
# --------------------------------------------------------------------------
@dataclass
class CleaningReport:
    """What the cleaning step actually did (all counts are measured)."""
    rows_loaded: int = 0
    rows_final: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    missing_target_removed: int = 0
    text_values_normalised: int = 0
    hardware_scores_repaired: int = 0
    missing_cells_before: int = 0
    notes: list[str] = field(default_factory=list)

    def as_table(self) -> pd.DataFrame:
        rows = [
            ("Rows loaded", self.rows_loaded),
            ("Exact duplicate rows removed", self.duplicates_removed),
            ("Rows removed: missing target (FPS)", self.missing_target_removed),
            ("Rows removed: invalid values", self.invalid_rows_removed),
            ("Text values normalised (case / whitespace)", self.text_values_normalised),
            ("Hardware scores restored from component name", self.hardware_scores_repaired),
            ("Missing cells before cleaning", self.missing_cells_before),
            ("Rows used", self.rows_final),
        ]
        return pd.DataFrame(rows, columns=["Step", "Count"])


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_dataset(path: Path = cfg.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and verify its structure. Raises DataValidationError."""
    path = Path(path)
    if not path.exists():
        raise DataValidationError(
            f"Dataset file '{path.name}' was not found. Place it in the project folder next to app.py."
        )
    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The dataset file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError("The dataset file is corrupted or is not a valid CSV file.") from exc
    except OSError as exc:
        raise DataValidationError(f"The dataset file could not be read ({exc.strerror}).") from exc

    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in cfg.REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataValidationError("The dataset is missing required columns: " + ", ".join(missing))
    if len(raw) < 100:
        raise DataValidationError(f"The dataset has only {len(raw)} rows; at least 100 are needed to train and evaluate models.")
    return raw


# --------------------------------------------------------------------------
# Feature engineering helpers
# --------------------------------------------------------------------------
_RES_P = re.compile(r"^(\d{3,4})p$")


def resolution_to_megapixels(label: object) -> float:
    """Convert a label like '1080p' or '4k' to megapixels (16:9). NaN if unknown."""
    text = str(label).strip().lower()
    if text in {"4k", "uhd"}:
        height = 2160
    else:
        match = _RES_P.match(text)
        if not match:
            return float("nan")
        height = int(match.group(1))
    width = round(height * 16 / 9)
    return width * height / 1e6


def score_gap_pct(cpu_score: float, gpu_score: float) -> float:
    """Dataset definition of Bottleneck_Pct: |GPU-CPU| / max(GPU, CPU) x 100."""
    top = max(cpu_score, gpu_score)
    return 0.0 if top <= 0 else abs(gpu_score - cpu_score) / top * 100.0


def _canonical_text(series: pd.Series) -> tuple[pd.Series, int]:
    """Trim/collapse whitespace and unify case variants to the most common spelling."""
    s = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    counts = s.dropna().value_counts()
    canonical: dict[str, str] = {}
    for value in counts.index:                      # most frequent spelling wins
        canonical.setdefault(value.lower(), value)
    out = s.map(lambda v: canonical.get(v.lower(), v) if pd.notna(v) else v)
    changed = int((out.fillna("") != series.astype("string").fillna("")).sum())
    return out, changed


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a clean, analysis-ready frame and a report of what was changed."""
    report = CleaningReport(rows_loaded=len(raw), missing_cells_before=int(raw.isna().sum().sum()))
    df = raw.copy()

    # 1. Text normalisation (case, whitespace). Resolution is lower-cased: '1080P' -> '1080p'.
    for col in cfg.TEXT_COLUMNS:
        df[col], changed = _canonical_text(df[col])
        report.text_values_normalised += changed
    res_lower = df["Resolution"].str.lower()
    report.text_values_normalised += int((res_lower.fillna("") != df["Resolution"].fillna("")).sum())
    df["Resolution"] = res_lower
    for col in cfg.TEXT_COLUMNS:                    # plain python str columns downstream
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    # 2. Numeric coercion: text such as 'N/A' or '12GB' becomes NaN, never a crash.
    for col in cfg.NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Exact duplicates.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report.duplicates_removed = before - len(df)

    # 4. Restore missing hardware scores from the component name (names map 1:1 to scores).
    for name_col, score_col in (("CPU", "CPU_Score"), ("GPU", "GPU_Score")):
        lookup = df.dropna(subset=[score_col]).groupby(name_col)[score_col].agg(lambda s: s.mode().iloc[0])
        gap = df[score_col].isna() & df[name_col].notna()
        fixed = df.loc[gap, name_col].map(lookup)
        report.hardware_scores_repaired += int(fixed.notna().sum())
        df.loc[gap, score_col] = fixed

    # 5. Rows without a target cannot be used for supervised learning.
    missing_target = df[cfg.TARGET].isna()
    report.missing_target_removed = int(missing_target.sum())
    df = df.loc[~missing_target]

    # 6. Invalid values: non-positive sizes/scores, impossible FPS, bad binary flags.
    invalid = pd.Series(False, index=df.index)
    for col in cfg.POSITIVE_COLUMNS:
        invalid |= df[col].notna() & (df[col] <= 0)
    invalid |= df[cfg.TARGET] > cfg.MAX_PLAUSIBLE_FPS
    invalid |= df["Ray_Tracing"].notna() & ~df["Ray_Tracing"].isin([0, 1])
    invalid |= df["Resolution"].map(resolution_to_megapixels).isna()
    invalid |= df[["CPU", "GPU", "Game"]].isna().any(axis=1)
    report.invalid_rows_removed = int(invalid.sum())
    df = df.loc[~invalid].reset_index(drop=True)

    if len(df) < 100:
        raise DataValidationError("Too few valid rows remain after cleaning to train a model.")

    # 7. Derived features (same function is used for prediction inputs).
    df["Resolution_MP"] = df["Resolution"].map(resolution_to_megapixels)
    df["Score_Gap_Pct"] = [score_gap_pct(c, g) for c, g in zip(df["CPU_Score"], df["GPU_Score"])]

    report.rows_final = len(df)
    return df, report


def load_clean_dataset(path: Path = cfg.DATA_PATH) -> tuple[pd.DataFrame, CleaningReport]:
    """Convenience: load + clean in one call."""
    return clean_dataset(load_dataset(path))


# --------------------------------------------------------------------------
# Catalog of supported hardware / options (what the UI is allowed to offer)
# --------------------------------------------------------------------------
def build_catalog(df: pd.DataFrame) -> dict:
    """Supported options, taken from the data so the UI can never offer unseen values."""
    def name_to_score(name_col: str, score_col: str) -> dict[str, float]:
        grouped = df.groupby(name_col)[score_col].agg(lambda s: float(s.mode().iloc[0]))
        return dict(sorted(grouped.items(), key=lambda kv: kv[1]))

    res_mp = df.groupby("Resolution")["Resolution_MP"].first().sort_values()
    return {
        "cpu": name_to_score("CPU", "CPU_Score"),
        "gpu": name_to_score("GPU", "GPU_Score"),
        "games": sorted(df["Game"].unique().tolist()),
        "resolutions": res_mp.index.tolist(),            # ascending pixel count
        "resolution_mp": res_mp.to_dict(),
        "ram_gb": sorted(int(v) for v in df["RAM_GB"].unique()),
        "vram_gb": sorted(int(v) for v in df["VRAM_GB"].unique()),
        "fps_min": float(df[cfg.TARGET].min()),
        "fps_max": float(df[cfg.TARGET].max()),
    }


# --------------------------------------------------------------------------
# Train / test split and preprocessing
# --------------------------------------------------------------------------
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproducible 80/20 split, stratified by GPU so every GPU appears in both parts."""
    return train_test_split(
        df, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=df["GPU"]
    )


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Imputation + scaling for numbers, imputation + one-hot for categories.

    * One-hot encoding is used for nominal variables (Game, OS, Storage_Type):
      label encoding would invent a numeric order between unrelated titles.
    * ``handle_unknown='ignore'`` keeps prediction safe if an unseen category slips in
      (the application layer rejects unsupported values before this point).
    * Scaling matters for linear models and is harmless for tree models.
    * Every transformer is fitted on the training data only (inside the Pipeline),
      so no information leaks from validation/test rows.
    """
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# --------------------------------------------------------------------------
# Data audit (quality checks performed BEFORE modelling)
# --------------------------------------------------------------------------
def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else 0.0


def audit_dataset(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Measured data-quality findings that drive the modelling decisions."""
    columns = []
    for col in cfg.REQUIRED_COLUMNS:
        series = df[col]
        numeric = pd.api.types.is_numeric_dtype(series)
        columns.append({
            "Column": col,
            "Type": "numeric" if numeric else "text",
            "Unique values": int(series.nunique()),
            "Missing (raw)": int(raw[col].isna().sum()),
            "Constant": bool(series.nunique() <= 1),
            "Range / examples": (f"{series.min():g} to {series.max():g}" if numeric
                                 else ", ".join(map(str, series.unique()[:4])) + (", ..." if series.nunique() > 4 else "")),
        })

    will_run = df["Will_Run"].value_counts().to_dict()
    quality_counts = df["Quality"].value_counts().reindex(cfg.QUALITY_LEVELS).fillna(0).astype(int)
    expected_gap = df[["CPU_Score", "GPU_Score"]].apply(lambda r: score_gap_pct(r.iloc[0], r.iloc[1]), axis=1)
    gap_error = float((expected_gap - df["Bottleneck_Pct"]).abs().max())
    name_score_consistent = bool(
        (df.groupby("CPU")["CPU_Score"].nunique() == 1).all() and (df.groupby("GPU")["GPU_Score"].nunique() == 1).all()
    )
    return {
        "n_rows": int(len(df)), "n_columns": int(df.shape[1]),
        "columns": pd.DataFrame(columns),
        "constant_columns": [c["Column"] for c in columns if c["Constant"]],
        "will_run_counts": {int(k): int(v) for k, v in will_run.items()},
        "playability_ml_supported": len(will_run) > 1,
        "quality_counts": quality_counts.to_dict(),
        "quality_balance_ratio": float(quality_counts.min() / quality_counts.max()) if quality_counts.max() else 0.0,
        "bottleneck_formula_max_error": gap_error,
        "hardware_name_score_one_to_one": name_score_consistent,
        "vram_gpu_cramers_v": _cramers_v(df["GPU"], df["VRAM_GB"]),
        "duplicates_in_raw": int(raw.duplicated().sum()),
        "fps_summary": df[cfg.TARGET].describe().to_dict(),
    }
