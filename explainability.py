"""Model-agnostic explanations computed from the trained model itself.

* Global: permutation importance (drop in R2 when a feature is shuffled).
* Local:  exact interventional Shapley values.  For one prediction every subset
  of features is "switched on" (taking the user's value) or "off" (taking values
  from a background sample of real training rows); the model is evaluated for each
  combination and the contributions are averaged with the Shapley weights.
  With <= 12 features this is exact (no sampling, no extra library), and the
  contributions always sum to (prediction - baseline).
"""
from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config as cfg

MAX_EXACT_FEATURES = 12


def shapley_values(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    row: pd.DataFrame,
    background: pd.DataFrame,
    features: list[str],
) -> tuple[float, dict[str, float], float]:
    """Return (baseline, {feature: contribution}, prediction) for a one-row frame."""
    p = len(features)
    if p == 0 or p > MAX_EXACT_FEATURES:
        raise ValueError(f"Exact Shapley needs between 1 and {MAX_EXACT_FEATURES} features (got {p}).")
    background = background.reset_index(drop=True)

    value: dict[frozenset, float] = {}
    for size in range(p + 1):
        for subset in combinations(range(p), size):
            frame = background.copy()
            for idx in subset:
                frame[features[idx]] = row[features[idx]].iloc[0]
            value[frozenset(subset)] = float(np.mean(predict_fn(frame)))

    phi = {}
    for i in range(p):
        total = 0.0
        others = [j for j in range(p) if j != i]
        for size in range(p):
            weight = factorial(size) * factorial(p - size - 1) / factorial(p)
            for subset in combinations(others, size):
                s = frozenset(subset)
                total += weight * (value[s | {i}] - value[s])
        phi[features[i]] = total
    return value[frozenset()], phi, value[frozenset(range(p))]


def permutation_importance_table(
    model, X: pd.DataFrame, y: pd.Series, features: list[str],
    n_repeats: int = cfg.PERMUTATION_REPEATS, seed: int = cfg.RANDOM_STATE,
) -> pd.DataFrame:
    """Permutation importance on held-out data, measured as the drop in R2."""
    result = permutation_importance(
        model, X[features], y, scoring="r2", n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    table = pd.DataFrame({
        "Feature": features,
        "Importance (drop in R2)": result.importances_mean,
        "Std": result.importances_std,
    })
    return table.sort_values("Importance (drop in R2)", ascending=False).reset_index(drop=True)
