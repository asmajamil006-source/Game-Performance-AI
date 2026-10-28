"""Training, evaluation, selection and prediction for FPS.

Pipeline
--------
1. Split into train/test (test is untouched until the final check).
2. Compare 6 regression algorithms x 2 feature sets with 5-fold CV on the
   training split, plus a mean-prediction baseline.
3. Select the winner on CV R2 using the one-standard-error rule: among models
   statistically indistinguishable from the best, prefer the simplest.
4. Evaluate the selected model once on the test set.
5. Build empirical prediction intervals from out-of-fold residuals and verify
   their coverage on the test set.
6. Run supporting experiments (feature screening, playability-tier agreement,
   Quality learnability) so every design decision is backed by measured evidence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config as cfg
import data_processor as dp
from explainability import permutation_importance_table, shapley_values
from utils import GamePerfError, InvalidInputError, ModelUnavailableError, get_logger, impact_label

log = get_logger()


# --------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------
def regression_candidates() -> dict[str, tuple[object, int]]:
    """name -> (estimator, complexity rank).  Lower rank = simpler model.

    Tree ensembles are regularised (large leaves, shallow depth) because the
    dataset is small and noisy; unconstrained trees would simply memorise noise.
    """
    rs = cfg.RANDOM_STATE
    return {
        "Linear Regression": (LinearRegression(), 0),
        "Ridge Regression": (RidgeCV(alphas=np.logspace(-3, 3, 13)), 1),
        "Random Forest": (RandomForestRegressor(n_estimators=150, min_samples_leaf=15, n_jobs=-1, random_state=rs), 3),
        "Extra Trees": (ExtraTreesRegressor(n_estimators=150, min_samples_leaf=15, n_jobs=-1, random_state=rs), 3),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=rs), 4),
        "HistGradientBoosting": (HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_depth=3, min_samples_leaf=40, random_state=rs), 4),
    }


def make_pipeline(feature_set: cfg.FeatureSet, estimator) -> Pipeline:
    return Pipeline([
        ("prep", dp.build_preprocessor(list(feature_set.numeric), list(feature_set.categorical))),
        ("model", clone(estimator)),
    ])


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# Model comparison and selection
# --------------------------------------------------------------------------
def compare_regressors(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Cross-validate every algorithm x feature-set; also report held-out test scores."""
    cv = KFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error"}
    rows = []
    entries = [("Baseline (always predict mean)", DummyRegressor(strategy="mean"), -1)]
    entries += [(n, est, rank) for n, (est, rank) in regression_candidates().items()]
    for name, estimator, rank in entries:
        for fs in cfg.FEATURE_SETS.values():
            if rank == -1 and fs.key != "core":
                continue
            pipe = make_pipeline(fs, estimator)
            start = time.perf_counter()
            res = cross_validate(pipe, train, train[cfg.TARGET], cv=cv, scoring=scoring, n_jobs=1)
            pipe.fit(train, train[cfg.TARGET])
            elapsed = time.perf_counter() - start
            test_m = regression_metrics(test[cfg.TARGET], pipe.predict(test))
            rows.append({
                "Model": name, "Features": fs.label, "feature_key": fs.key, "n_features": len(fs.columns),
                "complexity": rank,
                "CV R2": res["test_r2"].mean(), "CV R2 std": res["test_r2"].std(),
                "CV MAE": -res["test_mae"].mean(), "CV RMSE": float(np.sqrt(-res["test_mse"].mean())),
                "Test MAE": test_m["MAE"], "Test MSE": test_m["MSE"], "Test RMSE": test_m["RMSE"], "Test R2": test_m["R2"],
                "Time (s)": elapsed,
            })
            log.info("evaluated %-22s %-34s CV R2=%.4f", name, fs.label, rows[-1]["CV R2"])
    return pd.DataFrame(rows)


def select_best_model(table: pd.DataFrame) -> tuple[int, str]:
    """One-standard-error rule on CV R2, then prefer fewer features and simpler models.

    Returns (row index in ``table``, plain-language justification generated from the numbers).
    """
    models = table[table["complexity"] >= 0]
    best_row = models.loc[models["CV R2"].idxmax()]
    se = best_row["CV R2 std"] / np.sqrt(cfg.CV_FOLDS)
    eligible = models[models["CV R2"] >= best_row["CV R2"] - se]
    chosen = eligible.sort_values(["n_features", "complexity", "CV R2"], ascending=[True, True, False]).iloc[0]
    baseline = table[table["complexity"] < 0].iloc[0]
    reason = (
        f"{chosen['Model']} on '{chosen['Features']}' reaches CV R2 = {chosen['CV R2']:.3f} "
        f"(best observed: {best_row['Model']}, {best_row['CV R2']:.3f}). "
        f"{len(eligible)} of {len(models)} candidates lie within one standard error ({se:.4f}) of the best, "
        f"so they are statistically indistinguishable; the simplest of them "
        f"({int(chosen['n_features'])} features, lowest model complexity) was chosen. "
        f"The mean-prediction baseline scores R2 = {baseline['CV R2']:.3f}, confirming the model learns real structure."
    )
    return int(chosen.name), reason


# --------------------------------------------------------------------------
# Feature screening: which inputs carry measurable signal?
# --------------------------------------------------------------------------
def screen_features(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Combine permutation importance (with a random-noise reference) and residual tests."""
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    tr, te = train.copy(), test.copy()
    tr["Noise_Baseline"], te["Noise_Baseline"] = rng.normal(size=len(tr)), rng.normal(size=len(te))

    numeric = cfg.SCREENING_NUMERIC + ["Noise_Baseline"]
    reference = Pipeline([
        ("prep", dp.build_preprocessor(numeric, cfg.SCREENING_CATEGORICAL)),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ]).fit(tr, tr[cfg.TARGET])
    perm = permutation_importance_table(reference, te, te[cfg.TARGET], numeric + cfg.SCREENING_CATEGORICAL)
    noise = perm.loc[perm["Feature"] == "Noise_Baseline"].iloc[0]

    # Residual association tests: does a feature explain FPS *beyond* GPU score?
    base = LinearRegression().fit(train[["GPU_Score"]], train[cfg.TARGET])
    resid = train[cfg.TARGET] - base.predict(train[["GPU_Score"]])
    tests = []
    num_tested = [c for c in cfg.SCREENING_NUMERIC + cfg.INFORMATIONAL_NUMERIC if c != "GPU_Score"]
    cat_tested = cfg.SCREENING_CATEGORICAL + cfg.INFORMATIONAL_CATEGORICAL
    for col in num_tested:
        r, p = stats.pearsonr(train[col], resid)
        tests.append((col, "Pearson r vs residual", r, p))
    for col in cat_tested:
        groups = [resid[train[col] == lvl] for lvl in train[col].unique()]
        f, p = stats.f_oneway(*groups)
        tests.append((col, "ANOVA F vs residual", f, p))
    tests_df = pd.DataFrame(tests, columns=["Feature", "Test", "Statistic", "p-value"])
    tests_df["p (Bonferroni)"] = np.minimum(tests_df["p-value"] * len(tests_df), 1.0)

    out = perm.merge(tests_df, on="Feature", how="outer")
    out["Usable before launch"] = ~out["Feature"].isin(cfg.INFORMATIONAL_NUMERIC + cfg.INFORMATIONAL_CATEGORICAL)

    def verdict(r: pd.Series) -> str:
        if r["Feature"] == "Noise_Baseline":
            return "Random noise reference"
        if r["Feature"] == "GPU_Score":
            return "Dominant driver"
        if not r["Usable before launch"]:
            return "Informational only"
        informative = r["p (Bonferroni)"] < 0.05
        return "Informative" if informative else "No measurable effect"

    out["Verdict"] = out.apply(verdict, axis=1)
    out["Noise reference"] = float(noise["Importance (drop in R2)"])
    return out.sort_values("Importance (drop in R2)", ascending=False, na_position="last").reset_index(drop=True)


# --------------------------------------------------------------------------
# Derived-target evaluation: playability tiers and "reaches 60 FPS"
# --------------------------------------------------------------------------
def fps_to_tier(fps: float) -> str:
    for label, minimum, _, _ in cfg.PERFORMANCE_TIERS:
        if fps >= minimum:
            return label
    return cfg.PERFORMANCE_TIERS[-1][0]


def evaluate_tiers(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """How well do tiers derived from predicted FPS match tiers from measured FPS?"""
    actual = np.array([fps_to_tier(v) for v in y_true])
    predicted = np.array([fps_to_tier(v) for v in y_pred])
    order = cfg.TIER_LABELS_ASCENDING
    labels = [t for t in order if t in set(actual) | set(predicted)]
    idx = {t: i for i, t in enumerate(order)}
    within_one = np.mean([abs(idx[a] - idx[p]) <= 1 for a, p in zip(actual, predicted)])
    binary_true = (np.asarray(y_true) >= 60).astype(int)
    binary_pred = (np.asarray(y_pred) >= 60).astype(int)
    return {
        "labels": labels,
        "confusion": confusion_matrix(actual, predicted, labels=labels),
        "accuracy": float(accuracy_score(actual, predicted)),
        "within_one_tier": float(within_one),
        "macro_precision": float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "reach60": {
            "accuracy": float(accuracy_score(binary_true, binary_pred)),
            "precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
            "recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
            "f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(binary_true, y_pred)),
            "positive_rate": float(binary_true.mean()),
        },
    }


# --------------------------------------------------------------------------
# Quality learnability experiment (why Quality is NOT predicted by ML)
# --------------------------------------------------------------------------
def quality_learnability(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Test whether any classifier can predict Quality better than chance.

    The most generous setting is used: all configuration features AND the
    measured FPS are given to the classifiers.  If nothing beats the majority
    class there, a deployed 'Quality model' would only be reproducing noise.
    """
    full = cfg.FEATURE_SETS["full"]
    fs = cfg.FeatureSet("q", "Quality inputs", full.numeric + ("FPS",), full.categorical)
    rs = cfg.RANDOM_STATE
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=10, n_jobs=-1, random_state=rs),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, max_depth=2, random_state=rs),
    }
    labels = cfg.QUALITY_LEVELS
    majority = train["Quality"].value_counts(normalize=True).max()
    test_majority = float(test["Quality"].value_counts(normalize=True).max())
    cv = StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=rs)
    rows, best = [], None
    for name, clf in classifiers.items():
        pipe = Pipeline([("prep", dp.build_preprocessor(list(fs.numeric), list(fs.categorical))), ("model", clf)])
        cv_acc = cross_validate(pipe, train, train["Quality"], cv=cv, scoring="accuracy")["test_score"].mean()
        pipe.fit(train, train["Quality"])
        pred = pipe.predict(test)
        acc = accuracy_score(test["Quality"], pred)
        p_value = stats.binomtest(int((pred == test["Quality"]).sum()), len(test), 1 / len(labels), alternative="greater").pvalue
        rows.append({
            "Classifier": name, "CV accuracy": cv_acc, "Test accuracy": acc,
            "Precision (macro)": precision_score(test["Quality"], pred, labels=labels, average="macro", zero_division=0),
            "Recall (macro)": recall_score(test["Quality"], pred, labels=labels, average="macro", zero_division=0),
            "F1 (macro)": f1_score(test["Quality"], pred, labels=labels, average="macro", zero_division=0),
            "p-value vs chance": p_value,
        })
        if best is None or acc > best[0]:
            best = (acc, name, confusion_matrix(test["Quality"], pred, labels=labels))
    table = pd.DataFrame(rows)
    return {
        "table": table, "chance_level": 1 / len(labels), "majority_share": float(majority),
        "test_majority_share": test_majority,
        "best_classifier": best[1], "best_test_accuracy": float(best[0]), "confusion": best[2], "labels": labels,
        "learnable": bool((table["p-value vs chance"] < 0.01).any()),
    }


# --------------------------------------------------------------------------
# Training orchestrator
# --------------------------------------------------------------------------
def train_bundle(dataset_hash: str = "") -> dict:
    """Train, evaluate and package everything the application needs."""
    started = time.perf_counter()
    try:
        raw = dp.load_dataset()
        df, cleaning = dp.clean_dataset(raw)
        audit = dp.audit_dataset(raw, df)
        train, test = dp.split_data(df)
        log.info("train=%d test=%d", len(train), len(test))

        comparison = compare_regressors(train, test)
        best_idx, reason = select_best_model(comparison)
        chosen = comparison.loc[best_idx]
        fs = cfg.FEATURE_SETS[chosen["feature_key"]]
        estimator = regression_candidates()[chosen["Model"]][0]
        pipeline = make_pipeline(fs, estimator).fit(train, train[cfg.TARGET])

        y_test = test[cfg.TARGET].to_numpy()
        pred_test = pipeline.predict(test)
        test_metrics = regression_metrics(y_test, pred_test)

        # Empirical prediction interval from out-of-fold residuals (training data only).
        cv = KFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
        oof = cross_val_predict(make_pipeline(fs, estimator), train, train[cfg.TARGET], cv=cv)
        residuals = train[cfg.TARGET].to_numpy() - oof
        alpha = (1 - cfg.INTERVAL_COVERAGE) / 2
        q_lo, q_hi = np.quantile(residuals, [alpha, 1 - alpha])
        covered = (y_test >= pred_test + q_lo) & (y_test <= pred_test + q_hi)

        screening = screen_features(train, test)
        importance = permutation_importance_table(pipeline, test, test[cfg.TARGET], fs.columns)
        tiers = evaluate_tiers(y_test, pred_test)
        quality = quality_learnability(train, test)
        background = train[fs.columns].sample(min(cfg.SHAP_BACKGROUND_SIZE, len(train)), random_state=cfg.RANDOM_STATE)

        bundle = {
            "schema_version": cfg.BUNDLE_SCHEMA_VERSION,
            "sklearn_version": sklearn.__version__,
            "dataset_hash": dataset_hash,
            "pipeline": pipeline,
            "model_name": chosen["Model"],
            "feature_set": {"key": fs.key, "label": fs.label, "numeric": list(fs.numeric), "categorical": list(fs.categorical)},
            "selection_reason": reason,
            "comparison": comparison,
            "selected_row": int(best_idx),
            "test_metrics": test_metrics,
            "n_train": len(train), "n_test": len(test), "n_rows": len(df),
            "interval": {
                "coverage_nominal": cfg.INTERVAL_COVERAGE, "q_lo": float(q_lo), "q_hi": float(q_hi),
                "coverage_test": float(covered.mean()), "residuals": residuals,
                "residual_min": float(residuals.min()), "residual_max": float(residuals.max()),
            },
            "y_test": y_test, "pred_test": pred_test,
            "screening": screening, "importance": importance,
            "tiers": tiers, "quality_experiment": quality,
            "catalog": dp.build_catalog(df),
            "fps_reference": np.sort(df[cfg.TARGET].to_numpy()),
            "background": background,
            "cleaning": cleaning, "audit": audit,
            "train_seconds": time.perf_counter() - started,
        }
        if hasattr(pipeline.named_steps["model"], "coef_"):
            names = pipeline.named_steps["prep"].get_feature_names_out()
            bundle["coefficients"] = dict(zip(names, np.ravel(pipeline.named_steps["model"].coef_)))
        return bundle
    except GamePerfError:                                      # already user-friendly (e.g. bad dataset)
        raise
    except Exception as exc:                                   # unexpected: log details, show a friendly message
        log.exception("training failed")
        raise ModelUnavailableError("Model training failed. Check the dataset and the console log for details.") from exc


# --------------------------------------------------------------------------
# Prediction facade
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SystemConfig:
    """A validated hardware + game configuration."""
    cpu: str
    gpu: str
    ram_gb: int
    vram_gb: int
    game: str
    resolution: str
    ray_tracing: bool = False


@dataclass
class FpsPrediction:
    fps: float
    low: float
    high: float
    coverage: float
    samples: np.ndarray                  # plausible FPS values: prediction + OOF residuals
    outside_training_range: bool
    warnings: list[str]


@dataclass
class Explanation:
    baseline: float
    prediction: float
    contributions: pd.DataFrame          # Feature, Value, Contribution (FPS), Impact


class GamePerformancePredictor:
    """Validated, explainable FPS prediction on top of a trained bundle."""

    def __init__(self, bundle: dict):
        self.b = bundle
        self.pipeline = bundle["pipeline"]
        self.catalog = bundle["catalog"]
        self.features: list[str] = bundle["feature_set"]["numeric"] + bundle["feature_set"]["categorical"]

    # ----- validation ----------------------------------------------------
    def validate(self, cpu, gpu, ram_gb, vram_gb, game, resolution, ray_tracing=False) -> SystemConfig:
        cat = self.catalog
        problems = []
        if cpu not in cat["cpu"]:
            problems.append(f"CPU '{cpu}' is not in the training data. Supported: {', '.join(cat['cpu'])}.")
        if gpu not in cat["gpu"]:
            problems.append(f"GPU '{gpu}' is not in the training data. Supported: {', '.join(cat['gpu'])}.")
        if game not in cat["games"]:
            problems.append(f"Game '{game}' is not supported. Supported: {', '.join(cat['games'])}.")
        if resolution not in cat["resolutions"]:
            problems.append(f"Resolution '{resolution}' is not supported (the dataset contains {', '.join(cat['resolutions'])}).")
        for label, value, allowed in (("RAM", ram_gb, cat["ram_gb"]), ("VRAM", vram_gb, cat["vram_gb"])):
            try:
                ok = int(value) in allowed and float(value) == int(value)
            except (TypeError, ValueError):
                ok = False
            if not ok:
                problems.append(f"{label} value '{value}' is invalid. Supported values: {', '.join(map(str, allowed))} GB.")
        if problems:
            raise InvalidInputError(" ".join(problems))
        return SystemConfig(cpu, gpu, int(ram_gb), int(vram_gb), game, resolution, bool(ray_tracing))

    # ----- prediction ----------------------------------------------------
    def _frame(self, configs: list[SystemConfig]) -> pd.DataFrame:
        cat = self.catalog
        return pd.DataFrame({
            "GPU_Score": [cat["gpu"][c.gpu] for c in configs],
            "CPU_Score": [cat["cpu"][c.cpu] for c in configs],
            "RAM_GB": [c.ram_gb for c in configs],
            "VRAM_GB": [c.vram_gb for c in configs],
            "Ray_Tracing": [int(c.ray_tracing) for c in configs],
            "Resolution_MP": [cat["resolution_mp"][c.resolution] for c in configs],
            "Game": [c.game for c in configs],
        })[self.features]

    def predict_many(self, configs: list[SystemConfig]) -> np.ndarray:
        return self.pipeline.predict(self._frame(configs))

    def predict(self, config: SystemConfig) -> FpsPrediction:
        raw = float(self.predict_many([config])[0])
        iv = self.b["interval"]
        notes: list[str] = []
        # A frame rate cannot be negative: this is the only hard physical bound applied.
        fps = max(raw, 0.0)
        if fps != raw:
            notes.append("The raw model output was negative and was floored at 0 FPS (physically impossible otherwise).")
        samples = np.maximum(fps + iv["residuals"], 0.0)
        low = max(fps + iv["q_lo"], 0.0)
        high = fps + iv["q_hi"]
        # Predictions outside the benchmarked range are flagged, not silently altered.
        lo_ref, hi_ref = self.catalog["fps_min"], self.catalog["fps_max"]
        outside = not (lo_ref <= fps <= hi_ref)
        if outside:
            notes.append(
                f"Predicted FPS ({fps:.0f}) lies outside the benchmarked range ({lo_ref:.0f}-{hi_ref:.0f}); treat it as an extrapolation."
            )
        return FpsPrediction(fps, low, high, iv["coverage_nominal"], samples, outside, notes)

    # ----- explanation ---------------------------------------------------
    def explain(self, config: SystemConfig) -> Explanation:
        row = self._frame([config])
        base, phi, pred = shapley_values(self.pipeline.predict, row, self.b["background"][self.features], self.features)
        names = {"GPU_Score": "GPU Score", "CPU_Score": "CPU Score", "RAM_GB": "RAM", "VRAM_GB": "VRAM",
                 "Ray_Tracing": "Ray tracing", "Resolution_MP": "Resolution", "Game": "Game"}
        rows = [{
            "Feature": names.get(f, f), "Value": row[f].iloc[0], "Contribution": c,
            "Impact": impact_label(abs(c), cfg.IMPACT_THRESHOLDS_FPS),
        } for f, c in phi.items()]
        table = pd.DataFrame(rows).sort_values("Contribution", key=np.abs, ascending=False).reset_index(drop=True)
        return Explanation(base, pred, table)

    # ----- dataset-grounded evidence ----------------------------------------
    @staticmethod
    def similar_records(df: pd.DataFrame, config: SystemConfig) -> tuple[pd.DataFrame, str]:
        """Real benchmark rows closest to the configuration (most specific match that exists)."""
        levels = [
            ("same CPU, GPU, game and resolution", (df.CPU == config.cpu) & (df.GPU == config.gpu) & (df.Game == config.game) & (df.Resolution == config.resolution)),
            ("same CPU and GPU", (df.CPU == config.cpu) & (df.GPU == config.gpu)),
            ("same GPU", df.GPU == config.gpu),
        ]
        for label, mask in levels:
            subset = df[mask]
            if len(subset) >= 10:
                return subset, label
        return df[df.GPU == config.gpu], "same GPU"


# --------------------------------------------------------------------------
# Presentation helper for the model-comparison table
# --------------------------------------------------------------------------
def comparison_display(bundle: dict) -> pd.DataFrame:
    """Comparison table with a Status column (Selected / Within 1 SE / Evaluated / Baseline)."""
    t = bundle["comparison"]
    sel = bundle["selected_row"]
    models = t[t["complexity"] >= 0]
    best = models.loc[models["CV R2"].idxmax()]
    threshold = best["CV R2"] - best["CV R2 std"] / np.sqrt(cfg.CV_FOLDS)

    def status(row: pd.Series) -> str:
        if row.name == sel:
            return "Selected"
        if row["complexity"] < 0:
            return "Baseline"
        return "Within 1 SE of best" if row["CV R2"] >= threshold else "Evaluated"

    out = pd.DataFrame({
        "Model": t["Model"], "Features": t["Features"],
        "CV R²": t["CV R2"], "CV R² SD": t["CV R2 std"], "CV MAE": t["CV MAE"], "CV RMSE": t["CV RMSE"],
        "Test MAE": t["Test MAE"], "Test RMSE": t["Test RMSE"], "Test R²": t["Test R2"],
        "Status": t.apply(status, axis=1),
    })
    order = {"Selected": 0, "Within 1 SE of best": 1, "Evaluated": 2, "Baseline": 3}
    return out.sort_values(["Status", "CV R²"], key=lambda c: c.map(order) if c.name == "Status" else -c).reset_index(drop=True)
