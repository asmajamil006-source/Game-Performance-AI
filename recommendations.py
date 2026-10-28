"""Interpretation layer: turns raw model output into decisions a player can act on.

Everything here is derived from (a) the model's predictions, (b) measured
evidence stored with the model (screening, intervals) or (c) explicit rules whose
parameters are documented in config.py.  Rule-based parts are labelled as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

import config as cfg
from model_engine import (
    Explanation, FpsPrediction, GamePerformancePredictor, SystemConfig, fps_to_tier,
)
from data_processor import score_gap_pct


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class Tier:
    name: str
    min_fps: float
    color: str
    description: str


@dataclass
class Bottleneck:
    state: str                 # "GPU-limited" | "CPU-limited" | "Balanced"
    gap_pct: float
    cpu_score: float
    gpu_score: float
    limiting: str | None       # "GPU" | "CPU" | None
    summary: str


@dataclass
class QualityRecommendation:
    level: str
    headroom_level: str        # level allowed by FPS headroom alone
    limited_by: str | None     # "VRAM", "RAM" or None
    trace: list[dict] = field(default_factory=list)


@dataclass
class Recommendation:
    kind: str                  # "success" | "info" | "warning" | "error"
    title: str
    body: str
    source: str                # "Model", "Rule" or "Data"


@dataclass
class PerformanceReport:
    config: SystemConfig
    target_fps: int
    prediction: FpsPrediction
    explanation: Explanation
    tier: Tier
    tier_probabilities: dict[str, float]
    p_meet_target: float          # P(actual FPS >= target), from out-of-fold residuals
    confidence: float             # confidence in the 'meets / misses target' verdict
    confidence_label: str
    percentile: float
    level: str
    bottleneck: Bottleneck
    quality: QualityRecommendation
    upgrades: pd.DataFrame
    recommendations: list[Recommendation]
    ignored_inputs: list[dict]
    evidence: dict
    advisor_goal: str = "Balanced gameplay"
    advisor_summary: str = ""
    advisor_notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Playability, performance level, reliability
# --------------------------------------------------------------------------
def classify_playability(fps: float) -> Tier:
    for name, minimum, color, description in cfg.PERFORMANCE_TIERS:
        if fps >= minimum:
            return Tier(name, minimum, color, description)
    name, minimum, color, description = cfg.PERFORMANCE_TIERS[-1]
    return Tier(name, minimum, color, description)


def tier_probabilities(samples: np.ndarray) -> dict[str, float]:
    """Share of plausible FPS outcomes (prediction + residuals) in each tier."""
    tiers = [fps_to_tier(v) for v in samples]
    counts = pd.Series(tiers).value_counts(normalize=True)
    return {name: float(counts.get(name, 0.0)) for name, *_ in cfg.PERFORMANCE_TIERS}


def confidence_label(probability: float) -> str:
    if probability >= cfg.RELIABILITY_HIGH:
        return "High"
    if probability >= cfg.RELIABILITY_MODERATE:
        return "Moderate"
    return "Low"


def performance_level(fps: float, reference: np.ndarray) -> tuple[float, str]:
    """Percentile of the predicted FPS among all benchmark records, and a label for it."""
    percentile = float(np.searchsorted(reference, fps, side="right") / len(reference) * 100)
    if percentile >= 80:
        label = "High-end"
    elif percentile >= 50:
        label = "Upper mid-range"
    elif percentile >= 20:
        label = "Mid-range"
    else:
        label = "Entry-level"
    return percentile, label


# --------------------------------------------------------------------------
# Hardware balance (bottleneck) analysis
# --------------------------------------------------------------------------
def analyze_bottleneck(cpu_score: float, gpu_score: float) -> Bottleneck:
    """Classify the CPU/GPU pairing using the dataset's own Bottleneck_Pct definition."""
    gap = score_gap_pct(cpu_score, gpu_score)
    if gap < cfg.BALANCED_THRESHOLD_PCT:
        return Bottleneck("Balanced", gap, cpu_score, gpu_score, None,
                          f"CPU and GPU scores differ by only {gap:.1f}%, so neither clearly holds the other back.")
    if gpu_score < cpu_score:
        return Bottleneck("GPU-limited", gap, cpu_score, gpu_score, "GPU",
                          f"The GPU score is {gap:.0f}% below the CPU score; the graphics card is the weaker component.")
    return Bottleneck("CPU-limited", gap, cpu_score, gpu_score, "CPU",
                      f"The CPU score is {gap:.0f}% below the GPU score; the processor is the weaker component.")


# --------------------------------------------------------------------------
# Graphics quality: transparent rule engine fed by the ML-predicted FPS
# --------------------------------------------------------------------------
def _res_band(megapixels: float) -> str:
    return "low" if megapixels < 1.5 else ("mid" if megapixels <= 2.6 else "high")


def vram_guideline(level: str, megapixels: float) -> int:
    base = cfg.VRAM_GUIDELINE_GB[level]
    return 0 if base == 0 else max(base + cfg.VRAM_RES_ADJUST_GB[_res_band(megapixels)], 2)


def recommend_quality(fps: float, target_fps: int, config: SystemConfig, megapixels: float) -> QualityRecommendation:
    """RULE-BASED recommendation (the dataset's Quality column has no learnable signal).

    1. FPS headroom: the highest preset whose assumed cost still leaves predicted FPS
       at or above target x multiplier (see QUALITY_HEADROOM_MULTIPLIER).
    2. VRAM guideline: cap the preset if VRAM is below the texture guideline.
    3. RAM guideline: cap the preset if system memory is below the guideline.
    """
    trace = []
    levels = cfg.QUALITY_LEVELS
    headroom_level = "Low"
    for level in levels:
        if fps >= target_fps * cfg.QUALITY_HEADROOM_MULTIPLIER[level]:
            headroom_level = level
    ratio = fps / target_fps if target_fps else 0.0
    trace.append({
        "Rule": "FPS headroom (uses ML prediction)",
        "Check": f"Predicted {fps:.0f} FPS vs target {target_fps} FPS = {ratio:.2f}x",
        "Outcome": f"Up to {headroom_level}",
        "Status": "ok" if headroom_level != "Low" else "limit",
    })
    level, limited_by = headroom_level, None
    for name, available, guideline in (
        ("VRAM", config.vram_gb, lambda lv: vram_guideline(lv, megapixels)),
        ("RAM", config.ram_gb, lambda lv: cfg.RAM_GUIDELINE_GB[lv]),
    ):
        before = level
        while level != "Low" and available < guideline(level):
            level = levels[levels.index(level) - 1]
        need = guideline(before)
        advice = f"~{need} GB advised for {before}" if need else f"no minimum for {before}"
        if level != before:
            limited_by = limited_by or name
            trace.append({"Rule": f"{name} guideline", "Check": f"{available} GB available, {advice}",
                          "Outcome": f"Capped to {level}", "Status": "limit"})
        else:
            trace.append({"Rule": f"{name} guideline", "Check": f"{available} GB available, {advice}",
                          "Outcome": f"{level} allowed", "Status": "ok"})
    return QualityRecommendation(level, headroom_level, limited_by, trace)


# --------------------------------------------------------------------------
# What-if upgrade analysis (uses the trained model)
# --------------------------------------------------------------------------
def build_upgrade_scenarios(predictor: GamePerformancePredictor, config: SystemConfig, base_fps: float) -> pd.DataFrame:
    cat = predictor.catalog
    rows, configs = [], []
    for component, field_name, scores, current in (
        ("GPU", "gpu", cat["gpu"], config.gpu), ("CPU", "cpu", cat["cpu"], config.cpu),
    ):
        for option, score in scores.items():
            if score > scores[current]:
                configs.append(replace(config, **{field_name: option}))
                rows.append({"Component": component, "Upgrade to": option, "Score": score})
    if not rows:
        return pd.DataFrame(columns=["Component", "Upgrade to", "Score", "Predicted FPS", "Change (FPS)", "Change (%)", "Balance after"])
    preds = predictor.predict_many(configs)
    for row, cfg_i, pred in zip(rows, configs, preds):
        row["Predicted FPS"] = float(max(pred, 0))
        row["Change (FPS)"] = row["Predicted FPS"] - base_fps
        row["Change (%)"] = row["Change (FPS)"] / base_fps * 100 if base_fps else 0.0
        row["Balance after"] = analyze_bottleneck(cat["cpu"][cfg_i.cpu], cat["gpu"][cfg_i.gpu]).state
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Inputs the model found to have no measurable effect
# --------------------------------------------------------------------------
_INPUT_TO_FEATURE = {
    "Game": "Game", "Resolution": "Resolution_MP", "RAM": "RAM_GB", "VRAM": "VRAM_GB",
    "Ray tracing": "Ray_Tracing", "CPU": "CPU_Score",
}


def ignored_inputs(screening: pd.DataFrame) -> list[dict]:
    rows = []
    indexed = screening.set_index("Feature")
    for label, feature in _INPUT_TO_FEATURE.items():
        if feature in indexed.index and indexed.loc[feature, "Verdict"] == "No measurable effect":
            rows.append({"Input": label, "p (Bonferroni)": float(indexed.loc[feature, "p (Bonferroni)"])})
    return rows


# --------------------------------------------------------------------------
# Dynamic recommendations
# --------------------------------------------------------------------------
def generate_recommendations(
    config: SystemConfig, pred: FpsPrediction, tier: Tier, target_fps: int, bottleneck: Bottleneck,
    quality: QualityRecommendation, upgrades: pd.DataFrame, confidence: float, resolutions: list[str],
) -> list[Recommendation]:
    recs: list[Recommendation] = []
    fps = pred.fps

    # 1. Overall verdict against the target.
    if fps < 30:
        recs.append(Recommendation("error", "Below the playable minimum",
                                   f"Predicted {fps:.0f} FPS is under 30 FPS. Lower the graphics preset first; a stronger GPU is the only change the model predicts will help.", "Model"))
    elif fps < target_fps:
        recs.append(Recommendation("warning", f"Below your {target_fps} FPS target",
                                   f"Predicted {fps:.0f} FPS is {target_fps - fps:.0f} FPS short of the target. Use the '{quality.level}' preset and consider the upgrade options below.", "Model"))
    elif fps >= target_fps * 1.5:
        recs.append(Recommendation("success", "Large performance headroom",
                                   f"Predicted {fps:.0f} FPS is {fps / target_fps:.1f}x your {target_fps} FPS target, enough to run the '{quality.level}' preset.", "Model"))
    else:
        recs.append(Recommendation("success", f"Meets your {target_fps} FPS target",
                                   f"Predicted {fps:.0f} FPS reaches the target with '{quality.level}' settings.", "Model"))

    # 2. GPU upgrade (the only lever the model can quantify).
    gpu_up = upgrades[upgrades["Component"] == "GPU"] if len(upgrades) else upgrades
    if len(gpu_up):
        best = gpu_up.sort_values("Change (FPS)", ascending=False).iloc[0]
        if best["Change (FPS)"] >= cfg.MIN_MEANINGFUL_FPS_GAIN and fps < 90:
            recs.append(Recommendation("info", f"GPU upgrade has the biggest expected payoff",
                                       f"Moving to the {best['Upgrade to']} is predicted to raise FPS from {fps:.0f} to {best['Predicted FPS']:.0f} "
                                       f"({best['Change (FPS)']:+.0f} FPS, {best['Change (%)']:+.0f}%).", "Model"))
    else:
        recs.append(Recommendation("success", "Top GPU in the dataset",
                                   f"The {config.gpu} is the fastest GPU in the training data, so no GPU upgrade can be evaluated.", "Data"))

    # 3. Hardware balance, stated honestly with respect to the evidence.
    cpu_up = upgrades[upgrades["Component"] == "CPU"] if len(upgrades) else upgrades
    cpu_gain = float(cpu_up["Change (FPS)"].max()) if len(cpu_up) else 0.0
    if bottleneck.state == "CPU-limited":
        recs.append(Recommendation("info", "CPU is the lower-scoring component",
                                   f"{bottleneck.summary} Even so, the model predicts at most {cpu_gain:+.1f} FPS from a faster CPU, because FPS in this dataset "
                                   "tracks GPU score and shows no measurable CPU effect. Real CPU-heavy games may behave differently.", "Model"))
    elif bottleneck.state == "GPU-limited":
        recs.append(Recommendation("info", "GPU is the limiting component",
                                   f"{bottleneck.summary} This agrees with the model, which attributes almost all FPS variation to GPU score.", "Model"))
    else:
        recs.append(Recommendation("success", "Well-balanced CPU/GPU pairing", bottleneck.summary, "Data"))

    # 4. Memory guidance from the rule engine.
    if quality.limited_by == "VRAM":
        recs.append(Recommendation("warning", "VRAM limits the texture preset",
                                   f"{config.vram_gb} GB VRAM is below the guideline for higher presets at {config.resolution}. Keep texture quality at or below '{quality.level}'.", "Rule"))
    if config.ram_gb < 16:
        recs.append(Recommendation("info", "Consider 16 GB of RAM",
                                   f"{config.ram_gb} GB is below the 16 GB guideline for High/Ultra presets. This is general guidance; RAM showed no effect on FPS in the data.", "Rule"))

    # 5. Settings advice when under target, without inventing numbers.
    if fps < target_fps:
        recs.append(Recommendation("info", "Reduce resolution or preset as a first step",
                                   "This is standard practice, but the dataset shows no FPS change between "
                                   f"{' / '.join(resolutions)}, so the model cannot quantify the gain.", "Rule"))

    # 6. Reliability.
    if confidence < cfg.RELIABILITY_MODERATE:
        verdict = "meets" if fps >= target_fps else "misses"
        recs.append(Recommendation("warning", f"Close call against your {target_fps} FPS target",
                                   f"The model expects the system to {verdict} the target, but only {confidence:.0%} of plausible outcomes agree. "
                                   f"Expect results within roughly {pred.low:.0f}-{pred.high:.0f} FPS.", "Model"))
    recs.extend(Recommendation("warning", "Model warning", w, "Model") for w in pred.warnings)
    return recs


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def build_report(
    predictor: GamePerformancePredictor, config: SystemConfig, target_fps: int, dataset: pd.DataFrame,
) -> PerformanceReport:
    cat = predictor.catalog
    pred = predictor.predict(config)
    explanation = predictor.explain(config)
    tier = classify_playability(pred.fps)
    probs = tier_probabilities(pred.samples)
    p_meet = float(np.mean(pred.samples >= target_fps))
    confidence = p_meet if pred.fps >= target_fps else 1.0 - p_meet
    percentile, level = performance_level(pred.fps, predictor.b["fps_reference"])
    bottleneck = analyze_bottleneck(cat["cpu"][config.cpu], cat["gpu"][config.gpu])
    quality = recommend_quality(pred.fps, target_fps, config, cat["resolution_mp"][config.resolution])
    upgrades = build_upgrade_scenarios(predictor, config, pred.fps)
    ignored = ignored_inputs(predictor.b["screening"])
    subset, match_label = predictor.similar_records(dataset, config)
    evidence = {
        "match": match_label, "n": int(len(subset)), "mean": float(subset[cfg.TARGET].mean()),
        "min": float(subset[cfg.TARGET].min()), "max": float(subset[cfg.TARGET].max()),
    }
    recs = generate_recommendations(config, pred, tier, target_fps, bottleneck, quality, upgrades, confidence, cat["resolutions"])
    return PerformanceReport(
        config, target_fps, pred, explanation, tier, probs, p_meet, confidence, confidence_label(confidence),
        percentile, level, bottleneck, quality, upgrades, recs, ignored, evidence,
    )


PERFORMANCE_GOALS = {
    "Balanced gameplay": 60,
    "Smoother performance": 90,
    "Higher visual quality": 60,
}


def resolve_performance_goal(goal: str, custom_target: int | float | None = None) -> tuple[int, str]:
    """Resolve a named goal or validate a user-selected FPS target."""
    if goal == "Custom FPS target":
        if custom_target is None or isinstance(custom_target, bool):
            raise ValueError("Enter a target between 15 and 360 FPS.")
        try:
            value = float(custom_target)
        except (TypeError, ValueError) as exc:
            raise ValueError("Target FPS must be a number between 15 and 360.") from exc
        if not value.is_integer() or not 15 <= value <= 360:
            raise ValueError("Target FPS must be a whole number between 15 and 360.")
        return int(value), "Custom FPS target"
    if goal not in PERFORMANCE_GOALS:
        raise ValueError("Choose a supported performance goal.")
    return PERFORMANCE_GOALS[goal], goal


def build_advisor_summary(report: PerformanceReport, goal: str = "Balanced gameplay") -> tuple[str, list[str]]:
    """Evidence-bounded, offline advisor; never claims unmeasured settings gains."""
    p = report.prediction
    notes = []
    if p.outside_training_range:
        notes.append("The configuration is outside the model's reliable training range; treat FPS and upgrade comparisons as low-reliability estimates.")
    if report.evidence.get("n", 0) < 10:
        notes.append("Few closely matching benchmark records are available; use the interval and test error as context, not a guarantee.")
    if report.confidence_label == "Low":
        notes.append("The target verdict is uncertain based on the model's residual distribution.")
    if goal == "Smoother performance":
        notes.append("Goal: Smoother performance. Prioritize frame-rate headroom. The rule-based preset is the safer starting point; lower resolution or demanding effects if actual FPS falls short.")
    elif goal == "Higher visual quality":
        notes.append(f"Goal: Higher visual quality. Prioritize image quality while retaining the {report.target_fps} FPS target. The {report.quality.level} preset is a rule-based suggestion, not a benchmark-validated settings test.")
    else:
        notes.append(f"Goal: {goal}. Balance image quality and frame rate around {report.target_fps} FPS.")
    if p.fps < report.target_fps:
        notes.append(f"Predicted FPS is {report.target_fps - p.fps:.1f} below target. Try the suggested {report.quality.level} preset first, then compare real in-game measurements.")
    else:
        notes.append(f"Predicted FPS meets the target by {p.fps - report.target_fps:.1f} FPS; validate with an in-game benchmark before raising quality.")
    if report.bottleneck.limiting:
        notes.append(f"Rule-based hardware balance flags {report.bottleneck.limiting} as the lower-scoring component. This is not proof of a real-world bottleneck.")
    notes.append("Upgrade deltas are model what-if estimates from this dataset, not guaranteed gains. No OS, driver, overclock, or game-file changes are made.")
    notes.insert(0, f"Model prediction: {p.fps:.1f} FPS with a {p.coverage:.0%} prediction interval of {p.low:.1f}–{p.high:.1f} FPS.")
    summary = " ".join(notes)
    report.advisor_goal = goal
    report.advisor_summary = summary
    report.advisor_notes = notes
    return summary, notes


def report_to_text(report: PerformanceReport) -> str:
    """Plain-text summary of a prediction (for the download button)."""
    c, p, b = report.config, report.prediction, report.bottleneck
    lines = [
        "AI GAME PERFORMANCE PREDICTOR - PREDICTION REPORT", "=" * 52,
        f"System    : {c.cpu} + {c.gpu}, {c.ram_gb} GB RAM, {c.vram_gb} GB VRAM, ray tracing {'on' if c.ray_tracing else 'off'}",
        f"Game      : {c.game} at {c.resolution}",
        f"Target    : {report.target_fps} FPS", "",
        f"Predicted FPS     : {p.fps:.1f}  ({p.coverage:.0%} interval {p.low:.0f}-{p.high:.0f})",
        f"Playability       : {report.tier.name} - {report.tier.description}",
        f"Performance level : {report.level} ({report.percentile:.0f}th percentile of benchmark records)",
        f"Recommended preset: {report.quality.level} (rule-based)",
        f"Target confidence : {report.confidence:.0%} ({report.confidence_label}); P(FPS >= target) = {report.p_meet_target:.0%}",
        f"Hardware balance  : {b.state} (score gap {b.gap_pct:.1f}%)",
        f"Advisor goal      : {report.advisor_goal}",
        f"Advisor summary   : {report.advisor_summary or build_advisor_summary(report, report.advisor_goal)[0]}",
        "", "Why this prediction (contribution vs average system):",
    ]
    lines += [f"  - {r.Feature}: {r.Contribution:+.1f} FPS ({r.Impact})" for r in report.explanation.contributions.itertuples()]
    lines += ["", "Recommendations:"] + [f"  [{r.source}] {r.title}: {r.body}" for r in report.recommendations]
    return "\n".join(lines) + "\n"
