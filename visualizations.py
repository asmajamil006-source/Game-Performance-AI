"""Matplotlib charts for the dashboard.

Each function returns a Figure; ``fig_to_png`` converts it to bytes so that the
app can cache rendered charts and display them quickly.  A single consistent
style is applied to every chart.
"""
from __future__ import annotations

import io
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import config as cfg  # noqa: E402

P = cfg.PALETTE
SEQ = ["#1E3A8A", "#2563EB", "#0D9488", "#D97706", "#7C3AED", "#DB2777", "#64748B", "#0891B2"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": P["ink"],
    "axes.labelcolor": P["muted"], "axes.edgecolor": P["soft_grey"], "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "xtick.color": P["muted"], "ytick.color": P["muted"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#EEF2F7", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def fig_to_png(fig: Figure, dpi: int = 160) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return buf.getvalue()


def safe_corr(a, b) -> float:
    """Pearson r, or NaN (never a warning/exception) if either series is constant."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _fmt_r(r: float) -> str:
    return f"{r:+.2f}" if np.isfinite(r) else "n/a (constant)"


def _empty(message: str, size=(6, 3)) -> Figure:
    fig, ax = plt.subplots(figsize=size)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=P["muted"], fontsize=11)
    return fig


# ======================================================================
# Dataset analytics
# ======================================================================
def fps_distribution(df: pd.DataFrame) -> Figure:
    if df.empty:
        return _empty("No records match the selected filters")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(df[cfg.TARGET], bins=range(int(df[cfg.TARGET].min()), int(df[cfg.TARGET].max()) + 4, 3),
            color=P["primary"], alpha=0.85, edgecolor="white")
    mean, median = df[cfg.TARGET].mean(), df[cfg.TARGET].median()
    ax.axvline(mean, color=P["warn"], lw=2, label=f"Mean {mean:.1f}")
    ax.axvline(median, color=P["accent"], lw=2, ls="--", label=f"Median {median:.0f}")
    for _, minimum, _, _ in cfg.PERFORMANCE_TIERS[:4]:
        if df[cfg.TARGET].min() <= minimum <= df[cfg.TARGET].max():
            ax.axvline(minimum, color=P["soft_grey"], lw=1, ls=":")
    ax.set(xlabel="FPS", ylabel="Records", title=f"FPS distribution ({len(df):,} records)")
    ax.legend()
    return fig


def fps_boxplot(df: pd.DataFrame, by: str, order: list[str] | None = None, title: str = "", figsize=(7, 3.8)) -> Figure:
    if df.empty:
        return _empty("No records match the selected filters")
    order = [o for o in (order or sorted(df[by].unique())) if o in set(df[by])]
    data = [df.loc[df[by] == o, cfg.TARGET].to_numpy() for o in order]
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": "white", "lw": 2},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4}, whiskerprops={"color": P["soft_grey"]},
                    capprops={"color": P["soft_grey"]})
    for patch in bp["boxes"]:
        patch.set(facecolor=P["primary"], alpha=0.9, edgecolor="none")
    ax.scatter(range(1, len(order) + 1), [d.mean() for d in data], color="white", edgecolor=P["ink"], zorder=5, s=24, label="Mean")
    ax.set_xticks(range(1, len(order) + 1), [f"{textwrap.fill(str(o), 11)}\n(n={len(d):,})" for o, d in zip(order, data)])
    ax.tick_params(axis="x", labelsize=8)
    ax.set(ylabel="FPS", title=title or f"FPS by {by}")
    ax.legend(loc="upper left")
    return fig


def gpu_vs_fps(df: pd.DataFrame) -> Figure:
    if df.empty:
        return _empty("No records match the selected filters")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.scatter(df["GPU_Score"] + rng.normal(0, 180, len(df)), df[cfg.TARGET], s=7, alpha=0.25, color=P["primary"], label="Benchmark records")
    means = df.groupby("GPU_Score")[cfg.TARGET].mean()
    ax.plot(means.index, means.values, "o-", color=P["warn"], lw=2, ms=6, label="Mean FPS per GPU")
    if len(means) > 1:
        slope, intercept = np.polyfit(df["GPU_Score"], df[cfg.TARGET], 1)
        xs = np.array([df["GPU_Score"].min(), df["GPU_Score"].max()])
        ax.plot(xs, slope * xs + intercept, "--", color=P["ink"], lw=1.2, label=f"Linear fit ({slope * 1000:.1f} FPS per 1,000 pts)")
    ax.set(xlabel="GPU benchmark score", ylabel="FPS", title="GPU score vs FPS (jittered horizontally for visibility)")
    ax.legend(loc="upper left")
    return fig


def cpu_gpu_heatmap(df: pd.DataFrame) -> Figure:
    if df.empty:
        return _empty("No records match the selected filters")
    pivot = df.pivot_table(index="GPU", columns="CPU", values=cfg.TARGET, aggfunc="mean")
    pivot = pivot.loc[df.groupby("GPU")["GPU_Score"].first().sort_values(ascending=False).index.intersection(pivot.index)]
    pivot = pivot[df.groupby("CPU")["CPU_Score"].first().sort_values().index.intersection(pivot.columns)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto", vmin=30, vmax=100)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 70 else P["ink"])
    ax.set(title="Mean FPS by GPU (rows) and CPU (columns, weakest to strongest)")
    fig.colorbar(im, ax=ax, label="Mean FPS", fraction=0.04, pad=0.02)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> Figure:
    cols = [c for c in ["FPS", "GPU_Score", "CPU_Score", "RAM_GB", "VRAM_GB", "Ray_Tracing", "Resolution_MP",
                        "PSU_Watts", "Driver_Version", "Load_Time_Sec", "Bottleneck_Pct"] if c in df.columns]
    cols = [c for c in cols if df[c].nunique() > 1] if len(df) else []
    if len(df) < 3 or cfg.TARGET not in cols or len(cols) < 2:
        return _empty("Not enough varying data for correlation analysis\n(filters leave only constant columns)")
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5.4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > 0.55 else P["ink"])
    ax.set_title("Pearson correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    return fig


def quality_panels(df: pd.DataFrame) -> Figure:
    if df.empty:
        return _empty("No records match the selected filters", (7, 3.4))
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    counts = df["Quality"].value_counts().reindex(cfg.QUALITY_LEVELS).fillna(0)
    a.bar(counts.index, counts.values, color=SEQ[:4])
    for i, v in enumerate(counts.values):
        a.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    a.set(title="Quality label distribution", ylabel="Records")
    data = [df.loc[df["Quality"] == q, cfg.TARGET] for q in cfg.QUALITY_LEVELS]
    bp = b.boxplot(data, patch_artist=True, medianprops={"color": "white", "lw": 2}, widths=0.6,
                   flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
    b.set_xticks(range(1, len(cfg.QUALITY_LEVELS) + 1), cfg.QUALITY_LEVELS)
    for patch, c in zip(bp["boxes"], SEQ[:4]):
        patch.set(facecolor=c, edgecolor="none")
    b.set(title="FPS by recorded Quality (no relationship)", ylabel="FPS")
    fig.tight_layout()
    return fig


def bottleneck_confounding(df: pd.DataFrame) -> Figure:
    """Raw FPS-vs-bottleneck relationship, and the same after removing each GPU's mean FPS."""
    if len(df) < 10:
        return _empty("Not enough records", (8.4, 3.4))
    gpu_colors = {g: SEQ[i % len(SEQ)] for i, g in enumerate(sorted(df["GPU"].unique()))}
    resid = df[cfg.TARGET] - df.groupby("GPU")[cfg.TARGET].transform("mean")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.8, 3.6), sharex=True)
    for g, sub in df.groupby("GPU"):
        a.scatter(sub["Bottleneck_Pct"], sub[cfg.TARGET], s=6, alpha=0.35, color=gpu_colors[g], label=g)
    r_raw = safe_corr(df["Bottleneck_Pct"], df[cfg.TARGET])
    a.set(title=f"Raw: r = {_fmt_r(r_raw)}", xlabel="Bottleneck %", ylabel="FPS")
    a.legend(fontsize=7, markerscale=2.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    b.scatter(df["Bottleneck_Pct"], resid, s=6, alpha=0.3, color=P["primary"])
    r_adj = safe_corr(df["Bottleneck_Pct"], resid)
    if df["Bottleneck_Pct"].nunique() > 1:
        slope, intercept = np.polyfit(df["Bottleneck_Pct"], resid, 1)
        xs = np.array([df["Bottleneck_Pct"].min(), df["Bottleneck_Pct"].max()])
        b.plot(xs, slope * xs + intercept, color=P["warn"], lw=2)
    b.axhline(0, color=P["soft_grey"], lw=1)
    b.set(title=f"After removing each GPU's mean FPS: r = {_fmt_r(r_adj)}", xlabel="Bottleneck %", ylabel="FPS minus GPU mean")
    fig.tight_layout()
    return fig


# ======================================================================
# Prediction page
# ======================================================================
def explanation_chart(explanation, prediction_fps: float) -> Figure:
    table = explanation.contributions.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.9 + 0.55 * len(table)))
    colors = [P["good"] if v >= 0 else P["bad"] for v in table["Contribution"]]
    ax.barh(table["Feature"], table["Contribution"], color=colors, height=0.55)
    span = max(table["Contribution"].abs().max(), 1.0)
    for i, (v, imp) in enumerate(zip(table["Contribution"], table["Impact"])):
        ax.text(v + (0.03 * span if v >= 0 else -0.03 * span), i, f"{v:+.1f} FPS  ({imp})", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color=P["ink"], lw=1)
    ax.set_xlim(-span * 1.9, span * 1.9)
    ax.set(xlabel=f"Contribution to FPS vs average system ({explanation.baseline:.1f} FPS)",
           title=f"Why this prediction? ({prediction_fps:.0f} FPS)")
    ax.grid(axis="y", visible=False)
    return fig


def uncertainty_chart(prediction, target_fps: int) -> Figure:
    s = prediction.samples
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    ax.hist(s, bins=np.arange(np.floor(s.min()), np.ceil(s.max()) + 2, 2), color=P["soft_blue"], edgecolor="white")
    ax.axvspan(prediction.low, prediction.high, color=P["primary"], alpha=0.12, label=f"{prediction.coverage:.0%} interval ({prediction.low:.0f}-{prediction.high:.0f})")
    ax.axvline(prediction.fps, color=P["primary_dark"], lw=2.2, label=f"Prediction {prediction.fps:.0f}")
    ax.axvline(target_fps, color=P["warn"], lw=2, ls="--", label=f"Your target {target_fps}")
    if s.min() - 12 <= 30 <= s.max() + 12:
        ax.axvline(30, color=P["bad"], lw=1.2, ls=":", label="30 FPS minimum")
    ax.set(xlabel="Plausible FPS outcomes", ylabel="Frequency", title="Expected range of real-world FPS")
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    return fig


def balance_chart(b) -> Figure:
    fig, ax = plt.subplots(figsize=(6.2, 1.9))
    names, vals = ["GPU score", "CPU score"], [b.gpu_score, b.cpu_score]
    bars = ax.barh(names, vals, color=[P["primary"], P["accent"]], height=0.5)
    for bar, v in zip(bars, vals):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f"  {v:,.0f}", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set(title=f"{b.state}  (score gap {b.gap_pct:.1f}%)")
    ax.grid(axis="y", visible=False)
    return fig


def upgrade_chart(upgrades: pd.DataFrame) -> Figure:
    if upgrades.empty:
        return _empty("No upgrade options exist above this hardware in the dataset", (6.2, 2))
    d = upgrades.sort_values(["Component", "Score"], ascending=[False, True]).reset_index(drop=True)
    labels = [f"{r.Component}: {r['Upgrade to']}" for _, r in d.iterrows()]
    fig, ax = plt.subplots(figsize=(6.2, 0.9 + 0.36 * len(d)))
    colors = [P["primary"] if c == "GPU" else P["accent"] for c in d["Component"]]
    ax.barh(labels[::-1], d["Change (FPS)"][::-1], color=colors[::-1], height=0.6)
    for i, v in enumerate(d["Change (FPS)"][::-1]):
        ax.text(v + (0.8 if v >= 0 else -0.8), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color=P["ink"], lw=1)
    lim = max(d["Change (FPS)"].abs().max(), 5) * 1.25
    ax.set_xlim(-lim * 0.3, lim)
    ax.set(xlabel="Model-predicted FPS change", title="What-if: single-component upgrades")
    ax.grid(axis="y", visible=False)
    return fig


# ======================================================================
# Model insights
# ======================================================================
def model_comparison_chart(table: pd.DataFrame, selected_idx: int) -> Figure:
    models = table[table["complexity"] >= 0].copy()
    models["label"] = models["Model"] + "  [" + models["feature_key"] + "]"
    models = models.sort_values("CV R2")
    fig, ax = plt.subplots(figsize=(7.4, 0.9 + 0.33 * len(models)))
    colors = [P["warn"] if i == selected_idx else P["primary"] for i in models.index]
    ax.errorbar(models["CV R2"], range(len(models)), xerr=models["CV R2 std"], fmt="none", ecolor=P["soft_grey"], capsize=3, zorder=1)
    ax.scatter(models["CV R2"], range(len(models)), c=colors, s=55, zorder=3)
    ax.set_yticks(range(len(models)), models["label"])
    base = table[table["complexity"] < 0]["CV R2"].iloc[0]
    ax.set(xlabel=f"5-fold CV R2 on training data (dot = mean, bar = ±1 SD)\nMean-prediction baseline: R2 = {base:.3f} (off-scale)",
           title="Model comparison (selected model in amber)")
    ax.grid(axis="y", visible=False)
    return fig


def predicted_vs_actual(y_true, y_pred) -> Figure:
    fig, ax = plt.subplots(figsize=(4.6, 4))
    rng = np.random.default_rng(0)
    ax.scatter(y_pred + rng.normal(0, .25, len(y_pred)), y_true, s=8, alpha=0.35, color=P["primary"])
    lo, hi = min(np.min(y_true), np.min(y_pred)) - 2, max(np.max(y_true), np.max(y_pred)) + 2
    ax.plot([lo, hi], [lo, hi], color=P["warn"], lw=2, label="Perfect prediction")
    ax.set(xlabel="Predicted FPS", ylabel="Actual FPS", title="Held-out test set", xlim=(lo, hi), ylim=(lo, hi))
    ax.legend(loc="upper left")
    return fig


def residual_hist(residuals, q_lo: float, q_hi: float, coverage: float) -> Figure:
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.hist(residuals, bins=30, color=P["primary"], alpha=0.85, edgecolor="white")
    ax.axvline(q_lo, color=P["warn"], ls="--", lw=2, label=f"{coverage:.0%} interval ({q_lo:+.1f} to {q_hi:+.1f})")
    ax.axvline(q_hi, color=P["warn"], ls="--", lw=2)
    ax.axvline(0, color=P["ink"], lw=1)
    ax.set(xlabel="Prediction error (actual - predicted), FPS", ylabel="Records", title="Out-of-fold residuals")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return fig


def screening_chart(screening: pd.DataFrame) -> Figure:
    d = screening.dropna(subset=["Importance (drop in R2)"]).copy()
    noise = d.loc[d["Feature"] == "Noise_Baseline"].iloc[0]
    gpu = d.loc[d["Feature"] == "GPU_Score"].iloc[0]
    others = d[~d["Feature"].isin(["GPU_Score"])].sort_values("Importance (drop in R2)")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"width_ratios": [1, 2.6]})
    a.bar(["GPU_Score"], [gpu["Importance (drop in R2)"]], yerr=[gpu["Std"]], color=P["primary"], width=0.5)
    a.text(0, gpu["Importance (drop in R2)"] * 1.02, f"{gpu['Importance (drop in R2)']:.2f}", ha="center", fontsize=9)
    a.set(title="GPU score", ylabel="Permutation importance (drop in R2)")
    a.set_ylim(0, gpu["Importance (drop in R2)"] * 1.15)
    k = 1e4                                           # show the zoomed panel in units of 1e-4
    band = 2 * noise["Std"] * k
    b.axvspan(noise["Importance (drop in R2)"] * k - band, noise["Importance (drop in R2)"] * k + band, color=P["soft_grey"], alpha=0.45, label="Random-noise column ±2 SD")
    colors = [P["bad"] if f == "Noise_Baseline" else P["primary"] for f in others["Feature"]]
    b.errorbar(others["Importance (drop in R2)"] * k, range(len(others)), xerr=others["Std"] * k, fmt="none", ecolor=P["soft_grey"], capsize=2)
    b.scatter(others["Importance (drop in R2)"] * k, range(len(others)), c=colors, s=40, zorder=3)
    b.set_yticks(range(len(others)), others["Feature"])
    b.axvline(0, color=P["ink"], lw=0.8)
    b.set(title="All other candidate features (zoomed)", xlabel="Permutation importance (drop in R2), units of 1e-4")
    b.legend(loc="lower right", fontsize=8)
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def confusion_chart(cm: np.ndarray, labels: list[str], title: str, figsize=(4.8, 4)) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else P["ink"])
    return fig
