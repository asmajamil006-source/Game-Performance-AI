"""AI Game Performance Predictor - Streamlit dashboard.

Run with:  streamlit run app.py
Everything runs locally; no internet connection or API key is needed.

Original project authors:
    Muzammil Ahmed (2023F-BSE-045)
    Subhan Ahmed   (2023F-BSE-031)
"""
from __future__ import annotations

from datetime import datetime
import pandas as pd
import streamlit as st

import config as cfg
import data_processor as dp
import recommendations as rec
import ui_components as ui
import visualizations as viz
from model_engine import comparison_display
from model_manager import load_or_train
from report_exports import build_excel, build_word
from utils import GamePerfError, InvalidInputError, get_logger

log = get_logger()
P = cfg.PALETTE

st.set_page_config(page_title="AI Game Performance Predictor", page_icon="🎮", layout="wide")
st.markdown(ui.inject_css(st.session_state.get("theme_mode", "System")), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached resources: the model is trained once and then re-used.
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing the AI model. The first run trains and evaluates 12 model configurations (about 30 seconds)...")
def get_predictor():
    return load_or_train()


@st.cache_data(show_spinner=False)
def get_dataset() -> tuple[pd.DataFrame, dp.CleaningReport]:
    return dp.load_clean_dataset()


CORRELATE_COLUMNS = ["GPU_Score", "CPU_Score", "RAM_GB", "VRAM_GB", "Ray_Tracing", "Resolution_MP", "PSU_Watts",
                     "Driver_Version", "Load_Time_Sec", "Bottleneck_Pct"]


def strongest_correlate(frame: pd.DataFrame) -> tuple[str, float]:
    """Numeric column most correlated with FPS in the filtered data (computed, never assumed)."""
    best = ("n/a", 0.0)
    for col in CORRELATE_COLUMNS:
        r = viz.safe_corr(frame[col], frame[cfg.TARGET])
        if r == r and abs(r) > abs(best[1]):          # r == r filters NaN
            best = (col.replace("_", " "), r)
    return best


def show_png(png: bytes) -> None:
    st.image(png)


def card_chart(title: str, png: bytes, caption: str = "") -> None:
    with st.container(border=True):
        if title:
            st.markdown(f"**{title}**")
        show_png(png)
        if caption:
            st.caption(caption)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load everything, or show a friendly error state (never a traceback).
# ----------------------------------------------------------------------------
try:
    predictor = get_predictor()
    data, cleaning = get_dataset()
except GamePerfError as err:
    html(ui.hero("AI Game Performance Predictor", "The application could not start.", ["Setup problem"]))
    st.error(str(err))
    st.info("Check that game_performance_5000.csv is in the project folder, then reload the page.")
    st.stop()
except Exception:                                   # unexpected: log details, show a calm message
    log.exception("startup failure")
    html(ui.hero("AI Game Performance Predictor", "The application could not start.", ["Unexpected error"]))
    st.error("Something went wrong while preparing the model. Details were written to the console log.")
    st.stop()

bundle = predictor.b
catalog = predictor.catalog
metrics = bundle["test_metrics"]

_NICE = {"GPU_Score": "GPU score", "CPU_Score": "CPU score", "RAM_GB": "RAM", "VRAM_GB": "VRAM", "Ray_Tracing": "ray tracing",
         "Resolution_MP": "resolution", "PSU_Watts": "PSU wattage", "Driver_Version": "driver version", "Storage_Type": "storage type",
         "OS": "operating system", "Game": "game"}
_screen = bundle["screening"]
INFORMATIVE = [_NICE.get(f, f) for f in _screen.loc[_screen["Verdict"].isin(["Dominant driver", "Informative"]), "Feature"]]
NO_EFFECT = [_NICE.get(f, f) for f in _screen.loc[_screen["Verdict"] == "No measurable effect", "Feature"]]
TOP_DRIVER = _NICE.get(bundle["importance"].iloc[0]["Feature"], bundle["importance"].iloc[0]["Feature"])


@st.cache_data(show_spinner=False)
def analytics_chart(kind: str, games: tuple, resolutions: tuple, gpus: tuple) -> bytes:
    """Render (and cache) one analytics chart for the selected filters."""
    frame, _ = get_dataset()
    frame = frame[frame["Game"].isin(games) & frame["Resolution"].isin(resolutions) & frame["GPU"].isin(gpus)]
    cat = catalog
    figures = {
        "distribution": lambda: viz.fps_distribution(frame),
        "by_game": lambda: viz.fps_boxplot(frame, "Game", cat["games"], "FPS by game"),
        "by_resolution": lambda: viz.fps_boxplot(frame, "Resolution", cat["resolutions"], "FPS by resolution"),
        "by_gpu": lambda: viz.fps_boxplot(frame, "GPU", list(cat["gpu"]), "FPS by GPU (weakest to strongest)"),
        "by_cpu": lambda: viz.fps_boxplot(frame, "CPU", list(cat["cpu"]), "FPS by CPU (weakest to strongest)"),
        "gpu_scatter": lambda: viz.gpu_vs_fps(frame),
        "heatmap": lambda: viz.cpu_gpu_heatmap(frame),
        "correlation": lambda: viz.correlation_heatmap(frame),
        "quality": lambda: viz.quality_panels(frame),
        "bottleneck": lambda: viz.bottleneck_confounding(frame),
    }
    return viz.fig_to_png(figures[kind]())


@st.cache_data(show_spinner=False)
def model_chart(kind: str) -> bytes:
    """Render (and cache) the static Model Insights charts."""
    b, iv = bundle, bundle["interval"]
    figures = {
        "comparison": lambda: viz.model_comparison_chart(b["comparison"], b["selected_row"]),
        "pred_actual": lambda: viz.predicted_vs_actual(b["y_test"], b["pred_test"]),
        "residuals": lambda: viz.residual_hist(iv["residuals"], iv["q_lo"], iv["q_hi"], iv["coverage_nominal"]),
        "screening": lambda: viz.screening_chart(b["screening"]),
        "tier_cm": lambda: viz.confusion_chart(b["tiers"]["confusion"], b["tiers"]["labels"], "Actual vs predicted tier"),
        "quality_cm": lambda: viz.confusion_chart(b["quality_experiment"]["confusion"], b["quality_experiment"]["labels"],
                                                  b["quality_experiment"]["best_classifier"]),
    }
    return viz.fig_to_png(figures[kind]())


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
html(ui.hero(
    "AI Game Performance Predictor",
    "Machine-learning system that predicts FPS from your hardware, explains the result and recommends how to improve it. "
    "Trained and evaluated locally on 5,000 benchmark records.",
    [f"Model: {bundle['model_name']}", f"Test R² {metrics['R2']:.2f}", f"Typical error ±{metrics['MAE']:.1f} FPS (MAE)",
     f"{bundle['n_rows']:,} records", "Runs offline"],
))

# ----------------------------------------------------------------------------
# Sidebar: configuration form
# ----------------------------------------------------------------------------
with st.sidebar:
    html(ui.sidebar_brand())
    st.selectbox("Appearance", ["System", "Dark", "Light"], key="theme_mode",
                 help="System follows your operating system's light/dark preference.")
    with st.form("config_form"):
        st.markdown("**Hardware**")
        cpu = st.selectbox("CPU", list(catalog["cpu"]), index=2, help="Processor models present in the training data.")
        gpu = st.selectbox("GPU", list(catalog["gpu"]), index=2, help="Graphics cards present in the training data.")
        ram = st.select_slider("RAM", options=catalog["ram_gb"], value=16, format_func=lambda v: f"{v} GB")
        vram = st.select_slider("VRAM", options=catalog["vram_gb"], value=8, format_func=lambda v: f"{v} GB")
        st.markdown("**Game settings**")
        game = st.selectbox("Game", catalog["games"])
        resolution = st.selectbox("Resolution", catalog["resolutions"], index=min(1, len(catalog["resolutions"]) - 1))
        goal = st.selectbox("Performance goal", ["Balanced gameplay", "Smoother performance", "Higher visual quality", "Custom FPS target"],
                            help="Select a goal or enter a specific target FPS. The advisor does not change system settings.")
        if goal == "Custom FPS target":
            target = st.number_input("Target FPS", min_value=15, max_value=360, value=int(cfg.DEFAULT_TARGET_FPS), step=5,
                                     help="Choose a target from 15 to 360 FPS.")
        else:
            target, _ = rec.resolve_performance_goal(goal)
            st.caption(f"Advisor target: {target} FPS")
        with st.expander("Advanced settings"):
            ray_tracing = st.checkbox("Ray tracing enabled", value=False)
        submitted = st.form_submit_button("Predict performance", type="primary")
    if st.button("Clear results"):
        st.session_state.pop("result", None)
        st.session_state.pop("input_error", None)
    if st.session_state.get("prediction_history"):
        with st.expander("Recent predictions", expanded=False):
            history_frame = pd.DataFrame(st.session_state["prediction_history"])
            st.dataframe(history_frame, hide_index=True, use_container_width=True)
            st.download_button("Export history (CSV)", history_frame.to_csv(index=False).encode("utf-8"),
                               file_name="prediction_history.csv", mime="text/csv", use_container_width=True)
    st.divider()
    html(ui.status_box([
        f"Model: {bundle['model_name']}",
        f"Trained on {bundle['n_train']:,} rows, tested on {bundle['n_test']:,}",
        f"Test MAE {metrics['MAE']:.1f} FPS, R² {metrics['R2']:.2f}",
        "Local model, no external services",
    ]))

if submitted:
    try:
        system = predictor.validate(cpu, gpu, ram, vram, game, resolution, ray_tracing)
        report = rec.build_report(predictor, system, int(target), data)
        _, advisor_goal = rec.resolve_performance_goal(goal, target if goal == "Custom FPS target" else None)
        rec.build_advisor_summary(report, advisor_goal)
        st.session_state["result"] = {
            "report": report,
            "charts": {
                "explain": viz.fig_to_png(viz.explanation_chart(report.explanation, report.prediction.fps)),
                "range": viz.fig_to_png(viz.uncertainty_chart(report.prediction, report.target_fps)),
                "balance": viz.fig_to_png(viz.balance_chart(report.bottleneck)),
                "upgrades": viz.fig_to_png(viz.upgrade_chart(report.upgrades)),
            },
        }
        history = st.session_state.setdefault("prediction_history", [])
        history.insert(0, {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Game": report.config.game,
            "CPU": report.config.cpu,
            "GPU": report.config.gpu,
            "RAM (GB)": report.config.ram_gb,
            "VRAM (GB)": report.config.vram_gb,
            "Resolution": report.config.resolution,
            "Predicted FPS": round(report.prediction.fps, 1),
            "Target FPS": report.target_fps,
            "Preset": report.quality.level,
        })
        del history[20:]
        st.session_state.pop("input_error", None)
    except InvalidInputError as err:
        st.session_state["input_error"] = str(err)
        st.session_state.pop("result", None)
    except Exception:
        log.exception("prediction failed")
        st.session_state["input_error"] = "The prediction could not be completed. Please try different settings."
        st.session_state.pop("result", None)


# ----------------------------------------------------------------------------
# Tab: Prediction
# ----------------------------------------------------------------------------
def render_empty_state() -> None:
    html(ui.section("Get started", "Three steps to a prediction"))
    c1, c2, c3 = st.columns(3)
    with c1:
        html(ui.step(1, "Describe your system", "Pick a CPU, GPU, RAM and VRAM in the sidebar."))
    with c2:
        html(ui.step(2, "Choose game and resolution", "Select the title, resolution and the frame rate you want to reach."))
    with c3:
        html(ui.step(3, "Predict", "Press Predict performance to get FPS, an explanation and recommendations."))
    html(ui.section("What the model learned", "Measured on a held-out test set the model never saw during training"))
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        html(ui.kpi("Explained variance", f"{metrics['R2']:.0%}", "Test R² of the selected model"))
    with k2:
        html(ui.kpi("Typical error", f"±{metrics['MAE']:.1f} FPS", f"Mean absolute error; RMSE {metrics['RMSE']:.1f}"))
    with k3:
        html(ui.kpi("Main driver", TOP_DRIVER.capitalize(), "Highest permutation importance on the test set"))
    with k4:
        html(ui.kpi("Interval coverage", f"{bundle['interval']['coverage_test']:.0%}", f"Observed on test data (nominal {bundle['interval']['coverage_nominal']:.0%})"))


def render_result(result: dict) -> None:
    r: rec.PerformanceReport = result["report"]
    charts = result["charts"]
    c, p = r.config, r.prediction
    html(ui.section(f"Prediction: {c.game} at {c.resolution}",
                    f"{c.gpu} + {c.cpu}, {c.ram_gb} GB RAM, {c.vram_gb} GB VRAM, target {r.target_fps} FPS"))

    for warning in p.warnings:
        st.warning(warning)
    st.info("Game-specific limitation: the available benchmark data did not show a measurable FPS effect from the game title after accounting for the other inputs. The selected game is recorded in this report, but the estimate is not independently calibrated for each game.")

    k = st.columns(5)
    with k[0]:
        html(ui.kpi("Predicted FPS", f"{p.fps:.0f}", f"{p.coverage:.0%} range: {p.low:.0f} to {p.high:.0f} FPS", P["primary_dark"]))
    with k[1]:
        html(ui.kpi("Playability", r.tier.name, r.tier.description, r.tier.color))
    with k[2]:
        html(ui.kpi("Performance level", r.level, f"Faster than {r.percentile:.0f}% of benchmark records", P["ink"]))
    with k[3]:
        limit = f"limited by {r.quality.limited_by}" if r.quality.limited_by else "limited by FPS headroom"
        html(ui.kpi("Recommended quality", r.quality.level, f"Rule-based, {limit}", P["accent"]))
    with k[4]:
        verdict = "meets" if p.fps >= r.target_fps else "misses"
        color = {"High": P["good"], "Moderate": P["warn"], "Low": P["bad"]}[r.confidence_label]
        html(ui.kpi("Confidence", f"{r.confidence_label} ({r.confidence:.0%})", f"That the system {verdict} {r.target_fps} FPS", color))

    # ---- AI insights ------------------------------------------------------
    html(ui.section("AI insights", "Why the model predicts what it does"))
    left, right = st.columns(2, gap="large")
    with left:
        card_chart("", charts["explain"],
                   "Exact Shapley values: each bar is the FPS change caused by that input compared with an average system.")
        table = r.explanation.contributions[["Feature", "Contribution", "Impact"]].rename(columns={"Contribution": "Contribution (FPS)"})
        st.dataframe(table, hide_index=True, column_config={"Contribution (FPS)": st.column_config.NumberColumn(format="%+.2f")})
        if r.ignored_inputs:
            names = ", ".join(i["Input"] for i in r.ignored_inputs)
            worst_p = min(i["p (Bonferroni)"] for i in r.ignored_inputs)
            html(ui.callout("info", "Data finding: some inputs do not change FPS here",
                            f"In this dataset, {names} showed no measurable effect on FPS once GPU score is accounted for "
                            f"(Bonferroni-adjusted p >= {worst_p:.2f}). Changing them leaves the prediction essentially unchanged. "
                            "See Model Insights for the evidence.", "Data"))
    with right:
        card_chart("", charts["range"],
                   f"Built from the model's out-of-fold errors; {bundle['interval']['coverage_test']:.0%} of test records fell inside the {p.coverage:.0%} interval.")
        probs = pd.DataFrame({"Playability tier": list(r.tier_probabilities), "Probability (%)": [v * 100 for v in r.tier_probabilities.values()]})
        st.dataframe(probs[probs["Probability (%)"] > 0], hide_index=True,
                     column_config={"Probability (%)": st.column_config.NumberColumn(format="%.0f")})
        ev = r.evidence
        st.caption(f"Reality check: {ev['n']:,} benchmark records have the {ev['match']}; their mean FPS is {ev['mean']:.0f} "
                   f"(observed range {ev['min']:.0f} to {ev['max']:.0f}).")

    # ---- Hardware balance & upgrades -------------------------------------
    html(ui.section("Hardware analysis", "CPU/GPU balance and model-predicted upgrade effects"))
    left, right = st.columns(2, gap="large")
    with left:
        card_chart("Bottleneck analysis", charts["balance"], r.bottleneck.summary)
        st.caption(f"Balance is classified with the dataset's own definition (|GPU - CPU| / max score). "
                   f"Pairs within {cfg.BALANCED_THRESHOLD_PCT:.0f}% are treated as balanced.")
    with right:
        card_chart("", charts["upgrades"], "Each bar re-runs the trained model with only that component changed.")
        if len(r.upgrades):
            shown = r.upgrades[["Component", "Upgrade to", "Predicted FPS", "Change (FPS)", "Balance after"]]
            st.dataframe(shown, hide_index=True, column_config={
                "Predicted FPS": st.column_config.NumberColumn(format="%.0f"),
                "Change (FPS)": st.column_config.NumberColumn(format="%+.1f")})

    # ---- Quality ----------------------------------------------------------
    html(ui.section("Graphics quality recommendation", "Transparent rule engine fed by the ML-predicted FPS"))
    q = r.quality
    st.dataframe(pd.DataFrame(q.trace)[["Rule", "Check", "Outcome"]], hide_index=True)
    exp = bundle["quality_experiment"]
    st.caption(f"Why rules instead of a trained model? In this dataset the Quality label has no learnable relationship with hardware or FPS: "
               f"the best of four classifiers scored {exp['best_test_accuracy']:.1%} accuracy against {exp['chance_level']:.0%} chance. "
               "Thresholds are documented assumptions in config.py, not values learned from data.")

    # ---- Integrated offline advisor ---------------------------------------
    html(ui.section("AI auto-optimization advisor", "Evidence-bounded guidance integrated with this prediction"))
    st.caption(f"Goal: {r.advisor_goal} · Predicted {p.fps:.1f} FPS · model interval {p.low:.1f}–{p.high:.1f} FPS ({p.coverage:.0%} nominal coverage)")
    st.info(r.advisor_summary or rec.build_advisor_summary(r, r.advisor_goal)[0])
    st.caption("Evidence labels: FPS and interval are model output; graphics preset and CPU/GPU balance are rule-based; what-if upgrade deltas are model estimates, not guaranteed real-world gains.")
    if p.outside_training_range:
        st.warning("This configuration is outside the model's reliable training range. Avoid treating the prediction or upgrade ranking as a dependable optimization result.")
    if r.upgrades is not None and not r.upgrades.empty:
        feasible = r.upgrades[r.upgrades["Predicted FPS"] >= r.target_fps].sort_values("Predicted FPS")
        if len(feasible):
            st.markdown("**Model-estimated alternatives that reach the target**")
            st.dataframe(feasible[["Component", "Upgrade to", "Predicted FPS", "Change (FPS)", "Balance after"]].head(5),
                         hide_index=True, column_config={"Predicted FPS": st.column_config.NumberColumn(format="%.1f"),
                                                         "Change (FPS)": st.column_config.NumberColumn(format="%+.1f")})
        else:
            st.caption("No tested single-component upgrade in the available hardware catalogue is predicted to reach this target.")
    st.caption("No system, driver, overclock, or game-file changes are applied automatically. Apply suggestions manually and verify with an in-game benchmark.")

    # ---- Recommendations --------------------------------------------------
    html(ui.section("Recommendations", "Generated from this specific prediction and configuration"))
    html("".join(ui.callout(x.kind, x.title, x.body, x.source) for x in r.recommendations))
    st.markdown("**Export this prediction**")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.download_button("Download TXT", rec.report_to_text(r), file_name="game_performance_report.txt", mime="text/plain", use_container_width=True)
    with ex2:
        try:
            st.download_button("Download Word", build_word(r), file_name="game_performance_report.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        except Exception:
            log.exception("Word report export failed")
            st.error("Word export is unavailable. Install the project dependencies and try again.")
    with ex3:
        try:
            st.download_button("Download Excel", build_excel(r), file_name="game_performance_report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        except Exception:
            log.exception("Excel report export failed")
            st.error("Excel export is unavailable. Install the project dependencies and try again.")


tabs = st.tabs(["Prediction", "Dataset Analytics", "Model Insights", "About"])

with tabs[0]:
    if st.session_state.get("input_error"):
        st.error(st.session_state["input_error"])
    if st.session_state.get("result"):
        render_result(st.session_state["result"])
    else:
        render_empty_state()


# ----------------------------------------------------------------------------
# Tab: Dataset analytics
# ----------------------------------------------------------------------------
with tabs[1]:
    html(ui.section("Dataset analytics", "Explore the benchmark records the model was trained on"))
    f1, f2, f3 = st.columns(3)
    with f1:
        sel_games = st.multiselect("Games", catalog["games"], default=catalog["games"])
    with f2:
        sel_res = st.multiselect("Resolutions", catalog["resolutions"], default=catalog["resolutions"])
    with f3:
        sel_gpus = st.multiselect("GPUs", list(catalog["gpu"]), default=list(catalog["gpu"]))
    flt = data[data["Game"].isin(sel_games) & data["Resolution"].isin(sel_res) & data["GPU"].isin(sel_gpus)]
    if flt.empty:
        st.warning("No records match the selected filters. Select at least one value in each filter.")
    else:
        m = st.columns(4)
        with m[0]:
            html(ui.kpi("Records", f"{len(flt):,}", f"of {len(data):,} total"))
        with m[1]:
            html(ui.kpi("Mean FPS", f"{flt[cfg.TARGET].mean():.1f}", f"Median {flt[cfg.TARGET].median():.0f}"))
        with m[2]:
            html(ui.kpi("FPS range", f"{flt[cfg.TARGET].min():.0f} to {flt[cfg.TARGET].max():.0f}", f"Std. deviation {flt[cfg.TARGET].std():.1f}"))
        with m[3]:
            name, r_val = strongest_correlate(flt)
            html(ui.kpi("Strongest correlate", name, f"r = {r_val:+.2f} with FPS" if name != "n/a" else "Filters leave no varying column"))

    key = (tuple(sel_games), tuple(sel_res), tuple(sel_gpus))
    if not flt.empty:
        sub = st.tabs(["Performance", "Hardware", "Relationships", "Data quality"])
        with sub[0]:
            a, b = st.columns(2, gap="large")
            with a:
                card_chart("", analytics_chart("distribution", *key))
                card_chart("", analytics_chart("by_resolution", *key),
                           "Resolution shows no systematic FPS difference in this dataset.")
            with b:
                card_chart("", analytics_chart("by_game", *key), "Game titles perform almost identically in this dataset.")
                card_chart("Recorded Quality label", analytics_chart("quality", *key), "Quality is balanced across classes and unrelated to FPS.")
        with sub[1]:
            a, b = st.columns(2, gap="large")
            with a:
                card_chart("", analytics_chart("by_gpu", *key), "The clearest staircase in the data: stronger GPU, higher FPS.")
                card_chart("", analytics_chart("gpu_scatter", *key))
            with b:
                card_chart("", analytics_chart("by_cpu", *key), "CPU models show no systematic difference.")
                card_chart("", analytics_chart("heatmap", *key), "Rows differ strongly (GPU); columns barely differ (CPU).")
        with sub[2]:
            a, b = st.columns(2, gap="large")
            with a:
                card_chart("Correlation analysis", analytics_chart("correlation", *key))
            with b:
                card_chart("Bottleneck analysis", analytics_chart("bottleneck", *key),
                           "Bottleneck % looks harmful (left) only because stronger GPUs create bigger gaps. "
                           "Within each GPU there is no effect (right), so it is a confounded signal, not a cause.")
        with sub[3]:
            audit = bundle["audit"]
            a, b = st.columns(2, gap="large")
            with a:
                st.markdown("**Cleaning report**")
                st.dataframe(bundle["cleaning"].as_table(), hide_index=True)
                st.markdown("**Audit findings**")
                st.markdown(
                    f"- **Will_Run** is constant ({audit['will_run_counts']}), so no classifier can be trained on it.\n"
                    f"- **Quality** is balanced (smallest/largest class = {audit['quality_balance_ratio']:.2f}) but has no learnable signal.\n"
                    f"- **Bottleneck_Pct** equals |GPU - CPU| / max(GPU, CPU) x 100 (max error {audit['bottleneck_formula_max_error']:.3f}).\n"
                    f"- **Hardware names** map one-to-one to benchmark scores: {'yes' if audit['hardware_name_score_one_to_one'] else 'no'}.\n"
                    f"- **VRAM vs GPU model** association (Cramér's V) = {audit['vram_gpu_cramers_v']:.3f}; VRAM is effectively independent of the GPU, "
                    "which real hardware would not be. This indicates synthetic data.\n"
                    f"- **Duplicates in raw file:** {audit['duplicates_in_raw']}."
                )
            with b:
                st.markdown("**Column inventory**")
                st.dataframe(audit["columns"], hide_index=True)
            st.markdown("**Feature decisions**")
            decisions = pd.DataFrame(cfg.FEATURE_DECISIONS).rename(columns={"column": "Column", "role": "Role", "reason": "Reasoning"})
            st.dataframe(decisions, hide_index=True)


# ----------------------------------------------------------------------------
# Tab: Model insights
# ----------------------------------------------------------------------------
with tabs[2]:
    html(ui.section("Model insights", "How the AI was trained, evaluated and selected"))
    comp = comparison_display(bundle)
    n_models = int((bundle["comparison"]["complexity"] >= 0).sum())
    k = st.columns(5)
    with k[0]:
        html(ui.kpi("Training / test rows", f"{bundle['n_train']:,} / {bundle['n_test']:,}", "80/20 stratified split, fixed seed"))
    with k[1]:
        html(ui.kpi("Model inputs", str(len(predictor.features)), ", ".join(predictor.features)))
    with k[2]:
        html(ui.kpi("Configurations tested", str(n_models), "6 algorithms x 2 feature sets, plus a baseline"))
    with k[3]:
        html(ui.kpi("Selected model", bundle["model_name"], bundle["feature_set"]["label"]))
    with k[4]:
        html(ui.kpi("Test R² / MAE", f"{metrics['R2']:.3f} / {metrics['MAE']:.1f}", f"RMSE {metrics['RMSE']:.1f} FPS, MSE {metrics['MSE']:.1f}"))
    html(ui.callout("info", "Why this model was selected", bundle["selection_reason"], "Model"))

    ins = st.tabs(["Model comparison", "Accuracy and errors", "Feature importance", "Playability and quality", "Limitations"])
    with ins[0]:
        st.dataframe(comp, hide_index=True, column_config={
            "CV R²": st.column_config.NumberColumn(format="%.4f"), "CV R² SD": st.column_config.NumberColumn(format="%.4f"),
            "CV MAE": st.column_config.NumberColumn(format="%.2f"), "CV RMSE": st.column_config.NumberColumn(format="%.2f"),
            "Test MAE": st.column_config.NumberColumn(format="%.2f"), "Test RMSE": st.column_config.NumberColumn(format="%.2f"),
            "Test R²": st.column_config.NumberColumn(format="%.4f")})
        st.caption("Selection used cross-validation on the training split only. Test columns come from a single evaluation on held-out data and "
                   "were not used to choose the model.")
        card_chart("", model_chart("comparison"),
                   "Almost every configuration lands inside the same error bar: the data has one dominant signal (GPU score) plus irreducible noise. "
                   "Flexible models with all features do no better and some do slightly worse.")
    with ins[1]:
        iv = bundle["interval"]
        a, b = st.columns(2, gap="large")
        with a:
            card_chart("Predicted vs actual FPS", model_chart("pred_actual"),
                       "Vertical stripes appear because the model predicts one value per GPU; spread within a stripe is unexplained noise.")
        with b:
            card_chart("Prediction errors", model_chart("residuals"),
                       f"Errors are spread evenly between {iv['residual_min']:.1f} and {iv['residual_max']:+.1f} FPS. "
                       f"The {iv['coverage_nominal']:.0%} interval covered {iv['coverage_test']:.1%} of test records.")
        mm = st.columns(4)
        for col, name in zip(mm, ["MAE", "MSE", "RMSE", "R2"]):
            with col:
                html(ui.kpi(f"Test {name.replace('R2', 'R²')}", f"{metrics[name]:.3f}" if name == "R2" else f"{metrics[name]:.2f}",
                            "FPS" if name == "MAE" else ("FPS²" if name == "MSE" else ("FPS" if name == "RMSE" else "Held-out test set"))))
        if "coefficients" in bundle:
            coef = pd.DataFrame({"Standardised input": list(bundle["coefficients"]), "Coefficient (FPS per 1 SD)": list(bundle["coefficients"].values())})
            st.markdown("**Learned coefficients** (the selected model is linear, so these are its complete logic)")
            st.dataframe(coef, hide_index=True, column_config={"Coefficient (FPS per 1 SD)": st.column_config.NumberColumn(format="%+.3f")})
    with ins[2]:
        scr = bundle["screening"]
        card_chart("Feature screening: permutation importance vs a random-noise column",
                   model_chart("screening"),
                   "A feature is informative only if shuffling it hurts accuracy by more than shuffling pure noise does. Only GPU score passes.")
        show = scr[["Feature", "Importance (drop in R2)", "Test", "Statistic", "p (Bonferroni)", "Verdict"]].copy()
        show["Importance (drop in R2)"] = show["Importance (drop in R2)"].astype(float)
        st.dataframe(show, hide_index=True, column_config={
            "Importance (drop in R2)": st.column_config.NumberColumn(format="%.5f"),
            "Statistic": st.column_config.NumberColumn(format="%.4f"),
            "p (Bonferroni)": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Residual tests ask whether a feature explains FPS left over after accounting for GPU score "
                   "(Pearson r for numeric, ANOVA F for categorical; Bonferroni-corrected). Bottleneck_Pct, Load_Time_Sec and Quality are informational only.")
    with ins[3]:
        t, qx = bundle["tiers"], bundle["quality_experiment"]
        st.markdown("**Playability tiers derived from predicted FPS** (evaluated on the test set)")
        a, b = st.columns(2, gap="large")
        with a:
            card_chart("Tier confusion matrix", model_chart("tier_cm"),
                       "No record falls below 30 FPS, so 'Not Recommended' never occurs in the benchmark data.")
        with b:
            r60 = t["reach60"]
            tbl = pd.DataFrame({"Metric": ["Tier accuracy", "Within one tier", "Macro precision", "Macro recall", "Macro F1",
                                           "Reach 60 FPS: accuracy", "Reach 60 FPS: precision", "Reach 60 FPS: recall", "Reach 60 FPS: F1", "Reach 60 FPS: ROC-AUC"],
                                "Value": [t["accuracy"], t["within_one_tier"], t["macro_precision"], t["macro_recall"], t["macro_f1"],
                                          r60["accuracy"], r60["precision"], r60["recall"], r60["f1"], r60["roc_auc"]]})
            st.dataframe(tbl, hide_index=True, column_config={"Value": st.column_config.NumberColumn(format="%.3f")})
            st.caption("Tiers are defined by FPS thresholds (90, 60, 45, 30). Will_Run is constant, so no classifier is trained for it.")
        st.markdown("**Quality learnability experiment** (why graphics quality is not predicted by ML)")
        a, b = st.columns(2, gap="large")
        with a:
            st.dataframe(qx["table"], hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="%.3f") for c in qx["table"].columns if c != "Classifier"})
        with b:
            card_chart("Best classifier confusion matrix", model_chart("quality_cm"))
        verdict = "no classifier beats chance" if not qx["learnable"] else "at least one classifier beats chance"
        st.caption(f"Even when given the measured FPS, {verdict} (chance = {qx['chance_level']:.0%}; best test accuracy {qx['best_test_accuracy']:.1%}). "
                   "A deployed Quality model would only reproduce noise, so a labelled rule engine is used instead.")
    with ins[4]:
        html("".join([
            ui.callout("warning", "Most variation is not explained", f"About {1 - metrics['R2']:.0%} of FPS variance is unexplained. No tested model did better, which points to noise in the data rather than a weak algorithm."),
            ui.callout("warning", "Few inputs carry signal",
                       f"Measurable influence on FPS: {', '.join(INFORMATIVE) or 'none'}. No measurable effect: {', '.join(NO_EFFECT) or 'none'}. "
                       "Real games do respond to resolution and settings, so this dataset appears synthetic and does not represent real hardware behaviour."),
            ui.callout("warning", "Small hardware catalogue", f"Only {len(catalog['gpu'])} GPUs, {len(catalog['cpu'])} CPUs and {len(catalog['resolutions'])} resolutions exist. Other hardware and 4K cannot be predicted and are rejected rather than guessed."),
            ui.callout("warning", "No FPS below 30 in the data", "The model has never seen unplayable results, so 'Not Recommended' is defined by threshold but not validated by examples."),
            ui.callout("info", "Quality and bottleneck advice are rule-based", "Preset thresholds, VRAM/RAM guidelines and the 10% balance threshold are documented assumptions, not learned values."),
            ui.callout("info", "Fixed hyperparameters", "Models use sensible regularised defaults rather than exhaustive tuning; the data's noise ceiling makes tuning unlikely to change the conclusion."),
        ]))


# ----------------------------------------------------------------------------
# Tab: About
# ----------------------------------------------------------------------------
with tabs[3]:
    html(ui.section("What is this project?"))
    st.markdown(
        "An AI-based game performance prediction system. It learns from 5,000 benchmark records how hardware relates to frame rate, "
        "then predicts FPS for a configuration you choose, explains the prediction, and recommends settings and upgrades. "
        "All training and prediction happen on your computer."
    )
    html(ui.section("How does it work?"))
    html(ui.flow(["Dataset", "Data preprocessing", "Feature engineering", "Model training", "Model evaluation", "Prediction", "AI explanation", "Recommendation"]))
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(
            "**1. Dataset and preprocessing.** The CSV is checked for missing values, duplicates, invalid numbers and inconsistent text. "
            "Categories such as game titles are one-hot encoded so no false order is created; numbers are scaled where it matters.\n\n"
            "**2. Feature engineering.** Hardware names are replaced by their benchmark scores, and resolution becomes megapixels.\n\n"
            "**3. Model training and evaluation.** Six algorithms are compared on two feature sets with 5-fold cross-validation. "
            "The simplest model that is statistically tied for best is selected, then checked once on held-out test data."
        )
    with b:
        st.markdown(
            "**4. Prediction.** Your choices go through the identical preprocessing and the model returns FPS. "
            "An interval is attached using the model's own cross-validated errors.\n\n"
            "**5. Explanation.** Exact Shapley values show how much each input moved the prediction.\n\n"
            "**6. Recommendation.** Playability tiers, hardware balance, what-if upgrades and a rule-based graphics preset are generated "
            "from the prediction and your configuration."
        )
    html(ui.section("What the data revealed"))
    st.markdown(
        "Checking the data before modelling changed the design: `Will_Run` is constant, `Quality` is unpredictable, and "
        f"only {', '.join(INFORMATIVE) or 'no input'} showed a measurable effect on FPS. Instead of faking classifiers or explanations, "
        "the app reports these facts and builds recommendations only from what the data supports."
    )
    html(ui.section("Technology"))
    st.markdown("Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Streamlit and joblib. No external APIs, cloud services or keys are used.")
