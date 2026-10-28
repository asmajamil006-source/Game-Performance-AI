# AI Game Performance Predictor

An offline machine-learning application that predicts a game's frame rate (FPS) from hardware and game settings, explains each prediction, analyses CPU/GPU balance, simulates upgrades and recommends a graphics preset. Built with Python, scikit-learn and Streamlit. **No API keys, cloud services or internet connection are needed at runtime.**

## Overview

The project started as a basic regression demo. It has been rebuilt as an end-to-end AI/ML system: data audit, preprocessing pipeline, model comparison with proper validation, calibrated uncertainty, exact explanations, analytics and a multi-page dashboard.

A central design principle is **honesty about the data**. Before modelling, the dataset was audited, and the audit changed the design (see [What the data revealed](#what-the-data-revealed)). Every number shown in the application is computed from the dataset or from the trained model.

## Objectives

1. Predict FPS for a hardware/game configuration, with an uncertainty range.
2. Derive playability, performance level, a graphics-quality recommendation and upgrade advice from that prediction.
3. Explain *why* the model predicts what it does.
4. Evaluate models rigorously and select one on validation evidence.
5. Let users explore the dataset and inspect model quality and limitations.

## AI/ML approach

| Stage | What is done |
|---|---|
| Data audit | Missing values, duplicates, constants, class balance, formula/leakage checks, synthetic-data signs |
| Cleaning | Text normalisation, numeric coercion, duplicate and invalid-row removal, score repair from component names |
| Feature engineering | Hardware names → benchmark scores; resolution → megapixels; game one-hot encoded (never label-encoded) |
| Split | 80/20 stratified by GPU, fixed seed (train 4,000 / test 1,000) |
| Model selection | 6 algorithms × 2 feature sets, 5-fold CV on the training split, one-standard-error rule |
| Final check | Single evaluation on the held-out test set |
| Uncertainty | Empirical 80% interval from out-of-fold residuals; coverage verified on the test set |
| Explainability | Exact interventional Shapley values (local) and permutation importance (global) |
| Recommendations | Playability tiers, hardware balance, model-driven what-if upgrades, rule-based quality preset |

## Dataset

`game_performance_5000.csv`: 5,000 benchmark records, 18 columns, no missing values or duplicates.

* 6 CPUs, 6 GPUs, 6 games, 3 resolutions (720p, 1080p, 1440p), RAM 8/16/32 GB, VRAM 4-12 GB.
* Target: `FPS` (30-106, mean 67.2).
* **Excluded columns and why** (details in the app's *Data quality* tab):
  * `Will_Run`: constant (all 1), so no classifier can learn from it.
  * `Quality`: balanced but statistically unrelated to everything, including FPS.
  * `Load_Time_Sec`: measured after launch, not known at prediction time (leakage risk).
  * `Bottleneck_Pct`: exactly |GPU − CPU| / max(GPU, CPU) × 100 (max error 0.005); recomputed from the scores instead of used as a separate feature.

## What the data revealed

* **FPS is determined by GPU score** (permutation importance 1.51 R² drop) plus uniform random noise of about ±15 FPS.
* **No other input has a measurable effect** (game, resolution, RAM, VRAM, CPU, ray tracing, storage, OS, PSU, driver). Each one is statistically indistinguishable from a random-noise column (Bonferroni-adjusted p-values all ≥ 1.00).
* **`Quality` cannot be predicted**: four classifiers, even when given the measured FPS, score 26.6% at best against 25% chance.
* **VRAM is independent of the GPU model** (Cramér's V = 0.035), which real hardware would not be. Together with the points above, this indicates the data is synthetic.

Because of this, the application reports these facts openly instead of inventing game- or resolution-specific behaviour.

## Models and evaluation

Regression candidates: Linear, Ridge (RidgeCV), Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting (all regularised), plus a mean-prediction baseline. Each is evaluated on a *core* feature set (GPU + CPU score) and a *full* set (adds RAM, VRAM, ray tracing, resolution, game).

| Model | Features | CV R² (±SD) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| Linear Regression | Core hardware (GPU + CPU score) | 0.754 ± 0.019 | 7.81 | 9.00 | 0.759 |
| Ridge Regression | Core hardware (GPU + CPU score) | 0.754 ± 0.019 | 7.81 | 9.00 | 0.759 |
| Gradient Boosting | Core hardware (GPU + CPU score) | 0.753 ± 0.019 | 7.81 | 9.00 | 0.759 |
| Ridge Regression | Full configuration | 0.753 ± 0.018 | 7.83 | 9.01 | 0.759 |
| Linear Regression | Full configuration | 0.753 ± 0.018 | 7.83 | 9.01 | 0.759 |
| HistGradientBoosting | Core hardware (GPU + CPU score) | 0.753 ± 0.019 | 7.83 | 9.01 | 0.759 |
| Gradient Boosting | Full configuration | 0.753 ± 0.018 | 7.82 | 9.00 | 0.760 |
| Extra Trees | Core hardware (GPU + CPU score) | 0.751 ± 0.019 | 7.85 | 9.03 | 0.758 |
| Random Forest | Core hardware (GPU + CPU score) | 0.751 ± 0.019 | 7.85 | 9.03 | 0.758 |
| HistGradientBoosting | Full configuration | 0.750 ± 0.019 | 7.84 | 9.02 | 0.758 |
| Random Forest | Full configuration | 0.746 ± 0.020 | 7.94 | 9.14 | 0.752 |
| Extra Trees | Full configuration | 0.742 ± 0.019 | 8.06 | 9.31 | 0.743 |
| Baseline (always predict mean) | Core hardware (GPU + CPU score) | -0.001 ± 0.001 | 15.41 | 18.35 | -0.000 |

**Selected: Linear Regression on "Core hardware (GPU + CPU score)".** Linear Regression on 'Core hardware (GPU + CPU score)' reaches CV R2 = 0.754 (best observed: Linear Regression, 0.754). 11 of 12 candidates lie within one standard error (0.0084) of the best, so they are statistically indistinguishable; the simplest of them (2 features, lowest model complexity) was chosen. The mean-prediction baseline scores R2 = -0.001, confirming the model learns real structure.

Held-out test performance of the selected model: **MAE 7.81 FPS, MSE 81.04, RMSE 9.00, R² 0.759**. No tested model (including flexible ensembles using every feature) exceeded this, which suggests the remaining 24% of variance is irreducible noise. Extra Trees on all features is the only configuration that was measurably worse (CV R² 0.742), likely a sign of overfitting noise.

**Prediction interval:** the 80% interval (-12.3 to +12.2 FPS around the prediction) covered **79.0%** of test records.

**Playability tiers** (derived from FPS thresholds 90 / 60 / 45 / 30) evaluated on the test set: tier accuracy 65.2%, within one tier 100.0%, macro F1 0.562. For "reaches 60 FPS": accuracy 0.874, precision 1.000, recall 0.800, F1 0.889, ROC-AUC 0.945.

**Quality learnability experiment** (why quality is not an ML output):

| Classifier | CV accuracy | Test accuracy | Macro F1 | p-value vs chance |
|---|---|---|---|---|
| Logistic Regression | 0.249 | 0.266 | 0.266 | 0.129 |
| Decision Tree | 0.251 | 0.253 | 0.250 | 0.425 |
| Random Forest | 0.235 | 0.236 | 0.236 | 0.855 |
| Gradient Boosting | 0.253 | 0.233 | 0.231 | 0.900 |

## Features of the application

* **Prediction:** FPS with 80% range, playability tier, performance level (percentile among benchmark records), recommended preset, confidence in meeting your target FPS, "Why this prediction?" Shapley chart, FPS-outcome distribution, bottleneck analysis, upgrade what-if, dynamic recommendations, reality check against matching dataset records, downloadable report.
* **Dataset Analytics:** filters (game / resolution / GPU), FPS distribution, FPS by game / resolution / GPU / CPU, GPU score vs FPS, GPU × CPU heatmap, correlation matrix, bottleneck confounding analysis, quality distribution, cleaning report, column inventory, feature decisions.
* **Model Insights:** dataset size, inputs, all model results with status, selection reasoning, predicted-vs-actual, residuals, feature screening, confusion matrices, quality experiment, limitations.
* **About:** workflow and methodology in plain language.
* **Validation and errors:** unsupported hardware/games/resolutions and invalid numbers are rejected with clear messages; missing, empty, corrupt or incomplete datasets show friendly errors; corrupt or outdated model files are retrained automatically. No tracebacks are shown in the UI.

## How prediction works

1. Your CPU/GPU choice is mapped to the benchmark scores learned from the data (unsupported hardware is rejected, never guessed).
2. The same preprocessing pipeline used in training is applied.
3. The model predicts FPS (floored at 0, since negative FPS is impossible; predictions outside the benchmarked 30-106 range are flagged, not silently altered).
4. The interval, tier probabilities and target-confidence come from the model's out-of-fold errors.
5. Shapley values explain the result; the what-if table re-runs the model with one component swapped.
6. The quality preset comes from rules (see below) that use the predicted FPS.

**Quality rule engine** (rule-based by necessity; thresholds are documented assumptions in `config.py`): highest preset whose assumed cost leaves predicted FPS ≥ target × {Medium 1.0, High 1.25, Ultra 1.5}, then capped by VRAM and RAM guidelines.

## Project structure

```
app.py                  Streamlit dashboard (4 tabs)
config.py               Constants, feature sets, thresholds, documented assumptions
utils.py                Custom exceptions, logging, helpers
data_processor.py       Loading, cleaning, validation, audit, preprocessing pipeline
model_engine.py         Training, comparison, selection, screening, evaluation, prediction facade
explainability.py       Exact Shapley values, permutation importance
model_manager.py        Model persistence (joblib) with automatic invalidation
recommendations.py      Tiers, bottleneck analysis, quality rules, upgrades, recommendations
visualizations.py       Matplotlib charts
ui_components.py        HTML/CSS building blocks
assign3.py              Separate Selenium assignment (independent of the AI app)
tests/test_engine.py    21 unit tests
requirements.txt        Packages for the AI app
requirements-selenium.txt  Extra package for assign3.py only
.streamlit/config.toml  Light theme
```

## Installation and running

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first launch trains and evaluates all models (about 30 seconds) and saves `artifacts/fps_model_bundle.joblib`; later launches load it in about a second. The saved model is rebuilt automatically if the CSV, the scikit-learn version or the code schema changes. Command-line training and report: `python model_manager.py --retrain`.

Tests: `python -m unittest discover -s tests -v`

Selenium assignment (separate): `pip install -r requirements-selenium.txt` then `python assign3.py [picture.jpg]` (needs Chrome and internet).

## Limitations

* About 24% of FPS variance is unexplained; typical error is ±7.8 FPS.
* Only GPU score influences predictions. Resolution, game, RAM and others show no effect in this (apparently synthetic) data, unlike real games, so predictions must not be read as real-world hardware behaviour.
* Only 6 GPUs, 6 CPUs, 6 games and 3 resolutions are supported; 4K and other hardware are rejected.
* No record is below 30 FPS, so the "Not Recommended" tier is defined but never validated by examples.
* Graphics quality, balance threshold (10%) and VRAM/RAM guidelines are rule-based assumptions, not learned.
* Hyperparameters use regularised defaults rather than exhaustive tuning.
* Interpretive captions in the Analytics tab describe the shipped dataset; the numbers and model update automatically if the CSV changes, but caption wording does not.

## Future improvements

* Collect real benchmark data in which resolution, settings and CPU affect FPS, then retrain and re-screen features.
* Add per-game and quality-preset measurements so a quality model becomes learnable.
* Quantile or conformal regression for heteroscedastic uncertainty on richer data.
* Hyperparameter search once the data has more signal.

## Credits

Original project authors: Muzammil Ahmed (2023F-BSE-045) and Subhan Ahmed (2023F-BSE-031).

## Recent UI and reporting updates

- Appearance selector supports **System** (follows the operating system's preferred colour scheme), **Dark**, and **Light**.
- Prediction metric cards use consistent sizing and the page heading has additional top spacing to avoid clipping.
- Prediction reports can be exported as editable Word (`.docx`), Excel (`.xlsx`), or plain text (`.txt`) files. The Excel workbook includes the prediction summary, model explanation, upgrade scenarios, recommendations, and graphics-setting rule trace.
- The current browser session keeps the 20 most recent predictions and can export that history to CSV.
- Game-specific FPS limitations are disclosed: the included benchmark data did not demonstrate a measurable game-title effect, so the app does not claim calibrated per-game performance.

The Word/Excel export dependencies are included in `requirements.txt` (`python-docx` and `openpyxl`).


## Integrated AI auto-optimization advisor

The prediction page now includes an offline advisor attached to the existing model report. Choose **Balanced gameplay** (60 FPS), **Smoother performance** (90 FPS), **Higher visual quality** (60 FPS with the rule-based quality preset), or a custom whole-number target from 15–360 FPS. The existing residual-derived prediction interval is generated automatically; users do not enter interval boundaries.

The advisor combines the existing model prediction and uncertainty interval with clearly labelled rule-based graphics-preset and hardware-balance advice. Alternative CPU/GPU configurations are re-predicted by the existing model and are explicitly estimates rather than guaranteed real-world gains. It does not change operating-system settings, drivers, overclocking, or game files. Advisor goal and summary are included in TXT, Word, and Excel reports.

**Evidence limitation:** the supplied 5,000-row dataset has a narrow recorded FPS range (30–106 FPS) and appears synthetic in some hardware/feature relationships. Predictions outside the model's training range should be treated cautiously. The dataset does not provide controlled per-graphics-setting benchmark sweeps, so the advisor cannot claim validated FPS gains for individual graphics options. The quality preset remains rule-based because the existing quality-label experiment found no learnable signal.
