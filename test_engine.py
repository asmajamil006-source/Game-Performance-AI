"""Tests for the data pipeline, model, explanations and recommendations.

Run from the project folder:   python -m unittest discover -s tests -v
(the first run trains the model, about 30 seconds).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
import data_processor as dp  # noqa: E402
import model_manager  # noqa: E402
import recommendations as rec  # noqa: E402
from model_engine import fps_to_tier  # noqa: E402
from utils import DataValidationError, InvalidInputError  # noqa: E402


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = dp.load_dataset()
        cls.df, cls.report = dp.clean_dataset(cls.raw)

    def test_clean_data_untouched(self):
        self.assertEqual(len(self.df), 5000)
        self.assertEqual(self.report.duplicates_removed, 0)

    def test_resolution_megapixels(self):
        self.assertAlmostEqual(dp.resolution_to_megapixels("1080p"), 2.0736, places=3)
        self.assertTrue(np.isnan(dp.resolution_to_megapixels("banana")))

    def test_bottleneck_formula_matches_dataset(self):
        err = (self.df["Score_Gap_Pct"] - self.df["Bottleneck_Pct"]).abs().max()
        self.assertLess(err, 0.01)

    def test_corrupted_rows_are_repaired_or_removed(self):
        bad = self.raw.astype(object).copy()
        bad.loc[0, "GPU"] = "  rtx 3080 "
        bad.loc[1, "Resolution"] = "1080P"
        bad.loc[2, "GPU_Score"] = np.nan
        bad.loc[3, "FPS"] = np.nan
        bad.loc[4, "FPS"] = -5
        bad.loc[5, "RAM_GB"] = "N/A"
        bad.loc[6, "Resolution"] = "8k??"
        bad = pd.concat([bad, bad.iloc[[10]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            bad.to_csv(path, index=False)
            cleaned, rep = dp.clean_dataset(dp.load_dataset(path))
        self.assertEqual(rep.duplicates_removed, 1)
        self.assertEqual(rep.missing_target_removed, 1)
        self.assertEqual(rep.invalid_rows_removed, 2)           # negative FPS and unknown resolution
        self.assertEqual(rep.hardware_scores_repaired, 1)
        self.assertEqual(set(cleaned["Resolution"]), {"720p", "1080p", "1440p"})
        self.assertEqual(cleaned["GPU"].nunique(), 6)

    def test_bad_files_raise_friendly_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing, empty, other = Path(tmp) / "none.csv", Path(tmp) / "e.csv", Path(tmp) / "o.csv"
            empty.write_text("")
            other.write_text("a,b\n" + "1,2\n" * 200)
            for path in (missing, empty, other):
                with self.assertRaises(DataValidationError):
                    dp.load_dataset(path)

    def test_audit_detects_constant_will_run(self):
        audit = dp.audit_dataset(self.raw, self.df)
        self.assertIn("Will_Run", audit["constant_columns"])
        self.assertFalse(audit["playability_ml_supported"])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = model_manager.load_or_train()
        cls.b = cls.predictor.b
        cls.df, _ = dp.load_clean_dataset()
        cls.cat = cls.predictor.catalog

    def config(self, **kw):
        base = dict(cpu="i7-8700K", gpu="RX 6600", ram_gb=16, vram_gb=8, game="CS2", resolution="1080p")
        base.update(kw)
        return self.predictor.validate(**base)

    def test_no_leakage_columns_in_model(self):
        used = set(self.predictor.features)
        for forbidden in ("FPS", "Quality", "Will_Run", "Load_Time_Sec", "Bottleneck_Pct"):
            self.assertNotIn(forbidden, used)

    def test_model_beats_baseline_on_test_set(self):
        self.assertGreater(self.b["test_metrics"]["R2"], 0.6)
        base = self.b["comparison"].query("complexity < 0").iloc[0]
        self.assertLess(self.b["test_metrics"]["MAE"], base["Test MAE"])

    def test_interval_coverage_is_calibrated(self):
        self.assertAlmostEqual(self.b["interval"]["coverage_test"], cfg.INTERVAL_COVERAGE, delta=0.06)

    def test_stronger_gpu_never_predicts_lower_fps(self):
        preds = [self.predictor.predict(self.config(gpu=g)).fps for g in self.cat["gpu"]]   # sorted by score
        self.assertEqual(preds, sorted(preds))

    def test_predictions_are_in_plausible_range(self):
        for gpu in self.cat["gpu"]:
            for cpu in self.cat["cpu"]:
                p = self.predictor.predict(self.config(gpu=gpu, cpu=cpu))
                self.assertFalse(p.outside_training_range)
                self.assertGreaterEqual(p.low, 0)
                self.assertLess(p.low, p.fps)
                self.assertLess(p.fps, p.high)

    def test_shapley_efficiency(self):
        e = self.predictor.explain(self.config(gpu="RTX 3070"))
        self.assertAlmostEqual(e.baseline + e.contributions["Contribution"].sum(), e.prediction, places=8)
        self.assertEqual(e.contributions.iloc[0]["Feature"], "GPU Score")

    def test_invalid_inputs_are_rejected(self):
        for kw in (dict(gpu="RTX 9090"), dict(cpu="Pentium"), dict(game="Fortnite"), dict(resolution="4k"),
                   dict(ram_gb=64), dict(vram_gb=7.5), dict(ram_gb="abc")):
            with self.assertRaises(InvalidInputError, msg=str(kw)):
                self.config(**kw)

    def test_quality_never_is_a_deployed_ml_model(self):
        self.assertFalse(self.b["quality_experiment"]["learnable"])


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = model_manager.load_or_train()
        cls.df, _ = dp.load_clean_dataset()

    def report(self, target=60, **kw):
        base = dict(cpu="i7-8700K", gpu="RX 6600", ram_gb=16, vram_gb=8, game="CS2", resolution="1080p")
        base.update(kw)
        return rec.build_report(self.predictor, self.predictor.validate(**base), target, self.df)

    def test_tier_boundaries(self):
        expected = {29.9: "Not Recommended", 30: "Low Performance", 45: "Playable", 60: "Very Good", 90: "Excellent"}
        for fps, name in expected.items():
            self.assertEqual(fps_to_tier(fps), name)
            self.assertEqual(rec.classify_playability(fps).name, name)

    def test_bottleneck_states(self):
        self.assertEqual(rec.analyze_bottleneck(26000, 26000).state, "Balanced")
        self.assertEqual(rec.analyze_bottleneck(28000, 16000).state, "GPU-limited")
        self.assertEqual(rec.analyze_bottleneck(12000, 26000).state, "CPU-limited")

    def test_quality_is_capped_by_vram(self):
        r = self.report(gpu="RTX 3080", vram_gb=4, resolution="1440p")
        self.assertEqual(r.quality.headroom_level, "Ultra")
        self.assertEqual(r.quality.limited_by, "VRAM")
        self.assertNotEqual(r.quality.level, "Ultra")

    def test_quality_follows_headroom(self):
        weak = self.report(gpu="GTX 1660", target=60).quality.level
        strong = self.report(gpu="RTX 3080", vram_gb=12, target=60).quality.level
        self.assertEqual(weak, "Low")
        self.assertEqual(strong, "Ultra")

    def test_recommendations_depend_on_the_system(self):
        weak = {x.title for x in self.report(gpu="GTX 1660").recommendations}
        strong = {x.title for x in self.report(gpu="RTX 3080", cpu="Ryzen 9 5900X", ram_gb=32, vram_gb=12).recommendations}
        self.assertIn("GPU upgrade has the biggest expected payoff", weak)
        self.assertIn("Top GPU in the dataset", strong)
        self.assertNotEqual(weak, strong)

    def test_gpu_upgrade_gain_is_model_driven_and_cpu_gain_is_negligible(self):
        r = self.report(gpu="GTX 1660")
        gpu_gain = r.upgrades.query("Component == 'GPU'")["Change (FPS)"].max()
        cpu_gain = r.upgrades.query("Component == 'CPU'")["Change (FPS)"].abs().max()
        self.assertGreater(gpu_gain, 20)
        self.assertLess(cpu_gain, 1.5)

    def test_confidence_is_a_probability(self):
        for gpu in self.predictor.catalog["gpu"]:
            c = self.report(gpu=gpu).confidence
            self.assertGreaterEqual(c, 0.5 - 1e-9)
            self.assertLessEqual(c, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

class AdvisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = model_manager.load_or_train()
        cls.df, _ = dp.load_clean_dataset()
        cat = cls.predictor.catalog
        cfg_obj = cls.predictor.validate(list(cat["cpu"])[2], list(cat["gpu"])[2], 16, 8,
                                         cat["games"][0], cat["resolutions"][0], False)
        cls.report = rec.build_report(cls.predictor, cfg_obj, 60, cls.df)

    def test_goal_target_selection_and_validation(self):
        self.assertEqual(rec.resolve_performance_goal("Balanced gameplay"), (60, "Balanced gameplay"))
        self.assertEqual(rec.resolve_performance_goal("Smoother performance"), (90, "Smoother performance"))
        self.assertEqual(rec.resolve_performance_goal("Higher visual quality"), (60, "Higher visual quality"))
        self.assertEqual(rec.resolve_performance_goal("Custom FPS target", 144), (144, "Custom FPS target"))
        for bad in (None, 0, 361, 59.5, "abc"):
            with self.assertRaises(ValueError):
                rec.resolve_performance_goal("Custom FPS target", bad)
        with self.assertRaises(ValueError):
            rec.resolve_performance_goal("Unsupported goal")

    def test_goal_summary_is_automatic_and_goal_specific(self):
        summary, notes = rec.build_advisor_summary(self.report, "Smoother performance")
        self.assertIn("Smoother performance", summary)
        self.assertTrue(notes)
        self.assertEqual(self.report.advisor_goal, "Smoother performance")

    def test_uncertainty_comes_from_existing_model_prediction(self):
        p = self.report.prediction
        self.assertLessEqual(p.low, p.fps)
        self.assertGreaterEqual(p.high, p.fps)
        self.assertGreater(p.coverage, 0)
        summary, _ = rec.build_advisor_summary(self.report, "Balanced gameplay")
        self.assertIn("interval", summary.lower())

    def test_report_export_text_contains_advisor(self):
        rec.build_advisor_summary(self.report, "Higher visual quality")
        text = rec.report_to_text(self.report)
        self.assertIn("Advisor goal", text)
        self.assertIn("Higher visual quality", text)

    def test_upgrade_estimates_are_computed_by_existing_predictor(self):
        upgrades = self.report.upgrades
        if len(upgrades):
            self.assertIn("Predicted FPS", upgrades.columns)
            self.assertIn("Change (FPS)", upgrades.columns)
            self.assertTrue(np.isfinite(upgrades["Predicted FPS"]).all())

