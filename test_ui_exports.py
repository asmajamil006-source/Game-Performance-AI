"""Regression tests for theme CSS and downloadable prediction reports."""
from __future__ import annotations

import io
import zipfile
import unittest
import pandas as pd

import data_processor as dp
import recommendations as rec
import ui_components as ui
from model_manager import load_or_train
from report_exports import build_excel, build_word


class ThemeCssTests(unittest.TestCase):
    def test_all_theme_styles_are_injected(self):
        for theme in ("System", "Dark", "Light"):
            with self.subTest(theme=theme):
                css = ui.inject_css(theme)
                self.assertIn("<style>", css)
                self.assertIn("</style>", css)
                self.assertNotIn("{_theme_css", css)
        self.assertIn("prefers-color-scheme", ui.inject_css("System"))
        self.assertIn("#0B1120", ui.inject_css("Dark"))
        self.assertIn("#F6F8FB", ui.inject_css("Light"))
        for theme in ("System", "Dark", "Light"):
            css = ui.inject_css(theme)
            self.assertIn("--btn-primary-fg", css)
            self.assertIn("--btn-secondary-fg", css)
            self.assertIn("--btn-white-fg", css)
            self.assertIn(":focus-visible", css)
            self.assertIn(":disabled", css)
            self.assertIn("stDownloadButton", css)
            self.assertIn('[data-testid="stExpander"] > details > summary', css)
            self.assertIn("summary:hover button", css)
            self.assertIn("--surface", css)
            self.assertIn("min-height:184px", css)


class ReportExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = load_or_train()
        cls.data, _ = dp.load_clean_dataset()
        cat = cls.predictor.catalog
        system = cls.predictor.validate(
            list(cat["cpu"])[2], list(cat["gpu"])[2], 16, 8,
            cat["games"][0], cat["resolutions"][0], False,
        )
        cls.report = rec.build_report(cls.predictor, system, 60, cls.data)
        rec.build_advisor_summary(cls.report, "Balanced gameplay")

    def test_excel_export_is_valid_workbook(self):
        payload = build_excel(self.report)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIn("xl/workbook.xml", archive.namelist())
            self.assertIn("xl/worksheets/sheet1.xml", archive.namelist())

    def test_word_export_is_valid_document(self):
        payload = build_word(self.report)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIn("word/document.xml", archive.namelist())
            document = archive.read("word/document.xml").decode("utf-8")
            self.assertIn("AI auto-optimization advisor", document)
            self.assertIn("Balanced gameplay", document)

    def test_excel_export_contains_advisor_fields(self):
        payload = build_excel(self.report)
        with pd.ExcelFile(io.BytesIO(payload)) as workbook:
            summary = pd.read_excel(workbook, sheet_name="Prediction Summary")
        self.assertIn("Advisor goal", set(summary["Metric"]))
        self.assertIn("Advisor summary", set(summary["Metric"]))


if __name__ == "__main__":
    unittest.main()
