"""Synthetic report fixtures; never evidence for model/business performance."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import classification_metrics, review_metrics
from scripts.report import console_summary, prepare_report, render_saved_report


def fixture():
    labels = np.array([1] * 8 + [0, 0])
    probability = np.array([0.1, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.8])
    predictions = pd.DataFrame({
        "ticket_id": [f"synthetic-{i}" for i in range(10)],
        "region": ['<script>alert("unsafe")</script>'] * 10,
        "device_type": "synthetic-device", "fault_confirmed": labels,
        "fault_probability": probability, "predicted_fault_confirmed": (probability >= 0.5).astype(int),
        "recommendation": np.where(probability < 0.6, "remote_review", "normal_dispatch"),
    })
    metrics = {"classification": classification_metrics(labels, probability),
               "review": review_metrics(labels, probability < 0.6), "review_threshold": 0.6,
               "model": "synthetic-model", "data_quality": {"input_sha256": "synthetic-hash",
                   "first_business_day": "2026-01-01", "last_business_day": "2026-01-02"}}
    return metrics, predictions


class ReportTests(unittest.TestCase):
    def test_high_accuracy_is_not_automatically_called_effective(self):
        metrics, predictions = fixture()
        summary, details = prepare_report(metrics, predictions)
        self.assertAlmostEqual(summary["baseline_accuracy"], 0.8)
        self.assertAlmostEqual(summary["accuracy_difference_percentage_points"], -10)
        self.assertEqual(summary["missed_real_faults"], 2)
        self.assertEqual(summary["identified_false_alarms"], 1)
        self.assertFalse(summary["meets_review_reference_floor"])
        self.assertIsNone(summary["user_accepted"])
        self.assertEqual(details.classification_result.iloc[:2].tolist(), ["真故障漏判"] * 2)
        console = console_summary(summary, Path("synthetic-output"))
        self.assertIn("全判故障基线：80.00%", console)
        self.assertIn("漏判：2 条", console)
        self.assertIn("尚未代表用户验收通过", console)

    def test_mixed_metrics_and_prediction_runs_are_rejected(self):
        metrics, predictions = fixture()
        metrics["classification"]["accuracy"] = 0.99
        with self.assertRaisesRegex(ValueError, "disagree"):
            prepare_report(metrics, predictions)

    def test_inconsistent_fixed_threshold_predictions_are_rejected(self):
        metrics, predictions = fixture()
        predictions.loc[0, "predicted_fault_confirmed"] = 1
        with self.assertRaisesRegex(ValueError, "fixed 0.5"):
            prepare_report(metrics, predictions)

    def test_render_escapes_input_and_preserves_original_evidence(self):
        metrics, predictions = fixture()
        with tempfile.TemporaryDirectory(prefix="inspection-report-test-") as directory:
            output = Path(directory)
            (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            predictions.to_csv(output / "predictions.csv", index=False)
            originals = {name: (output / name).read_bytes() for name in ["metrics.json", "predictions.csv"]}
            render_saved_report(output)
            page = (output / "report.html").read_text(encoding="utf-8")
            self.assertNotIn('<script>alert("unsafe")</script>', page)
            self.assertIn("&lt;script&gt;", page)
            for name, content in originals.items():
                self.assertEqual((output / name).read_bytes(), content)
            self.assertTrue((output / "summary.md").exists())
            self.assertEqual(len(pd.read_csv(output / "validation_details.csv")), 10)


if __name__ == "__main__":
    unittest.main()
