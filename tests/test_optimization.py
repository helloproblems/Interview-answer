"""Synthetic fixtures for selection boundaries, held-out isolation and thresholds."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from scripts.common import classification_metrics, load_tickets, make_features, predictions_frame
from scripts.evaluate import evaluate
from scripts.optimize import error_counts, optimize, select_candidate, validate_test
from scripts.predict import predict_csv
from scripts.train import build_models


def synthetic_rows(n, day, prefix):
    timestamps = pd.date_range(f"2026-01-{day:02d} 08:00", periods=n, freq="min", tz="Asia/Shanghai")
    return pd.DataFrame({
        "ticket_id": [f"synthetic-{prefix}-{i}" for i in range(n)],
        "ticket_time": [t.isoformat() for t in timestamps],
        "last_telemetry_time": [(t - pd.Timedelta(minutes=5)).isoformat() for t in timestamps],
        "region": "synthetic-region", "device_type": "synthetic-device",
        "reported_health_score": np.arange(n) % 20, "rated_power_kw": 50,
        "fault_confirmed": np.arange(n) % 2,
    })


class OptimizationTests(unittest.TestCase):
    def test_selection_does_not_hide_one_error_type_by_improving_the_other(self):
        reference = {"false_positives": 6, "false_negatives": 2}
        rows = [
            {"model": "unsafe-tradeoff", "false_positives": 0, "false_negatives": 3, "errors": 3,
             "brier_score": 0.1, "classification_threshold": 0.5},
            {"model": "both-improved", "false_positives": 4, "false_negatives": 1, "errors": 5,
             "brier_score": 0.2, "classification_threshold": 0.4},
        ]
        self.assertEqual(select_candidate(rows, reference)["model"], "both-improved")

    def test_always_fault_is_not_zero_error(self):
        counts = error_counts([0, 1, 1], [1, 1, 1])
        self.assertEqual(counts["false_negatives"], 0)
        self.assertEqual(counts["false_positives"], 1)
        self.assertEqual(counts["errors"], 1)

    def test_test_boundaries_reject_overlap_and_backwards_time(self):
        train = synthetic_rows(10, 1, "train")
        later = synthetic_rows(10, 2, "test")
        validate_test(train, later)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_test(train, train.copy())
        with self.assertRaisesRegex(ValueError, "after training"):
            validate_test(later, train)

    def test_threshold_changes_decisions_but_not_probabilities(self):
        frame = synthetic_rows(3, 1, "threshold")
        frame["telemetry_age_minutes"] = 5
        result = predictions_frame(frame, [0.39, 0.4, 0.5], 0.6, 0.4)
        self.assertEqual(result.predicted_fault_confirmed.tolist(), [0, 1, 1])
        self.assertEqual(result.fault_probability.tolist(), [0.39, 0.4, 0.5])
        metrics = classification_metrics([0, 1, 1], [0.39, 0.4, 0.5], 0.4)
        self.assertEqual(metrics["accuracy"], 1)
        for threshold in [np.nan, -0.1, 1.1]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                predictions_frame(frame, [0.39, 0.4, 0.5], 0.6, threshold)

    def test_saved_threshold_is_shared_by_prediction_evaluation_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "synthetic.csv"
            synthetic_rows(10, 1, "threshold").to_csv(csv, index=False)
            frame, _ = load_tickets(csv)
            pipeline = DummyClassifier(strategy="prior").fit(make_features(frame), frame.fault_confirmed)
            model = root / "synthetic.joblib"
            joblib.dump({"pipeline": pipeline, "review_threshold": 0.7, "classification_threshold": 0.6,
                         "model_name": "synthetic", "review_reference_floor": 0.95}, model)
            predictions = predict_csv(csv, model, root / "inference.csv")
            evaluate(csv, model, root / "evaluation")
            evaluation = pd.read_csv(root / "evaluation/predictions.csv")
            self.assertEqual(predictions.predicted_fault_confirmed.tolist(), [0] * 10)
            self.assertEqual(evaluation.predicted_fault_confirmed.tolist(), [0] * 10)
            page = (root / "evaluation/report.html").read_text(encoding="utf-8")
            self.assertIn("分类 0.60 / 复核 0.70", page)
            summary = json.loads((root / "evaluation/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["classification_threshold"], 0.6)

    def test_candidate_is_frozen_before_test_is_opened_and_validation_is_never_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, test = root / "train.csv", root / "test.csv"
            synthetic_rows(48, 1, "train").to_csv(train, index=False)
            synthetic_rows(10, 2, "test").to_csv(test, index=False)
            outputs, artifacts = root / "outputs", root / "artifacts"
            seen = []

            def guarded_load(path):
                seen.append(path)
                if path == test:
                    self.assertTrue((outputs / "selection.json").exists())
                    self.assertTrue((artifacts / "fault_model.joblib").exists())
                self.assertIn(path, [train, test])
                return load_tickets(path)

            tiny_search = {"random_forest": build_models()["logistic_regression"]}
            with patch("scripts.optimize.load_tickets", side_effect=guarded_load), \
                    patch("scripts.optimize.build_candidates", return_value=tiny_search), \
                    contextlib.redirect_stdout(io.StringIO()):
                selection, comparison = optimize(train, test, outputs, artifacts)
            self.assertEqual(seen, [train, test])
            self.assertFalse(selection["test_used_for_selection"])
            self.assertFalse(comparison["final_validation_evaluated"])
            page = (outputs / "report.html").read_text(encoding="utf-8")
            self.assertIn("测试集评估 · 非最终验收", page)
            self.assertTrue((outputs / "comparison.md").exists())


if __name__ == "__main__":
    unittest.main()
