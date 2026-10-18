"""Synthetic recall-calibration fixtures, not model performance evidence."""

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

from scripts.common import load_tickets, make_features
from scripts.train import build_models
from scripts.train_recall import select_recall_candidate, split_training_calibration, train_recall, zero_fn_threshold


def rows(day, prefix, n=48):
    times = pd.date_range(f"2026-01-{day:02d} 08:00", periods=n, freq="min", tz="Asia/Shanghai")
    return pd.DataFrame({
        "ticket_id": [f"synthetic-{prefix}-{i}" for i in range(n)],
        "ticket_time": [t.isoformat() for t in times],
        "last_telemetry_time": [(t - pd.Timedelta(minutes=5)).isoformat() for t in times],
        "region": "synthetic-region", "device_type": "synthetic-device",
        "reported_health_score": np.arange(n) % 10, "rated_power_kw": 50,
        "fault_confirmed": np.arange(n) % 2,
    })


class RecallTests(unittest.TestCase):
    def test_threshold_is_conservative_including_boundary_and_zero(self):
        threshold, minimum = zero_fn_threshold([0, 1, 1], [0.1, 0.4, 0.9], 0.05)
        self.assertEqual(threshold, 0.35)
        self.assertEqual(minimum, 0.4)
        self.assertTrue((np.array([0.4, 0.9]) >= threshold).all())
        self.assertEqual(zero_fn_threshold([0, 1], [0.1, 0.0])[0], 0)

    def test_invalid_or_fault_free_calibration_is_rejected(self):
        for labels, probability, margin in [([0, 0], [0.1, 0.2], 0.05),
                                            ([0, 1], [0.1, np.nan], 0.05),
                                            ([0, 1], [0.1, 0.2], -0.1)]:
            with self.subTest(labels=labels, probability=probability), self.assertRaises(ValueError):
                zero_fn_threshold(labels, probability, margin)

    def test_zero_misses_take_priority_over_fewer_total_errors(self):
        candidates = [
            {"model": "misses-one", "false_negatives": 1, "false_positives": 0, "brier_score": 0.01},
            {"model": "misses-none", "false_negatives": 0, "false_positives": 10, "brier_score": 0.1},
        ]
        self.assertEqual(select_recall_candidate(candidates)["model"], "misses-none")

    def test_last_training_day_is_separate_from_estimator_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "synthetic.csv"
            pd.concat([rows(1, "a"), rows(2, "b"), rows(3, "c")]).to_csv(csv, index=False)
            frame, _ = load_tickets(csv)
            fitting, calibration = split_training_calibration(frame)
            self.assertEqual(len(fitting), 96)
            self.assertEqual(len(calibration), 48)
            self.assertFalse(set(fitting.ticket_id) & set(calibration.ticket_id))
            self.assertLess(fitting.business_day.max(), calibration.business_day.min())

    def test_candidate_and_threshold_are_frozen_before_external_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, test = root / "train.csv", root / "test.csv"
            pd.concat([rows(1, "a"), rows(2, "b"), rows(3, "c")]).to_csv(train, index=False)
            rows(4, "test", 12).to_csv(test, index=False)
            frame, _ = load_tickets(train)
            base = DummyClassifier(strategy="prior").fit(make_features(frame), frame.fault_confirmed)
            baseline = root / "baseline.joblib"
            joblib.dump({"pipeline": base, "review_threshold": 0.6}, baseline)
            outputs, artifacts = root / "outputs", root / "artifacts"
            seen = []

            def guarded_load(path):
                self.assertIn(path, [train, test])
                seen.append(path)
                if path == test:
                    self.assertTrue((artifacts / "fault_model.joblib").exists())
                    frozen = json.loads((outputs / "selection.json").read_text(encoding="utf-8"))
                    self.assertEqual(frozen["calibration_rows"], 48)
                    self.assertEqual(frozen["estimator_fit_rows"], 96)
                return load_tickets(path)

            with patch("scripts.train_recall.load_tickets", side_effect=guarded_load), \
                    patch("scripts.train_recall.build_candidates", return_value={"synthetic": build_models()["logistic_regression"]}), \
                    contextlib.redirect_stdout(io.StringIO()):
                selection, comparison = train_recall(train, test, baseline, outputs, artifacts)
            self.assertEqual(seen, [train, test])
            self.assertLessEqual(selection["classification_threshold"], selection["selected_training_oof"]["classification_threshold"])
            self.assertEqual(selection["calibration_results"]["false_negatives"], 0)
            self.assertFalse(comparison["future_zero_fn_guaranteed"])


if __name__ == "__main__":
    unittest.main()
