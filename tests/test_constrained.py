"""Synthetic checks for the hard observed-miss constraint and test-set isolation."""

import contextlib
import hashlib
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

from scripts.common import FEATURES, load_tickets, make_features
from scripts.fault_features import FaultInteractions
from scripts.predict import predict_csv
from scripts.report import render_saved_report
from scripts.train import build_models
from scripts.train_constrained import (
    adoption_gate, constrained_counts, daily_folds, evaluate_frozen, select_constrained, strict_threshold, train_and_freeze,
)
from tests.test_recall import rows


class ConstrainedTests(unittest.TestCase):
    def test_maximum_threshold_preserves_fault_boundary_without_decimal_rounding(self):
        p = np.array([0.263, 0.26476911889907523, 0.92])
        threshold = strict_threshold([0, 1, 1], p)
        self.assertEqual(threshold, np.nextafter(p[1], -np.inf))
        self.assertEqual(constrained_counts([0, 1, 1], p, threshold)["true_negatives"], 1)
        self.assertEqual(constrained_counts([0, 1, 1], p, threshold)["false_negatives"], 0)
        self.assertEqual(strict_threshold([1, 0], [0, 0.1]), 0)
        # Identical scores cannot be separated; ties must remain predicted faults.
        threshold = strict_threshold([0, 1], [0.3, 0.3])
        self.assertEqual(constrained_counts([0, 1], [0.3, 0.3], threshold)["true_negatives"], 0)

    def test_missing_fault_labels_and_invalid_scores_do_not_calibrate(self):
        for labels, probabilities in [([0], [0.1]), ([1], [np.nan]), ([1], [1.1]), ([1, 0], [0.1]), ([2], [0.1])]:
            with self.subTest(labels=labels, probabilities=probabilities), self.assertRaises(ValueError):
                strict_threshold(labels, probabilities)

    def test_avoid_any_miss_before_maximizing_true_negatives(self):
        candidates = [
            {"model": "unsafe", "false_negatives": 1, "true_negatives": 90, "days_with_true_negatives": 3, "brier_score": 0.01},
            {"model": "always-fault", "false_negatives": 0, "true_negatives": 0, "days_with_true_negatives": 0, "brier_score": 0.01},
            {"model": "useful", "false_negatives": 0, "true_negatives": 3, "days_with_true_negatives": 1, "brier_score": 0.2},
        ]
        self.assertEqual(select_constrained(candidates)["model"], "useful")

    def test_any_miss_or_no_identified_false_alarm_blocks_adoption(self):
        for fn, tn, expected in [(1, 10, "rejected"), (0, 0, "rejected"), (0, 2, "awaiting_user_acceptance")]:
            result = adoption_gate({"faults": 100, "model_sha256": "synthetic",
                                    "candidate": {"false_negatives": fn, "true_negatives": tn}})
            self.assertEqual(result["status"], expected)
            self.assertIsNone(result["user_accepted"])
            self.assertFalse(result["default_model_replaced"])

    def test_features_ignore_labels_ids_and_absolute_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.csv"
            rows(1, "a").to_csv(source, index=False)
            frame, _ = load_tickets(source)
            transformer = FaultInteractions().fit(frame, frame.fault_confirmed)
            original = transformer.transform(frame)
            changed = frame.assign(fault_confirmed=1 - frame.fault_confirmed,
                                   ticket_id="changed", ticket_time="2099-01-01")
            pd.testing.assert_frame_equal(original, transformer.transform(changed))
            self.assertFalse({"fault_confirmed", "ticket_id", "ticket_time", "business_day"} & set(original.columns))
            self.assertTrue(np.isfinite(original.select_dtypes(include="number").to_numpy()).all())
            self.assertTrue(set(FEATURES).issubset(original.columns))

    def test_daily_folds_do_not_split_dates_or_look_into_future(self):
        frame = pd.DataFrame({"ticket_id": [str(i) for i in range(6)],
                              "ticket_time": [f"2026-01-0{day}T0{hour}:00:00+08:00" for day in [1, 2, 3] for hour in [1, 2]],
                              "business_day": np.repeat(["2026-01-01", "2026-01-02", "2026-01-03"], 2),
                              "fault_confirmed": [0, 1, 0, 1, 0, 1]})
        folds = daily_folds(frame)
        self.assertEqual(len(folds), 2)
        for fit, check in folds:
            self.assertLess(frame.iloc[fit].business_day.max(), frame.iloc[check].business_day.min())
            self.assertEqual(frame.iloc[check].business_day.nunique(), 1)

    def test_invalid_fold_scores_cannot_silently_remove_faults_from_the_objective(self):
        def invalid_probability(estimator, features):
            probability = np.tile([0.2, 0.8], (len(features), 1))
            probability[1] = np.nan
            return probability

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "training.csv"
            pd.concat([rows(day, str(day)) for day in [1, 2, 3, 4]]).to_csv(source, index=False)
            with patch("scripts.train_constrained.build_constrained_candidates", return_value={"invalid": DummyClassifier()}), \
                    patch.object(DummyClassifier, "predict_proba", invalid_probability), \
                    self.assertRaisesRegex(ValueError, "Invalid fault scores"):
                train_and_freeze(source, root / "outputs", root / "artifacts")

    def test_freeze_precedes_test_read_and_inference_uses_identical_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, test = root / "train.csv", root / "test.csv"
            pd.concat([rows(day, str(day)) for day in [1, 2, 3, 4]]).to_csv(train, index=False)
            rows(5, "test", 12).to_csv(test, index=False)
            frame, _ = load_tickets(train)
            base = DummyClassifier(strategy="prior").fit(make_features(frame), frame.fault_confirmed)
            baseline = root / "baseline.joblib"
            joblib.dump({"pipeline": base, "review_threshold": 0.5}, baseline)
            outputs, artifacts = root / "outputs", root / "artifacts"
            seen = []

            def guarded_load(path):
                self.assertIn(path, [train, test])
                seen.append(path)
                if path == test:
                    frozen = json.loads((outputs / "selection.json").read_text(encoding="utf-8"))
                    self.assertEqual(hashlib.sha256((artifacts / "fault_model.joblib").read_bytes()).hexdigest(), frozen["model_sha256"])
                    self.assertEqual(frozen["estimator_fit_rows"], 144)
                    self.assertEqual(frozen["calibration_rows"], 48)
                return load_tickets(path)

            with patch("scripts.train_constrained.load_tickets", side_effect=guarded_load), \
                    patch("scripts.train_constrained.build_constrained_candidates", return_value={"synthetic": build_models()["logistic_regression"]}), \
                    contextlib.redirect_stdout(io.StringIO()):
                training, selection, model_path = train_and_freeze(train, outputs, artifacts)
                comparison = evaluate_frozen(training, selection, model_path, test, outputs, baseline)
            self.assertEqual(seen, [train, test])
            self.assertEqual(selection["calibration_results"]["false_negatives"], 0)
            self.assertLessEqual(selection["classification_threshold"], selection["selected_training_oof"]["classification_threshold"])
            self.assertFalse(comparison["future_zero_fn_guaranteed"])
            rendered = render_saved_report(outputs)
            inferred = predict_csv(test, model_path, root / "inference.csv")
            saved = pd.read_csv(outputs / "predictions.csv", float_precision="round_trip")
            self.assertEqual(inferred.predicted_fault_confirmed.tolist(), saved.predicted_fault_confirmed.tolist())
            self.assertEqual(rendered["missed_real_faults"], comparison["candidate"]["false_negatives"])


if __name__ == "__main__":
    unittest.main()
