"""Synthetic fixtures test behavior only; they are never business evidence."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.common import (
    FEATURES, choose_review_threshold, load_tickets, make_features,
    predictions_frame, review_metrics,
)
from scripts.predict import predict_csv
from scripts.train import build_models, split_by_time


def ticket(ticket_id="synthetic-1", **changes):
    row = {
        "ticket_id": ticket_id, "ticket_time": "2026-09-01T08:00:00+08:00",
        "last_telemetry_time": "2026-08-31T23:40:00Z", "region": "synthetic-region",
        "device_type": "synthetic-device", "reported_health_score": 0,
        "rated_power_kw": 0, "fault_confirmed": 1,
    }
    row.update(changes)
    return row


class DataAndPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="inspection-model-tests-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def load(self, rows, require_labels=True):
        path = self.root / "synthetic.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return load_tickets(path, require_labels=require_labels)

    def test_timezone_conversion_and_zero_values_are_preserved(self):
        frame, quality = self.load([ticket()])
        self.assertEqual(frame.telemetry_age_minutes.iloc[0], 20)
        self.assertEqual(frame.business_day.iloc[0], "2026-09-01")
        self.assertEqual(quality["zero_health_scores"], 1)
        self.assertEqual(quality["zero_rated_power"], 1)

    def test_identical_duplicate_collapses_before_training(self):
        frame, quality = self.load([ticket(), ticket()])
        self.assertEqual(len(frame), 1)
        self.assertEqual(quality["exact_duplicates_removed"], 1)

    def test_conflicting_duplicate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.load([ticket(), ticket(fault_confirmed=0)])

    def test_unknown_label_is_not_converted_to_false(self):
        frame, quality = self.load([ticket(fault_confirmed=None)])
        self.assertTrue(pd.isna(frame.fault_confirmed.iloc[0]))
        self.assertEqual(quality["confirmed_false_alarms"], 0)

    def test_invalid_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "0, 1"):
            self.load([ticket(fault_confirmed=2)])

    def test_future_or_naive_telemetry_is_rejected(self):
        for telemetry in ["2026-09-01T00:01:00Z", "2026-09-01T00:00:00"]:
            with self.subTest(telemetry=telemetry), self.assertRaises(ValueError):
                self.load([ticket(last_telemetry_time=telemetry)])

    def test_missing_required_feature_is_rejected(self):
        row = ticket()
        del row["rated_power_kw"]
        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.load([row])

    def test_nonfinite_or_negative_measurement_is_rejected(self):
        for value in [float("inf"), -1]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                self.load([ticket(rated_power_kw=value)])

    def test_feature_matrix_excludes_labels_ids_and_absolute_dates(self):
        frame, _ = self.load([ticket()])
        features = make_features(frame)
        self.assertEqual(list(features.columns), FEATURES)
        frame["fault_confirmed"] = 0
        frame["ticket_id"] = "changed-id"
        pd.testing.assert_frame_equal(features, make_features(frame))

    def test_unknown_outcome_has_cost_without_avoided_trip_credit(self):
        result = review_metrics([0, 1, np.nan], [True, True, True])
        self.assertEqual(result["potential_net_hours_at_100pct_success"], 84)
        self.assertEqual(result["unknown_outcomes_reviewed"], 1)
        self.assertAlmostEqual(result["break_even_review_success"], 0.3)

    def test_no_review_and_no_false_alarms_do_not_divide_by_zero(self):
        result = review_metrics([1, np.nan], [False, False])
        self.assertEqual(result["potential_net_hours_at_100pct_success"], 0)
        self.assertIsNone(result["break_even_review_success"])

    def test_classification_and_review_thresholds_are_distinct(self):
        frame, _ = self.load([ticket(), ticket("synthetic-2"), ticket("synthetic-3")])
        result = predictions_frame(frame, [0.49, 0.5, 0.65], 0.65)
        self.assertEqual(result.predicted_fault_confirmed.tolist(), [0, 1, 1])
        self.assertEqual(result.recommendation.tolist(), ["remote_review", "remote_review", "normal_dispatch"])

    def test_threshold_selection_respects_validation_risk_budget(self):
        threshold, _ = choose_review_threshold([0, 1, 1], [0.1, 0.4, 0.9], recall_floor=1.0)
        result = review_metrics([0, 1, 1], np.array([0.1, 0.4, 0.9]) < threshold)
        self.assertEqual(result["real_faults_reviewed"], 0)
        self.assertEqual(result["false_alarms_reviewed"], 1)

    def test_time_splits_are_disjoint_and_boundary_inclusive(self):
        rows = []
        for day in ["2026-09-04", "2026-09-05", "2026-09-06"]:
            for label in [0, 1]:
                rows.append(ticket(f"synthetic-{day}-{label}", ticket_time=f"{day}T08:00:00+08:00",
                                   last_telemetry_time=f"{day}T00:00:00Z", fault_confirmed=label))
        rows.append(ticket("synthetic-unreturned", fault_confirmed=None))
        frame, _ = self.load(rows)
        split = split_by_time(frame)
        self.assertEqual(split.split.tolist(), ["train", "train", "test", "test", "validation", "validation"])
        self.assertEqual(split.ticket_id.nunique(), len(split))
        self.assertEqual(len(split), 6)

    def test_serialized_pipeline_predicts_without_labels_and_unseen_category(self):
        rows = [ticket(f"synthetic-{i}", fault_confirmed=i % 2, reported_health_score=i) for i in range(12)]
        frame, _ = self.load(rows)
        model = build_models()["logistic_regression"].fit(make_features(frame), frame.fault_confirmed)
        model_path = self.root / "synthetic_model.joblib"
        joblib.dump({"pipeline": model, "review_threshold": 0.65}, model_path)
        no_label = frame.iloc[:2].copy().drop(columns="fault_confirmed")
        no_label["region"] = "previously-unseen-region"
        input_path = self.root / "unlabeled.csv"
        no_label.to_csv(input_path, index=False)
        output = predict_csv(input_path, model_path, self.root / "predictions.csv")
        self.assertEqual(len(output), 2)
        self.assertTrue(output.fault_probability.between(0, 1).all())
        no_label["fault_confirmed"] = [0, 1]
        no_label.to_csv(input_path, index=False)
        with_labels = predict_csv(input_path, model_path, self.root / "predictions.csv")
        np.testing.assert_allclose(output.fault_probability, with_labels.fault_probability)


if __name__ == "__main__":
    unittest.main()
