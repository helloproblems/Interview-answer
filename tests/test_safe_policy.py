"""Safety-contract tests preserve abstentions instead of counting them as correct."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from scripts.common import classification_metrics, load_tickets, make_features, review_metrics
from scripts.evaluate import evaluate
from scripts.predict import predict_csv
from scripts.report import prepare_report, render_saved_report
from scripts.safe_policy import POLICY, decision_metrics, safe_predictions_frame


def input_frame(n=11):
    return pd.DataFrame({
        "ticket_id": [f"synthetic-safe-{i}" for i in range(n)],
        "ticket_time": "2026-01-01T08:00:00+08:00", "last_telemetry_time": "2026-01-01T00:00:00Z",
        "region": "synthetic-region", "device_type": "synthetic-device",
        "reported_health_score": 10, "rated_power_kw": 50, "telemetry_age_minutes": 0,
        "fault_confirmed": np.arange(n) % 2,
    })


class SafePolicyTests(unittest.TestCase):
    def test_every_probability_and_threshold_combination_excludes_no_fault_output(self):
        frame, probability = input_frame(), np.linspace(0, 1, 11)
        for classification, review in [(0, 0), (0.5, 0.6), (0.9, 0.1), (1, 1)]:
            with self.subTest(classification=classification, review=review):
                output = safe_predictions_frame(frame, probability, review, classification)
                self.assertTrue(output.predicted_fault_confirmed.dropna().eq(1).all())
                self.assertTrue(output.dispatch_action.eq("preserve_dispatch").all())
                self.assertEqual(output.predicted_fault_confirmed.isna().tolist(), (probability < max(review, classification)).tolist())
                np.testing.assert_array_equal(output.fault_probability, probability)

    def test_unresolved_faults_are_counted_and_raw_classifier_misses_remain_visible(self):
        output = safe_predictions_frame(input_frame(4), [0.1, 0.2, 0.7, 0.9], 0.6, 0.5)
        result = decision_metrics([1, 0, 0, 1], output, 0.5)
        self.assertEqual(result["real_faults_pending_review"], 1)
        self.assertEqual(result["false_alarms_pending_review"], 1)
        self.assertEqual(result["automatic_classification_coverage"], 0.5)
        self.assertIsNone(result["full_workflow_accuracy"])
        self.assertEqual(result["underlying_classifier"]["confusion_matrix_0_1"][1][0], 1)
        self.assertEqual(result["real_faults_pending_review"] + result["real_faults_auto_flagged"], result["faults"])

    def test_safe_artifact_drives_both_inference_and_evaluation_without_label_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.csv"
            input_frame(10).to_csv(source, index=False)
            frame, _ = load_tickets(source)
            model = DummyClassifier(strategy="prior").fit(make_features(frame), frame.fault_confirmed)
            artifact = root / "safe.joblib"
            joblib.dump({"pipeline": model, "model_name": "synthetic", "review_threshold": 0.6,
                         "classification_threshold": 0.5, "decision_policy": POLICY}, artifact)
            inferred = predict_csv(source, artifact, root / "inference.csv")
            metrics = evaluate(source, artifact, root / "evaluation")
            self.assertTrue(inferred.predicted_fault_confirmed.isna().all())
            self.assertEqual(metrics["decision_metrics"]["pending_reviews"], 10)
            self.assertEqual(metrics["decision_metrics"]["automatic_classification_coverage"], 0)
            inferred_again = input_frame(10).drop(columns="fault_confirmed")
            inferred_again.to_csv(source, index=False)
            actual = predict_csv(source, artifact, root / "no-labels.csv")
            pd.testing.assert_frame_equal(inferred, actual)
            render_saved_report(root / "evaluation")
            self.assertIn("待复核不计为预测正确", (root / "evaluation/report.html").read_text(encoding="utf-8"))

    def test_an_accidental_no_fault_verdict_is_rejected(self):
        output = safe_predictions_frame(input_frame(2), [0.1, 0.9], 0.6)
        output.loc[0, "predicted_fault_confirmed"] = 0
        with self.assertRaisesRegex(ValueError, "never output"):
            decision_metrics([1, 0], output, 0.5)

    def test_recall_report_does_not_call_all_positive_predictions_a_useful_zero_error_model(self):
        labels, probability = np.array([0, 1, 1]), np.array([0.7, 0.8, 0.9])
        predictions = pd.DataFrame({"ticket_id": ["s1", "s2", "s3"], "fault_confirmed": labels,
                                    "fault_probability": probability, "predicted_fault_confirmed": 1,
                                    "recommendation": "normal_dispatch"})
        metrics = {"classification": classification_metrics(labels, probability),
                   "review": review_metrics(labels, probability < 0.5), "review_threshold": 0.5,
                   "model": "synthetic", "optimization_goal": "zero_false_negatives",
                   "previous_model_error_counts": {"false_positives": 0, "false_negatives": 1}}
        summary, _ = prepare_report(metrics, predictions)
        self.assertIn("全部预测为故障", summary["interpretation"])
        self.assertFalse(summary["observed_zero_errors"])


if __name__ == "__main__":
    unittest.main()
