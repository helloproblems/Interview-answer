"""Feature ablation must remove columns without leaking labels or external evaluations."""

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
from sklearn.linear_model import LogisticRegression

from scripts.common import FEATURES, load_tickets, make_features
from scripts.predict import predict_csv
from scripts.select_features import (
    AGE, HEALTH, ALGORITHMS, FEATURE_SETS, build_ablation_candidates, build_feature_pipeline,
    experiment_markdown, select_feature_candidate, train_feature_selection,
)
from scripts.train_constrained import evaluate_frozen
from tests.test_recall import rows


class FeatureSelectionTests(unittest.TestCase):
    def test_excluded_columns_cannot_influence_serialized_predictions(self):
        x = pd.DataFrame({AGE: np.arange(20), HEALTH: np.arange(20) % 5,
                          "rated_power_kw": 50, "region": "synthetic", "device_type": "synthetic"})
        model = build_feature_pipeline([AGE, HEALTH], LogisticRegression()).fit(x, np.arange(20) % 2)
        changed = x.assign(rated_power_kw=99999, region="unseen_region", device_type="unseen_device",
                           fault_confirmed=1, ticket_id="unused", ticket_time="2099-01-01")
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "pipeline.joblib"
            joblib.dump(model, artifact)
            loaded = joblib.load(artifact)
            np.testing.assert_array_equal(loaded.predict_proba(x), loaded.predict_proba(changed))
            self.assertEqual(loaded.named_steps["preprocessing"].transform(x).shape[1], 2)

    def test_forbid_label_id_dates_duplicate_columns_and_empty_subsets(self):
        for columns in [[], ["fault_confirmed"], ["ticket_id"], ["ticket_time"], [AGE, AGE]]:
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                build_feature_pipeline(columns, LogisticRegression())

    def test_zero_misses_then_false_alarm_detection_then_simplicity(self):
        def candidate(name, fn, tn, count, brier):
            return {"model": name, "false_negatives": fn, "true_negatives": tn,
                    "days_with_true_negatives": 1, "feature_count": count, "brier_score": brier}
        candidates = [candidate("unsafe", 1, 100, 1, 0.01), candidate("complex", 0, 3, 5, 0.01),
                      candidate("simple", 0, 3, 2, 0.1), candidate("all_fault", 0, 0, 1, 0.01)]
        self.assertEqual(select_feature_candidate(candidates)["model"], "simple")

    def test_each_feature_set_uses_the_same_algorithm_hyperparameters(self):
        candidates, details = build_ablation_candidates()
        self.assertEqual(len(candidates), len(FEATURE_SETS) * len(ALGORITHMS))
        for name, model in candidates.items():
            algorithm = details[name]["algorithm"]
            reference = candidates[f"all_features__{algorithm}"]
            self.assertEqual(model.named_steps["classifier"].get_params(), reference.named_steps["classifier"].get_params())
            self.assertNotIn("interactions", model.named_steps)
            self.assertFalse(set(details[name]["selected_features"]) & set(details[name]["excluded_features"]))
            self.assertEqual(set(details[name]["selected_features"] + details[name]["excluded_features"]), set(FEATURES))

    def test_subset_and_matched_control_are_frozen_before_test_and_no_validation_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, test = root / "train.csv", root / "test.csv"
            pd.concat([rows(day, str(day)) for day in [1, 2, 3, 4]]).to_csv(train, index=False)
            rows(5, "test", 12).to_csv(test, index=False)
            frame, _ = load_tickets(train)
            baseline = root / "baseline.joblib"
            joblib.dump({"pipeline": DummyClassifier().fit(make_features(frame), frame.fault_confirmed),
                         "review_threshold": 0.5}, baseline)
            outputs, artifacts, seen = root / "outputs", root / "artifacts", []

            def guarded_load(path):
                self.assertIn(path, [train, test])
                seen.append(path)
                if path == test:
                    frozen = json.loads((outputs / "selection.json").read_text(encoding="utf-8"))
                    control = frozen["matched_full_features_control"]
                    self.assertEqual(hashlib.sha256(Path(control["path"]).read_bytes()).hexdigest(), control["model_sha256"])
                    self.assertEqual(hashlib.sha256((artifacts / "fault_model.joblib").read_bytes()).hexdigest(), frozen["model_sha256"])
                    self.assertEqual(control["estimator_fit_rows"], frozen["estimator_fit_rows"])
                    self.assertEqual(control["calibration_rows"], frozen["calibration_rows"])
                return load_tickets(path)

            with patch("scripts.select_features.FEATURE_SETS", {"all_features": FEATURES, "age_health": [AGE, HEALTH]}), \
                    patch("scripts.select_features.ALGORITHMS", ["logistic_regression"]), \
                    patch("scripts.train_constrained.load_tickets", side_effect=guarded_load), \
                    contextlib.redirect_stdout(io.StringIO()):
                training, selection, model_path = train_feature_selection(train, outputs, artifacts)
                comparison = evaluate_frozen(training, selection, model_path, test, outputs, baseline,
                                             experiment_renderer=experiment_markdown)
            self.assertEqual(seen, [train, test])
            self.assertIn("matched_full_features_control", comparison)
            self.assertEqual(joblib.load(model_path)["features"], selection["selected_features"])
            inferred = predict_csv(test, model_path, root / "inference.csv")
            saved = pd.read_csv(outputs / "predictions.csv", float_precision="round_trip")
            self.assertEqual(inferred.predicted_fault_confirmed.tolist(), saved.predicted_fault_confirmed.tolist())
            self.assertIn("同算法全部 5 列对照", (outputs / "experiment.md").read_text(encoding="utf-8"))
            control_path = Path(selection["matched_full_features_control"]["path"])
            control_path.write_bytes(control_path.read_bytes() + b"synthetic-tamper")
            with patch("scripts.train_constrained.load_tickets") as reader, \
                    self.assertRaisesRegex(ValueError, "control fingerprint"):
                evaluate_frozen(training, selection, model_path, test, outputs, baseline)
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
