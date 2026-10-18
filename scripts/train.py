"""Select within training data, evaluate on test data, reserve validation for the user."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.common import (
    CATEGORICAL_FEATURES, FEATURES, INPUT_COLUMNS, NUMERIC_FEATURES, FALSE_TRIP_HOURS, REVIEW_HOURS,
    choose_review_threshold, classification_metrics, load_tickets, make_features,
    predictions_frame, review_metrics,
)


def build_models():
    """Small fixed model set; transformations fit only on training rows."""
    estimators = {
        "prior_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=12,
            max_features=1.0, random_state=42, n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=7, min_samples_leaf=20,
            learning_rate=0.05, l2_regularization=5.0,
            early_stopping=False, random_state=42,
        ),
    }
    result = {}
    for name, estimator in estimators.items():
        preprocessing = ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ])
        result[name] = Pipeline([("preprocessing", preprocessing), ("classifier", estimator)])
    return result


def split_by_time(frame, test_start="2026-09-05", validation_start="2026-09-06"):
    """Only labeled rows are eligible; final validation belongs to the user."""
    if pd.Timestamp(test_start) >= pd.Timestamp(validation_start):
        raise ValueError("test_start must precede validation_start")
    output = frame.loc[frame.fault_confirmed.notna()].copy().reset_index(drop=True)
    output["split"] = np.select(
        [output.business_day < test_start, output.business_day < validation_start],
        ["train", "test"], default="validation",
    )
    for split in ["train", "test"]:
        labels = output.loc[output.split.eq(split), "fault_confirmed"]
        if set(labels.unique()) != {0, 1}:
            raise ValueError(f"{split} must contain both known classes")
    if not output.split.eq("validation").any():
        raise ValueError("Final user validation set must be nonempty")
    return output


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(input_path, output_dir, artifact_dir, dataset_dir, test_start, validation_start, recall_floor):
    if not 0 < recall_floor <= 1:
        raise ValueError("recall_floor must be in (0, 1]")
    raw, quality = load_tickets(input_path)
    frame = split_by_time(raw, test_start, validation_start)
    output_dir, artifact_dir, dataset_dir = map(Path, [output_dir, artifact_dir, dataset_dir])
    for folder in [output_dir, artifact_dir, dataset_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": quality["input_sha256"], "eligible_rows": len(frame), "sets": {}}
    for name in ["train", "test", "validation"]:
        part = frame[frame.split.eq(name)]
        part[INPUT_COLUMNS + ["fault_confirmed"]].to_csv(dataset_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        manifest["sets"][name] = {"rows": len(part), "first_day": part.business_day.min(),
                                 "last_day": part.business_day.max(),
                                 "purpose": "user_acceptance_only" if name == "validation" else name}
    write_json(dataset_dir / "manifest.json", manifest)
    training = frame[frame.split.eq("train")].reset_index(drop=True)
    testing = frame[frame.split.eq("test")].reset_index(drop=True)
    x_train, y_train = make_features(training), training.fault_confirmed.astype(int)
    models = build_models()
    cv_rows, oof_probabilities, cv_summary = [], {}, {}
    # All selection occurs inside the training dataset; neither external set is fitted.
    for name, prototype in models.items():
        oof = np.full(len(training), np.nan)
        scores = []
        for fold, (fit_indices, check_indices) in enumerate(TimeSeriesSplit(n_splits=3).split(x_train), start=1):
            candidate = clone(prototype)
            candidate.fit(x_train.iloc[fit_indices], y_train.iloc[fit_indices])
            probability = candidate.predict_proba(x_train.iloc[check_indices])[:, 1]
            oof[check_indices] = probability
            score = classification_metrics(y_train.iloc[check_indices], probability)
            scores.append(score["brier_score"])
            cv_rows.append({"model": name, "fold": fold, "fit_rows": len(fit_indices),
                            "check_rows": len(check_indices), **score})
        cv_summary[name] = {"mean_brier_score": float(np.mean(scores)), "folds": len(scores)}
        oof_probabilities[name] = oof
    selected_name = min(cv_summary, key=lambda name: (cv_summary[name]["mean_brier_score"], name))
    covered = np.isfinite(oof_probabilities[selected_name])
    review_threshold, threshold_rows = choose_review_threshold(
        y_train[covered], oof_probabilities[selected_name][covered], recall_floor,
    )
    selected = clone(models[selected_name]).fit(x_train, y_train)
    baseline = clone(models["prior_baseline"]).fit(x_train, y_train)
    test_probability = selected.predict_proba(make_features(testing))[:, 1]
    test_metrics = classification_metrics(testing.fault_confirmed, test_probability)
    baseline_metrics = classification_metrics(
        testing.fault_confirmed, baseline.predict_proba(make_features(testing))[:, 1],
    )
    policies = {
        "selected_model": test_probability < review_threshold,
        "telemetry_gt20": testing.telemetry_age_minutes > 20,
        "review_all": np.ones(len(testing), dtype=bool),
        "review_none": np.zeros(len(testing), dtype=bool),
    }
    policy_rows = [{"policy": name, **review_metrics(testing.fault_confirmed, mask)} for name, mask in policies.items()]
    metrics = {
        "data_quality": quality, "eligible_labeled_rows": len(frame), "dataset_manifest": manifest,
        "selection": {"model": selected_name, "criterion": "lowest mean Brier score in 3 expanding training-only folds",
                      "classification_threshold": 0.5, "review_threshold": review_threshold,
                      "training_oof_rows": int(covered.sum()), "training_oof_fault_retention_floor": recall_floor,
                      "test_start": test_start, "validation_start": validation_start,
                      "random_seed": 42, "final_validation_evaluated": False},
        "training_cross_validation": cv_summary, "test_selected_model": test_metrics,
        "test_prior_baseline": baseline_metrics, "test_review_policy": policy_rows[0],
        "cost_assumptions": {"false_trip_hours": FALSE_TRIP_HOURS, "review_hours": REVIEW_HOURS,
                             "review_success_observed": False, "fault_delay_cost_observed": False},
        "limitations": [
            "Only labeled unique tickets are eligible; missing labels are not converted to zero.",
            "The final validation file is exported for the user and never scored in this training workflow.",
            "An earlier project version inspected the final period; this is not a never-seen blind holdout.",
            "Only one week of data, missing device identity and fault severity; future accuracy is not guaranteed.",
            "Model probabilities are not independently calibrated; recommendations do not cancel dispatch.",
        ],
    }
    bundle = {"pipeline": selected, "review_threshold": review_threshold, "features": FEATURES,
              "model_name": selected_name, "training_input_sha256": quality["input_sha256"],
              "sklearn_version": sklearn.__version__, "test_start": test_start,
              "validation_start": validation_start, "selection_scope": "training_only_timeseries_cv",
              "mode": "advisory_only"}
    joblib.dump(bundle, artifact_dir / "fault_model.joblib", compress=3)
    write_json(artifact_dir / "model_metadata.json", {k: v for k, v in bundle.items() if k != "pipeline"})
    write_json(output_dir / "metrics.json", metrics)
    pd.DataFrame(cv_rows).to_csv(output_dir / "training_cv.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cv_summary).T.to_csv(output_dir / "model_comparison.csv", index_label="model", encoding="utf-8-sig")
    pd.DataFrame(threshold_rows).to_csv(output_dir / "training_thresholds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(policy_rows).to_csv(output_dir / "policy_comparison.csv", index=False, encoding="utf-8-sig")
    predictions = predictions_frame(testing, test_probability, review_threshold)
    predictions["fault_confirmed"] = testing.fault_confirmed
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    testing[INPUT_COLUMNS].head(5).to_csv(output_dir / "predict_input_example.csv", index=False, encoding="utf-8-sig")
    sensitivity = [{"policy": row["policy"], "assumed_review_success": success,
                    "potential_net_hours": row["false_alarms_reviewed"] * FALSE_TRIP_HOURS * success - row["review_hours"]}
                   for row in policy_rows for success in [0, 0.25, 0.5, 0.75, 1.0]]
    pd.DataFrame(sensitivity).to_csv(output_dir / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    matrix = np.array(test_metrics["confusion_matrix_0_1"])
    fig, axis = plt.subplots(figsize=(5, 4), layout="constrained")
    axis.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=20,
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    axis.set_xticks([0, 1], ["0: no fault", "1: fault"])
    axis.set_yticks([0, 1], ["0: no fault", "1: fault"])
    axis.set_xlabel("Predicted at p(fault) >= 0.5")
    axis.set_ylabel("Actual")
    axis.set_title("Test set only - user validation not evaluated")
    fig.savefig(output_dir / "evaluation.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"model": selected_name, "review_threshold": review_threshold,
                      "cv": cv_summary, "test": test_metrics, "review": policy_rows[0],
                      "user_validation_rows": manifest["sets"]["validation"]["rows"]}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=Path("inspection_tickets.csv"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--artifact-dir", default=Path("artifacts"), type=Path)
    parser.add_argument("--dataset-dir", default=Path("datasets"), type=Path)
    parser.add_argument("--test-start", default="2026-09-05")
    parser.add_argument("--validation-start", default="2026-09-06")
    parser.add_argument("--recall-floor", default=0.95, type=float)
    args = parser.parse_args()
    try:
        train(args.input, args.output_dir, args.artifact_dir, args.dataset_dir,
              args.test_start, args.validation_start, args.recall_floor)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Training aborted: {exc}\n")


if __name__ == "__main__":
    main()
