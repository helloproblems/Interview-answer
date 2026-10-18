"""Explicit user-invoked acceptance evaluation; never trains or changes the model."""

import argparse
import json
from pathlib import Path

import joblib

from scripts.common import classification_metrics, load_tickets, make_features, predictions_frame, review_metrics


def evaluate(input_path, model_path, output_dir):
    frame, quality = load_tickets(input_path)
    if frame.fault_confirmed.isna().any():
        raise ValueError("Acceptance data must have a known fault_confirmed value on every row")
    bundle = joblib.load(model_path)
    probability = bundle["pipeline"].predict_proba(make_features(frame))[:, 1]
    metrics = {"data_quality": quality, "classification": classification_metrics(frame.fault_confirmed, probability),
               "review": review_metrics(frame.fault_confirmed, probability < bundle["review_threshold"]),
               "review_threshold": bundle["review_threshold"], "model": bundle["model_name"]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = predictions_frame(frame, probability, bundle["review_threshold"])
    output["fault_confirmed"] = frame.fault_confirmed
    output.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", default=Path("artifacts/fault_model.joblib"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/user_acceptance"), type=Path)
    args = parser.parse_args()
    try:
        metrics = evaluate(args.input, args.model, args.output_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Evaluation aborted: {exc}\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
