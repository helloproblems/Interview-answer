"""Batch inference; fault_confirmed is optional and never used for prediction."""

import argparse
from pathlib import Path

import joblib

from scripts.common import load_tickets, make_features, predictions_frame


def predict_csv(input_path, model_path, output_path):
    # joblib/pickle must only be loaded from our own trusted training output.
    bundle = joblib.load(model_path)
    frame, _ = load_tickets(input_path, require_labels=False)
    probability = bundle["pipeline"].predict_proba(make_features(frame))[:, 1]
    output = predictions_frame(frame, probability, bundle["review_threshold"], bundle.get("classification_threshold", 0.5))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", default=Path("artifacts/fault_model.joblib"), type=Path)
    parser.add_argument("--output", default=Path("outputs/predictions.csv"), type=Path)
    args = parser.parse_args()
    try:
        output = predict_csv(args.input, args.model, args.output)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f"Prediction aborted; preserve existing dispatch. {exc}\n")
    print(f"Predicted {len(output)} unique tickets -> {args.output}")


if __name__ == "__main__":
    main()
