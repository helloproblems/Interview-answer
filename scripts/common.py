"""Input validation, feature engineering, and business metrics shared by both CLIs."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score,
)

NUMERIC_FEATURES = ["telemetry_age_minutes", "reported_health_score", "rated_power_kw"]
CATEGORICAL_FEATURES = ["region", "device_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
INPUT_COLUMNS = [
    "ticket_id", "ticket_time", "last_telemetry_time", "region", "device_type",
    "reported_health_score", "rated_power_kw",
]
FALSE_TRIP_HOURS = 120
REVIEW_HOURS = 12


def load_tickets(path: str | Path, require_labels: bool = True):
    """Reject ambiguous inputs; collapse identical records before any time split."""
    path = Path(path)
    raw = pd.read_csv(path, dtype={"ticket_id": "string"}, encoding="utf-8-sig")
    required = INPUT_COLUMNS + (["fault_confirmed"] if require_labels else [])
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if raw.empty:
        raise ValueError("Input CSV is empty")
    if raw.ticket_id.isna().any() or raw.ticket_id.str.strip().eq("").any():
        raise ValueError("ticket_id must be nonempty")
    # Inference ignores outcome labels, including when checking duplicate features.
    frame = raw[required].drop_duplicates().copy()
    if frame.ticket_id.duplicated().any():
        raise ValueError("Conflicting records share ticket_id; resolve before continuing")
    frame = frame.reset_index(drop=True)
    if not require_labels:
        frame["fault_confirmed"] = np.nan
    frame["fault_confirmed"] = pd.to_numeric(frame.fault_confirmed, errors="raise")
    if not frame.fault_confirmed.dropna().isin([0, 1]).all():
        raise ValueError("fault_confirmed must be 0, 1, or empty")
    for column in ["reported_health_score", "rated_power_kw"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError(f"{column} must be finite and nonnegative; zero is retained")
    for column in CATEGORICAL_FEATURES:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be nonempty")
    parsed = {}
    for column in ["ticket_time", "last_telemetry_time"]:
        aware = frame[column].astype("string").str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", na=False)
        if not aware.all():
            raise ValueError(f"{column} must contain explicit timezone offsets")
        parsed[column] = pd.to_datetime(frame[column], format="ISO8601", utc=True, errors="raise")
    frame["telemetry_age_minutes"] = (
        parsed["ticket_time"] - parsed["last_telemetry_time"]
    ).dt.total_seconds() / 60
    if (frame.telemetry_age_minutes < 0).any():
        raise ValueError("Telemetry is in the future; do not infer on invalid timestamps")
    frame["business_day"] = parsed["ticket_time"].dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["ticket_time", "ticket_id"]).reset_index(drop=True)
    quality = {
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "exact_duplicates_removed": len(raw) - len(frame),
        "unique_tickets": len(frame), "labeled_tickets": int(frame.fault_confirmed.notna().sum()),
        "confirmed_faults": int(frame.fault_confirmed.eq(1).sum()),
        "confirmed_false_alarms": int(frame.fault_confirmed.eq(0).sum()),
        "unreturned_labels": int(frame.fault_confirmed.isna().sum()),
        "zero_health_scores": int(frame.reported_health_score.eq(0).sum()),
        "zero_rated_power": int(frame.rated_power_kw.eq(0).sum()),
        "first_business_day": frame.business_day.min(), "last_business_day": frame.business_day.max(),
    }
    return frame, quality


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Never pass ticket identifiers, dates, or outcome labels to the estimator."""
    return frame.loc[:, FEATURES].copy()


def wilson_interval(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def classification_metrics(y, probability):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": len(y), "faults": int(y.sum()), "false_alarms": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y, prediction)),
        "fault_precision": float(precision_score(y, prediction, zero_division=0)),
        "fault_recall": float(recall_score(y, prediction, zero_division=0)),
        "fault_recall_95pct_wilson": wilson_interval(int(tp), int(tp + fn)),
        "false_alarm_recall": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "false_alarm_average_precision": float(average_precision_score(1 - y, 1 - probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "confusion_matrix_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def review_metrics(y, review_mask):
    """Unknown outcomes incur review cost but receive no avoided-trip credit."""
    labels = np.asarray(y, dtype=float)
    mask = np.asarray(review_mask, dtype=bool)
    if len(labels) != len(mask):
        raise ValueError("Label and review-mask lengths differ")
    reviewed = int(mask.sum())
    false_reviewed = int(((labels == 0) & mask).sum())
    faults_reviewed = int(((labels == 1) & mask).sum())
    unknown_reviewed = int((np.isnan(labels) & mask).sum())
    total_false = int((labels == 0).sum())
    total_faults = int((labels == 1).sum())
    return {
        "tickets": len(labels), "reviews": reviewed,
        "review_share": reviewed / len(labels) if len(labels) else 0.0,
        "false_alarms_reviewed": false_reviewed, "real_faults_reviewed": faults_reviewed,
        "unknown_outcomes_reviewed": unknown_reviewed,
        "false_alarm_capture": false_reviewed / total_false if total_false else 0.0,
        "faults_outside_review_rate": 1 - faults_reviewed / total_faults if total_faults else None,
        "faults_outside_review_95pct_wilson": wilson_interval(total_faults - faults_reviewed, total_faults),
        "review_hours": reviewed * REVIEW_HOURS,
        "potential_net_hours_at_100pct_success": false_reviewed * FALSE_TRIP_HOURS - reviewed * REVIEW_HOURS,
        "break_even_review_success": reviewed * REVIEW_HOURS / (false_reviewed * FALSE_TRIP_HOURS) if false_reviewed else None,
    }


def choose_review_threshold(y, probability, recall_floor=0.95):
    """Choose on training out-of-fold predictions; the risk budget is advisory."""
    rows = []
    for threshold in np.round(np.arange(0, 1.0, 0.05), 2):
        row = review_metrics(y, np.asarray(probability) < threshold)
        row["threshold"] = float(threshold)
        row["meets_recall_floor"] = row["faults_outside_review_rate"] is not None and row["faults_outside_review_rate"] >= recall_floor
        rows.append(row)
    eligible = [row for row in rows if row["meets_recall_floor"]]
    if not eligible:
        raise ValueError("Threshold selection requires labeled real faults")
    best = max(eligible, key=lambda row: (row["potential_net_hours_at_100pct_success"], -row["reviews"], -row["threshold"]))
    return best["threshold"], rows


def predictions_frame(frame, probability, review_threshold):
    probability = np.asarray(probability, dtype=float)
    if not math.isfinite(review_threshold) or not 0 <= review_threshold <= 1:
        raise ValueError("Review threshold must be within [0, 1]")
    if len(frame) != len(probability) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid model probabilities")
    output = frame[INPUT_COLUMNS + ["telemetry_age_minutes"]].copy()
    output["fault_probability"] = probability
    output["predicted_fault_confirmed"] = (probability >= 0.5).astype(int)
    output["recommendation"] = np.where(probability < review_threshold, "remote_review", "normal_dispatch")
    output["recommendation_reason"] = np.where(
        probability < review_threshold,
        f"fault_probability < {review_threshold:.2f}; verify fresh telemetry; do not auto-cancel",
        f"fault_probability >= {review_threshold:.2f}; retain current dispatch",
    )
    return output
