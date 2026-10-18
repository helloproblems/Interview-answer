# Repository Guidelines

## Project Structure & Module Organization

Trains fault_confirmed classifiers and exports advisory inspection predictions.

- inspection_tickets.csv: original supplied data; preserve its bytes.
- scripts/common.py: validation, features, deduplication, and business metrics.
- scripts/train.py / scripts/optimize.py / scripts/predict.py: baseline training, candidate search, and inference.
- scripts/evaluate.py / scripts/report.py: explicit acceptance evaluation and saved-result reporting.
- tests/: synthetic pipeline and report checks.
- datasets/: labeled training, test, and user acceptance validation CSVs.
- artifacts/: trained pipeline and model metadata.
- outputs/: reproducible evaluation tables, plots, and historical predictions.
- README.md, docs/TASK.md, AI_LOG.md: findings, task specification, and actual AI interactions.
- pyproject.toml, uv.lock, .python-version: uv dependencies and Python 3.10 selection.

## Build, Test, and Development Commands

Run from the repository root:

~~~powershell
uv sync --locked
uv run --locked python -m scripts.train
uv run --locked python -m scripts.optimize
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m scripts.report --output-dir outputs/user_acceptance
uv run --locked python -m scripts.predict --input outputs/predict_input_example.csv --output outputs/example_predictions.csv
uv pip check
git diff --check
~~~

Commands synchronize dependencies, train, search, test, report, predict, and check consistency. Candidate outputs live in outputs/optimization/ and artifacts/optimized/. Open report.html locally. Use uv add/remove and commit both dependency files.

## Coding Style & Naming Conventions

Use four-space indentation, PEP 8, snake_case functions/files, PascalCase classes, and UPPER_CASE constants. Preserve UTF-8 Chinese documentation. Share training/inference features. No formatter or linter is configured.

## Testing Guidelines

Use standard-library unittest; name files test_<feature>.py and methods test_<behavior>. Cover data validation, leakage boundaries, serialization, inference without labels, report consistency, and HTML escaping. No coverage percentage is mandated. Synthetic fixtures verify behavior only; business claims must use supplied data.

## Commit & Pull Request Guidelines

Use descriptive commit subjects; history has no formal convention. PRs should document assumptions, checks, metric changes, and relevant issues. Include updated visual results when applicable.

## Evidence & Model Boundaries

Deduplicate and exclude missing labels before chronological splitting. Select models and thresholds only within training data using expanding time folds. Freeze candidates before test evaluation; retain the default model when tests regress. Read classification thresholds from model metadata (legacy default: 0.50). Reserve datasets/validation.csv for the user's explicit acceptance run via scripts.evaluate; never fit, select, or automatically evaluate on it. Record actual AI prompts and corrections. Predictions are advisory; do not cancel dispatch. Separate estimated savings from observed outcomes. Keep credentials and .venv/ out of Git; only load trusted joblib artifacts.
