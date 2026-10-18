# Repository Guidelines

## Project Structure & Module Organization

This project trains fault_confirmed classifiers for electrical inspection tickets and exports advisory dispatch-review predictions.

- inspection_tickets.csv: original supplied data; preserve its bytes.
- scripts/common.py: validation, features, deduplication, and business metrics.
- scripts/train.py / scripts/predict.py: training and batch inference commands.
- tests/test_pipeline.py: synthetic behavioral and integration fixtures.
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
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m scripts.predict --input outputs/predict_input_example.csv --output outputs/example_predictions.csv
uv pip check
git diff --check
~~~

These commands synchronize the environment, regenerate model/evaluation artifacts, run tests, demonstrate unlabeled inference, and check dependencies and whitespace. No application server or package build is required. Use uv add <package> or uv remove <package> and commit both dependency files.

## Coding Style & Naming Conventions

Use four-space Python indentation, PEP 8, snake_case functions/files, PascalCase classes, and UPPER_CASE constants. Write UTF-8 files and preserve existing Chinese documentation. Share feature logic between training and prediction. No formatter or linter is configured.

## Testing Guidelines

Use standard-library unittest; name files test_<feature>.py and methods test_<behavior>. Cover timezones, duplicates, missing labels, threshold boundaries, invalid inputs, serialization, and inference without labels. No coverage percentage is mandated. Synthetic fixtures verify behavior only; business claims must use supplied data.

## Commit & Pull Request Guidelines

History uses short subjects such as first commit, 准备工作, and 数据; no formal convention exists. Keep changes focused with descriptive subjects. PRs should state the problem, assumptions, validation commands, metric changes, and relevant issues. Include updated plots when visual results change.

## Evidence & Model Boundaries

Deduplicate and exclude missing labels before chronological splitting. Select models and thresholds only within training data using expanding time folds. Evaluate on test data; reserve datasets/validation.csv for the user's explicit acceptance run via scripts.evaluate. Never use final validation for fitting, selection, or automatic evaluation. Record actual AI prompts and corrections. Predictions are advisory; do not cancel dispatch. Separate estimated savings from observed outcomes. Keep credentials and .venv/ out of Git; only load trusted joblib artifacts.
