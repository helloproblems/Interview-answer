# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains preparation materials for a timed CSV analysis task:

- `README.md`: business judgment, implementation scope, evidence, reproduction steps, and limitations.
- `AI_LOG.md`: actual AI prompts, mistakes, corrections, and verification records.
- `PREPARATION.md`: timing and submission checklist.
- `pyproject.toml`: project metadata and direct dependencies.
- `uv.lock`: resolved dependency versions; commit alongside dependency changes.
- `.python-version`: Python 3.10 selection.
- `.venv/`: ignored local environment.

No analysis code, datasets, tests, or assets exist yet. As needed, add executable analysis under `scripts/`, tests under `tests/`, supplied CSVs under `data/raw/`, and generated tables or charts under `outputs/`. Keep original inputs unchanged.

## Build, Test, and Development Commands

Use uv with Python 3.10. Run from the repository root in PowerShell:

```powershell
uv sync --locked
uv pip check
git diff --check
```

These commands synchronize `.venv` from the lockfile, check dependency compatibility, and detect whitespace errors. Use `uv add <package>` or `uv remove <package>` to manage dependencies, and `uv run --locked python <script>` to execute scripts. No build system or analysis entry point exists yet; document the analysis command when implemented.

## Coding Style & Naming Conventions

Use four-space Python indentation, PEP 8 conventions, `snake_case` functions and filenames, `PascalCase` classes, and `UPPER_CASE` constants. Keep files UTF-8 and preserve the existing Chinese documentation language. Prefer small functions with explicit input paths and documented output schemas. No formatter or linter is configured.

## Testing Guidelines

No test suite or coverage threshold is configured. For calculation logic, prefer standard-library `unittest`, with `tests/test_<feature>.py` files covering missing values, duplicates, zero denominators, and relevant boundaries. Once tests exist, run:

```powershell
uv run --locked python -m unittest discover -s tests -v
```

Label synthetic fixtures explicitly. Reproduce reported figures from the supplied CSV; fixtures cannot support business conclusions.

## Commit & Pull Request Guidelines

History contains only `first commit`; no established convention exists. Use short, imperative subjects such as `Add reproducible CSV analysis`. Keep commits focused. PR descriptions should state the problem, assumptions, changes, and validation commands; link relevant issues and include screenshots for visual changes.

## Evidence & Submission

Document metric definitions, denominators, exclusions, and uncertainty. Distinguish observations from hypotheses and estimated benefits. Update `AI_LOG.md` truthfully. Keep credentials and local environments out of Git. Push before submitting the repository URL; subsequent pushes require resubmission to count.
