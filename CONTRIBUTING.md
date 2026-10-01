# Contributing

Thank you for considering a contribution. Please read the patient data rule below first.

## Patient data rule

TriNetX exports contain patient-level data under data use agreements.

- Never attach or paste TriNetX files, converted OMOP output rows, or anything derived from patient records to issues, discussions, or pull requests. Use synthetic examples, aggregate counts, logs with data rows removed, or `coverage.csv` style summaries.
- Tests use synthetic data only. Never commit TriNetX files or output derived from them.

See [SECURITY.md](SECURITY.md) for the full rule and for reporting vulnerabilities.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) and a supported Python version (see `requires-python` in `pyproject.toml`; uv installs one when needed).

```bash
git clone https://github.com/kostabotnar/tnx-omop-converter.git
cd tnx-omop-converter
uv sync
```

## Tests

```bash
uv run pytest -q
```

The Data Quality Dashboard (DQD) test needs R, a Java JDK, and the R packages installed with `uv run tnx-omop dqd --install-r-packages`. Without them it is skipped.

## Lint and format

```bash
uvx ruff@0.15.10 format .
uvx ruff@0.15.10 check .
```

## Code conventions

Conventions used in this project:

- Use Polars expressions (`with_columns()`, `select()`), not row-by-row loops.
- Use the constants in `tnx_omop/util/tables.py` and `tnx_omop/util/columns.py` instead of string literals.
- Use `logging` (`logger = logging.getLogger(__name__)`), never `print`.
- Use concept 0, not null, when a record has no concept.
- Write Parquet output through `tnx_omop/util/parquet_io.py`.
- Use built-in generics (`list[str]`) and `X | None` in new code.
- Comment only non-obvious reasoning.

Every change needs tests. Update README.md and CHANGELOG.md when behavior changes.

## Pull requests

- Keep them small and focused, one logical change each.
- Open them against the `dev` branch.
- Fill in the pull request template checklist.

## License

By contributing, you agree that your work is licensed under the Apache License 2.0, the license of this project.

## Conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
