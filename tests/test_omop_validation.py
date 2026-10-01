"""Run the package validation checks as pytest tests on a converted output folder.

Usage:
    # Run every check on an output directory
    uv run pytest tests/test_omop_validation.py -v --output-dir=/path/to/output

    # Run the checks of one table
    uv run pytest tests/test_omop_validation.py -k "PERSON" -v --output-dir=/path/to/output

    # Print the coverage report (rows per table, excluded rows per reason)
    uv run pytest tests/test_omop_validation.py -k report -s --output-dir=/path/to/output

The checks live in tnx_omop/quality/validation.py; `tnx-omop validate <dir>` runs the same
ones without pytest. Every test skips when --output-dir is not given.
"""

from pathlib import Path

import pytest

from tnx_omop.util import tables
from tnx_omop.quality.validation import (
    CHECKS,
    OPTIONAL_CHECKS,
    Check,
    coverage_report,
    run_check,
)


@pytest.fixture
def output_dir(request: pytest.FixtureRequest) -> Path:
    """Output directory from --output-dir; skips the test when it is not given."""
    value = request.config.getoption("--output-dir")
    if value is None:
        pytest.skip("No --output-dir specified")
    return Path(value)


ALL_CHECKS = CHECKS + list(OPTIONAL_CHECKS.values())


@pytest.mark.parametrize("check", ALL_CHECKS, ids=[check.name for check in ALL_CHECKS])
def test_check(check: Check, output_dir: Path) -> None:
    """One validation check passes on the output folder."""
    result = run_check(check, output_dir)
    if result.skipped is not None:
        pytest.skip(result.skipped)
    assert not result.issues, "\n".join(str(issue) for issue in result.issues)


def test_coverage_report(output_dir: Path) -> None:
    """Print the coverage report (run with -s)."""
    if not (output_dir / tables.excluded).exists():
        pytest.skip(f"{output_dir / tables.excluded} not found")
    print("\n" + coverage_report(output_dir))
