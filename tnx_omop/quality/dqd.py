"""OHDSI Data Quality Dashboard (DQD) run on an output folder: `tnx-omop dqd`.

DQD is an R package that reads a SQL database, so `run_dqd()` runs `r/run_dqd.R`
with Rscript. The script creates every OMOP CDM 5.4 table in a DuckDB file from the
official DDL (CommonDataModel package), loads the output Parquet files into them and
runs `DataQualityDashboard::executeDqChecks()`, which writes a results JSON file.
Loading into the typed DDL tables is itself a test: a wrong type, a column name that
is not in the CDM or a null in a NOT NULL column stops the load.

`summarize()` reads the results and sets apart the failed checks that are accepted
in `config/dqd_accepted.json` (or the file given with `--accepted`): each entry is a
DQD check ID or an `fnmatch` pattern of IDs with the reason it is accepted.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections import deque
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

from ..schema.omop import ERA_SCHEMAS, OMOP_SCHEMAS, VOCABULARY_SCHEMAS
from ..omop_vocab.concept_table import table_path
from ..util import tables

logger = logging.getLogger(__name__)

R_DIR = Path(__file__).parent / "r"
R_SCRIPT = R_DIR / "run_dqd.R"
INSTALL_SCRIPT = R_DIR / "install_packages.R"
ACCEPTED_FILE = Path(__file__).parent.parent / "config" / "dqd_accepted.json"
RESULTS_DIR = "dqd"
RESULTS_FILE = "dqd_results.json"
DATABASE_FILE = "cdm.duckdb"
DEFAULT_SOURCE_NAME = "TriNetX"
# Exit status of run_dqd.R when an R package is missing
EXIT_MISSING_PACKAGES = 3
REQUIRED_TABLES = [
    tables.omop_person,
    tables.omop_observation_period,
    tables.omop_cdm_source,
    tables.omop_concept,
]
# R output lines repeated in the log when the script fails and the log level hides them
_TAIL_LINES = 40


class DqdError(Exception):
    """The DQD run or its input or result files are unusable."""


def output_tables(output_dir: Path) -> dict[str, Path]:
    """Lowercase CDM table name to Parquet file, for the tables present in output_dir."""
    names = [*OMOP_SCHEMAS, *VOCABULARY_SCHEMAS, *ERA_SCHEMAS]
    return {
        name.lower(): table_path(output_dir, name)
        for name in names
        if table_path(output_dir, name).exists()
    }


def _stream(command: list[str]) -> tuple[int, deque[str]]:
    """Run a command, log its output line by line at INFO; return the exit code and
    the last lines."""
    tail: deque[str] = deque(maxlen=_TAIL_LINES)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            tail.append(line)
            logger.info("R: %s", line)
    return process.returncode, tail


def install_r_packages(rscript: str) -> int:
    """Run the bundled install_packages.R with rscript and return its exit code.

    The last lines of the output are repeated at ERROR when the script fails and
    INFO messages are not shown.
    """
    logger.info("Installing the R packages with %s", rscript)
    returncode, tail = _stream([rscript, str(INSTALL_SCRIPT)])
    if returncode != 0 and not logger.isEnabledFor(logging.INFO):
        for line in tail:
            logger.error("R: %s", line)
    return returncode


def run_dqd(
    output_dir: Path,
    results_dir: Path,
    rscript: str,
    source_name: str = DEFAULT_SOURCE_NAME,
    keep_database: bool = False,
) -> Path:
    """Load output_dir into DuckDB and run DQD; return the results JSON path.

    The R output is logged line by line at INFO; the last lines are repeated at
    ERROR when the script fails.

    Raises:
        DqdError: a required table is missing, an R package is missing, or the
            script fails or writes no results
    """
    found = output_tables(output_dir)
    missing = [name for name in REQUIRED_TABLES if name.lower() not in found]
    if missing:
        raise DqdError(f"{output_dir} has no {', '.join(missing)} table")
    if tables.omop_concept_ancestor not in found:
        logger.warning(
            "No exported vocabulary tables in %s: CONCEPT holds only the referenced "
            "concepts and the other vocabulary tables stay empty. Convert without "
            "--no-export-vocabulary for a complete DQD run.",
            output_dir,
        )
    if tables.omop_condition_era.lower() not in found:
        logger.warning(
            "No era tables in %s: the era checks will fail. Convert without "
            "--no-eras for a complete DQD run.",
            output_dir,
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / RESULTS_FILE
    results_path.unlink(missing_ok=True)
    command = [
        rscript,
        str(R_SCRIPT),
        str(results_dir / DATABASE_FILE),
        str(results_path),
        source_name,
        "true" if keep_database else "false",
        *(f"{name}={path}" for name, path in found.items()),
    ]
    logger.info("Running DQD with %s on %d tables", rscript, len(found))
    returncode, tail = _stream(command)
    if returncode == EXIT_MISSING_PACKAGES:
        raise DqdError(
            "R packages are missing; install them with: tnx-omop dqd "
            "--install-r-packages"
        )
    if returncode != 0:
        if not logger.isEnabledFor(logging.INFO):
            for line in tail:
                logger.error("R: %s", line)
        raise DqdError(f"{R_SCRIPT.name} failed with exit code {returncode}")
    if not results_path.exists():
        raise DqdError(f"{R_SCRIPT.name} wrote no results file {results_path}")
    return results_path


@dataclass(frozen=True)
class DqdCheck:
    """One row of the DQD `CheckResults`."""

    check_id: str
    check_name: str
    table: str
    field: str | None
    category: str
    subcategory: str
    violated_rows: int | None
    denominator_rows: int | None
    pct_violated: float | None
    threshold: float | None
    failed: bool
    is_error: bool
    not_applicable: bool
    error: str | None


def _flag(value: object) -> bool:
    return value in (1, True, "1")


def _number(value: object) -> float | None:
    """A JSON number, or None for null, a missing value or R's "NA" string."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return value
    try:
        return float(str(value))
    except ValueError:
        return None


def _int(value: object) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _text(value: object) -> str | None:
    return None if value in (None, "", "NA") else str(value)


def load_results(path: Path) -> list[DqdCheck]:
    """Read the `CheckResults` of a DQD results JSON file.

    Raises:
        DqdError: the file is unreadable or has no `CheckResults` list
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DqdError(f"Cannot read DQD results {path}: {e}") from e
    rows = data.get("CheckResults") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise DqdError(f"{path} has no CheckResults list")
    return [
        DqdCheck(
            check_id=str(row.get("checkId", "")),
            check_name=str(row.get("checkName", "")),
            table=str(row.get("cdmTableName", "")),
            field=_text(row.get("cdmFieldName")),
            category=str(row.get("category", "")),
            subcategory=str(row.get("subcategory", "")),
            violated_rows=_int(row.get("numViolatedRows")),
            denominator_rows=_int(row.get("numDenominatorRows")),
            pct_violated=_number(row.get("pctViolatedRows")),
            threshold=_number(row.get("thresholdValue")),
            failed=_flag(row.get("failed")),
            is_error=_flag(row.get("isError")),
            not_applicable=_flag(row.get("notApplicable")),
            error=_text(row.get("error")),
        )
        for row in rows
    ]


def load_accepted(path: Path) -> dict[str, str]:
    """Read an accepted-failures file: check ID pattern to reason.

    The file is `{"accepted": [{"check_id": "...", "reason": "..."}, ...]}`.

    Raises:
        DqdError: the file is unreadable or an entry lacks a check ID or a reason
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DqdError(f"Cannot read accepted DQD checks {path}: {e}") from e
    entries = data.get("accepted") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise DqdError(f"{path} must hold an object with an 'accepted' list")
    accepted: dict[str, str] = {}
    for i, entry in enumerate(entries):
        pattern = entry.get("check_id") if isinstance(entry, dict) else None
        reason = entry.get("reason") if isinstance(entry, dict) else None
        if not isinstance(pattern, str) or not pattern:
            raise DqdError(f"{path}: entry {i} has no check_id")
        if not isinstance(reason, str) or not reason.strip():
            raise DqdError(f"{path}: entry {i} ({pattern}) has no reason")
        accepted[pattern.lower()] = reason
    return accepted


@dataclass
class DqdSummary:
    """Checks by outcome; failures and errors matched by an accepted pattern are
    in `accepted`, not in `failed` or `errors`."""

    passed: list[DqdCheck] = field(default_factory=list)
    failed: list[DqdCheck] = field(default_factory=list)
    errors: list[DqdCheck] = field(default_factory=list)
    accepted: list[DqdCheck] = field(default_factory=list)
    not_applicable: list[DqdCheck] = field(default_factory=list)
    unused_patterns: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.errors


def summarize(checks: list[DqdCheck], accepted: dict[str, str]) -> DqdSummary:
    """Sort checks by outcome and list the accepted patterns that matched nothing.

    A SQL error of a check counts as an error, not as a failure; both can be
    accepted, because some DQD queries may not run on DuckDB.
    """
    summary = DqdSummary()
    used: set[str] = set()
    for check in checks:
        if not (check.is_error or check.failed):
            target = summary.not_applicable if check.not_applicable else summary.passed
            target.append(check)
            continue
        patterns = [p for p in accepted if fnmatchcase(check.check_id.lower(), p)]
        used.update(patterns)
        if patterns:
            summary.accepted.append(check)
        elif check.is_error:
            summary.errors.append(check)
        else:
            summary.failed.append(check)
    summary.unused_patterns = [p for p in accepted if p not in used]
    return summary


def _describe_failure(check: DqdCheck) -> str:
    location = check.table + (f".{check.field}" if check.field else "")
    rows = (
        f"{check.violated_rows:,} of {check.denominator_rows:,} rows"
        if check.violated_rows is not None and check.denominator_rows is not None
        else "rows unknown"
    )
    pct = (
        f" ({check.pct_violated * 100:.2f}%)" if check.pct_violated is not None else ""
    )
    threshold = (
        f"threshold {check.threshold:g}%"
        if check.threshold is not None
        else "no threshold"
    )
    return (
        f"FAILED {check.check_id}: {location}, {rows}{pct}, {threshold} "
        f"[{check.category}/{check.subcategory}]"
    )


def format_summary(summary: DqdSummary) -> str:
    """One line per failed check and per error, then the counts."""
    lines = [_describe_failure(check) for check in summary.failed]
    lines += [
        f"ERROR {check.check_id}: {(check.error or 'unknown error').splitlines()[0]}"
        for check in summary.errors
    ]
    lines.append(
        f"{len(summary.passed)} passed, {len(summary.failed)} failed, "
        f"{len(summary.errors)} errors, {len(summary.accepted)} accepted, "
        f"{len(summary.not_applicable)} not applicable"
    )
    return "\n".join(lines)
