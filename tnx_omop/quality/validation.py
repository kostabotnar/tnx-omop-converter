"""Checks of an OMOP CDM 5.4 output folder written by the converter.

Every check is a function over the output folder that returns a list of `Issue`
records; an empty list means the check passed. `validate()` runs all checks in
`CHECKS`, plus the schema check of each optional vocabulary or era table that is present
(see `checks_for`), and returns a `ValidationReport`. Tables are read with
`scan_parquet` and only the needed columns are collected, so the checks also run on
large outputs (tens of millions of rows).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ..schema.domains import (
    DOMAIN_TABLES,
    EVENT_CONCEPT_COLUMNS,
    NO_SOURCE_CONCEPT,
    NO_STANDARD_MAPPING,
    UNSUPPORTED_DOMAIN,
)
from ..merge import COVERAGE_FILE
from ..schema.omop import ERA_SCHEMAS, OMOP_SCHEMAS, VOCABULARY_SCHEMAS, TableSchema
from ..omop_vocab.concept_table import table_path
from ..transformers.clinical import COVERAGE_SCHEMA
from ..util import columns as col
from ..util import tables

CONCEPT_ID_SUFFIX = "_concept_id"
SOURCE_CONCEPT_ID_SUFFIX = "_source_concept_id"
STANDARD_CONCEPT = "S"
EXCLUSION_REASONS = [NO_SOURCE_CONCEPT, NO_STANDARD_MAPPING, UNSUPPORTED_DOMAIN]
# Report column holding the OMOP table name
OMOP_TABLE = "omop_table"
# Number of offending values listed in an error message
_SAMPLE_SIZE = 5
_TABLE_DOMAINS: dict[str, str] = {
    table_name: domain for domain, table_name in DOMAIN_TABLES.items()
}
# Tables without a person_id column
_NO_PERSON_TABLES = (tables.omop_concept, tables.omop_cdm_source)


@dataclass(frozen=True)
class Issue:
    """One problem found by a check.

    Attributes:
        table: OMOP table name, or None for a cross-table problem
        error_type: Short upper-case code such as MISSING_COLUMNS
        message: Human readable description
    """

    table: str | None
    error_type: str
    message: str

    def __str__(self) -> str:
        prefix = f"[{self.table}] " if self.table else ""
        return f"{prefix}{self.error_type}: {self.message}"


class CheckSkipped(Exception):
    """Raised by a check that cannot run, for example because a file is absent."""


@dataclass(frozen=True)
class Check:
    """A named check over an output folder."""

    name: str
    run: Callable[[Path], list[Issue]]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check: its issues, or the reason it was skipped."""

    name: str
    issues: list[Issue] = field(default_factory=list)
    skipped: str | None = None

    @property
    def passed(self) -> bool:
        """True when the check ran and found no issue."""
        return self.skipped is None and not self.issues


@dataclass(frozen=True)
class ValidationReport:
    """Results of all checks of one output folder."""

    results: list[CheckResult]

    @property
    def passed(self) -> list[CheckResult]:
        """Checks that ran and found nothing."""
        return [r for r in self.results if r.passed]

    @property
    def failed(self) -> list[CheckResult]:
        """Checks that found at least one issue."""
        return [r for r in self.results if r.issues]

    @property
    def skipped(self) -> list[CheckResult]:
        """Checks that could not run."""
        return [r for r in self.results if r.skipped is not None]

    @property
    def ok(self) -> bool:
        """True when no check failed (skipped checks do not fail)."""
        return not self.failed


def _sample(values: Sequence) -> list:
    return sorted(values)[:_SAMPLE_SIZE]


def _distinct_concept_ids(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Distinct non-zero, non-null values of one concept column, named concept_id."""
    return (
        lf.select(pl.col(column).cast(pl.Int64).alias(col.concept_id))
        .filter(pl.col(col.concept_id).is_not_null() & (pl.col(col.concept_id) != 0))
        .unique()
    )


def _concept_columns(lf: pl.LazyFrame) -> list[str]:
    return [
        name for name in lf.collect_schema().names() if name.endswith(CONCEPT_ID_SUFFIX)
    ]


def _types_compatible(actual: pl.DataType, expected: pl.DataType) -> bool:
    """Check if actual type is compatible with expected type."""
    if actual == expected:
        return True
    # A column with only nulls is written as Null; the required-column check
    # reports it when the column must not be null
    if actual == pl.Null:
        return True
    if isinstance(actual, pl.Datetime) and isinstance(expected, pl.Datetime):
        return True
    return actual in (pl.Utf8, pl.String) and expected in (pl.Utf8, pl.String)


def check_table(output_dir: Path, schema: TableSchema) -> list[Issue]:
    """Check one table against its schema.

    Covers file presence, columns, data types, primary key uniqueness and
    null-free required columns. Person references are checked by
    `check_person_references`.
    """
    path = table_path(output_dir, schema.name)
    if not path.exists():
        return [Issue(schema.name, "FILE_MISSING", f"Parquet file not found: {path}")]

    try:
        lf = pl.scan_parquet(path)
        file_schema = lf.collect_schema()
    except Exception as e:
        return [Issue(schema.name, "READ_ERROR", f"Failed to read Parquet file: {e}")]

    issues: list[Issue] = []
    names = set(file_schema.names())

    missing = set(schema.column_names) - names
    if missing:
        issues.append(
            Issue(schema.name, "MISSING_COLUMNS", f"Missing columns: {sorted(missing)}")
        )
    extra = names - set(schema.column_names)
    if extra:
        issues.append(
            Issue(schema.name, "EXTRA_COLUMNS", f"Unexpected columns: {sorted(extra)}")
        )

    dtype_map = schema.dtype_map
    mismatches = [
        f"{name}: expected {dtype_map[name]}, got {dtype}"
        for name, dtype in file_schema.items()
        if name in dtype_map and not _types_compatible(dtype, dtype_map[name])
    ]
    if mismatches:
        issues.append(
            Issue(schema.name, "TYPE_MISMATCH", f"Column type mismatches: {mismatches}")
        )

    pk_col = schema.primary_key
    # A missing primary key column is already reported as MISSING_COLUMNS
    if pk_col is not None and pk_col in file_schema:
        total_rows, unique_rows = (
            lf.select(pl.len(), pl.col(pk_col).n_unique())
            .collect(engine="streaming")
            .row(0)
        )
        if unique_rows != total_rows:
            issues.append(
                Issue(
                    schema.name,
                    "DUPLICATE_PRIMARY_KEY",
                    f"Primary key '{pk_col}' has {total_rows - unique_rows} "
                    "duplicate values",
                )
            )

    present = [c for c in schema.required_columns if c in file_schema]
    if present:
        counts = (
            lf.select(pl.col(c).null_count() for c in present)
            .collect(engine="streaming")
            .row(0)
        )
        null_columns = [
            f"{name}: {n} nulls" for name, n in zip(present, counts) if n > 0
        ]
        if null_columns:
            issues.append(
                Issue(
                    schema.name,
                    "NULL_REQUIRED_COLUMN",
                    f"Required columns with null values: {null_columns}",
                )
            )
    return issues


def check_cdm_source_row(output_dir: Path) -> list[Issue]:
    """CDM_SOURCE must hold exactly one row."""
    path = table_path(output_dir, tables.omop_cdm_source)
    if not path.exists():
        raise CheckSkipped(f"{path} not found")
    rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    if rows == 1:
        return []
    return [Issue(tables.omop_cdm_source, "ROW_COUNT", f"Expected 1 row, found {rows}")]


def check_all_tables_present(output_dir: Path) -> list[Issue]:
    """Every OMOP table of the converter output has a Parquet file."""
    missing = [
        name for name in OMOP_SCHEMAS if not table_path(output_dir, name).exists()
    ]
    if not missing:
        return []
    return [Issue(None, "TABLES_MISSING", f"Missing tables: {missing}")]


def check_person_references(output_dir: Path) -> list[Issue]:
    """Every person_id in every table with that column exists in PERSON."""
    person_path = table_path(output_dir, tables.omop_person)
    if not person_path.exists():
        raise CheckSkipped("PERSON table not available")
    known = pl.scan_parquet(person_path).select(pl.col(col.person_id).cast(pl.Int64))

    issues = []
    for table_name in [*OMOP_SCHEMAS, *ERA_SCHEMAS]:
        if table_name == tables.omop_person or table_name in _NO_PERSON_TABLES:
            continue
        path = table_path(output_dir, table_name)
        if not path.exists():
            continue  # Reported by check_table
        lf = pl.scan_parquet(path)
        if col.person_id not in lf.collect_schema():
            continue  # Reported by check_table
        missing = (
            lf.select(pl.col(col.person_id).cast(pl.Int64))
            .unique()
            .join(known, on=col.person_id, how="anti")
            .collect(engine="streaming")
            .get_column(col.person_id)
        )
        if missing.len():
            issues.append(
                Issue(
                    table_name,
                    "INVALID_PERSON_ID",
                    f"Found {missing.len()} person_id values not in PERSON table "
                    f"(examples: {_sample(missing.drop_nulls().to_list())})",
                )
            )
    return issues


def _scan_concepts(output_dir: Path) -> pl.LazyFrame:
    """CONCEPT columns used by the concept checks.

    Raises:
        CheckSkipped: when the CONCEPT file is missing (check_table reports it)
    """
    path = table_path(output_dir, tables.omop_concept)
    if not path.exists():
        raise CheckSkipped(f"{path} not found")
    return pl.scan_parquet(path).select(
        pl.col(col.concept_id).cast(pl.Int64),
        pl.col(col.domain_id),
        pl.col(col.standard_concept).fill_null(""),
    )


def _scan_tables(output_dir: Path) -> list[tuple[str, pl.LazyFrame]]:
    """Scans of every existing OMOP table (era tables included) except CONCEPT."""
    found = []
    for table_name in [*OMOP_SCHEMAS, *ERA_SCHEMAS]:
        path = table_path(output_dir, table_name)
        if table_name != tables.omop_concept and path.exists():
            found.append((table_name, pl.scan_parquet(path)))
    return found


def check_concepts_exist(output_dir: Path) -> list[Issue]:
    """Every non-zero *_concept_id value in every table is in CONCEPT.

    Covers source, type, unit, route, value and status concept columns too.
    """
    concept_ids = _scan_concepts(output_dir).select(col.concept_id)
    issues = []
    for table_name, lf in _scan_tables(output_dir):
        for column in _concept_columns(lf):
            missing = (
                _distinct_concept_ids(lf, column)
                .join(concept_ids, on=col.concept_id, how="anti")
                .collect(engine="streaming")
                .get_column(col.concept_id)
            )
            if missing.len():
                issues.append(
                    Issue(
                        table_name,
                        "MISSING_CONCEPT",
                        f"{column}: {missing.len()} concept IDs not in CONCEPT "
                        f"(examples: {_sample(missing.to_list())})",
                    )
                )
    return issues


def check_standard_or_zero(output_dir: Path) -> list[Issue]:
    """Every non-source *_concept_id value is 0 or a standard concept.

    *_source_concept_id columns hold source concepts and are skipped. No other
    column is exempt: type concepts (32817) and cdm_version_concept_id (756265)
    are standard in Athena. IDs missing from CONCEPT are reported by
    `check_concepts_exist`, not here.
    """
    concepts = _scan_concepts(output_dir)
    issues = []
    for table_name, lf in _scan_tables(output_dir):
        for column in _concept_columns(lf):
            if column.endswith(SOURCE_CONCEPT_ID_SUFFIX):
                continue
            non_standard = (
                _distinct_concept_ids(lf, column)
                .join(concepts, on=col.concept_id, how="inner")
                .filter(pl.col(col.standard_concept) != STANDARD_CONCEPT)
                .collect(engine="streaming")
                .get_column(col.concept_id)
            )
            if non_standard.len():
                issues.append(
                    Issue(
                        table_name,
                        "NON_STANDARD_CONCEPT",
                        f"{column}: {non_standard.len()} concept IDs are not "
                        f"standard (examples: {_sample(non_standard.to_list())})",
                    )
                )
    return issues


def check_event_domains(output_dir: Path) -> list[Issue]:
    """Check the main concept column of each clinical table.

    Per `domains.EVENT_CONCEPT_COLUMNS`, the column must be non-zero and its
    concepts must have the domain whose records are routed to that table.
    """
    concepts = _scan_concepts(output_dir)
    issues = []
    for table_name, column in EVENT_CONCEPT_COLUMNS.items():
        path = table_path(output_dir, table_name)
        if not path.exists():
            continue  # Reported by check_table
        lf = pl.scan_parquet(path)

        zero_rows = (
            lf.select((pl.col(column).fill_null(0) == 0).sum())
            .collect(engine="streaming")
            .item()
        )
        if zero_rows:
            issues.append(
                Issue(
                    table_name,
                    "ZERO_EVENT_CONCEPT",
                    f"{column}: {zero_rows} rows with concept 0 or null",
                )
            )

        expected = _TABLE_DOMAINS[table_name]
        wrong = (
            _distinct_concept_ids(lf, column)
            .join(concepts, on=col.concept_id, how="inner")
            .filter(pl.col(col.domain_id).fill_null("") != expected)
            .select(col.concept_id, col.domain_id)
            .collect(engine="streaming")
        )
        if wrong.height:
            issues.append(
                Issue(
                    table_name,
                    "DOMAIN_MISMATCH",
                    f"{column}: {wrong.height} concept IDs not in domain "
                    f"{expected} (examples: {_sample(wrong.rows())})",
                )
            )
    return issues


def check_era_concepts(output_dir: Path) -> list[Issue]:
    """Era concepts have the right kind: conditions, and drug ingredients.

    CONDITION_ERA concepts are in the Condition domain, DRUG_ERA concepts are
    concepts of class Ingredient. Runs on the era tables that exist; concepts
    missing from CONCEPT are reported by `check_concepts_exist`.

    Raises:
        CheckSkipped: when no era table exists
    """
    present = {
        name: column
        for name, column in (
            (tables.omop_condition_era, col.condition_concept_id),
            (tables.omop_drug_era, col.drug_concept_id),
        )
        if table_path(output_dir, name).exists()
    }
    if not present:
        raise CheckSkipped("no era table")
    concept_path = table_path(output_dir, tables.omop_concept)
    if not concept_path.exists():
        raise CheckSkipped(f"{concept_path} not found")
    concepts = pl.scan_parquet(concept_path).select(
        pl.col(col.concept_id).cast(pl.Int64), col.domain_id, col.concept_class_id
    )
    issues = []
    for table_name, column in present.items():
        is_condition = table_name == tables.omop_condition_era
        wrong_concept = (
            pl.col(col.domain_id).fill_null("") != "Condition"
            if is_condition
            else pl.col(col.concept_class_id).fill_null("") != "Ingredient"
        )
        wrong = (
            _distinct_concept_ids(
                pl.scan_parquet(table_path(output_dir, table_name)), column
            )
            .join(concepts, on=col.concept_id, how="inner")
            .filter(wrong_concept)
            .collect(engine="streaming")
            .get_column(col.concept_id)
        )
        if wrong.len():
            expected = "in domain Condition" if is_condition else "of class Ingredient"
            issues.append(
                Issue(
                    table_name,
                    "ERA_CONCEPT_MISMATCH",
                    f"{column}: {wrong.len()} concept IDs not {expected} "
                    f"(examples: {_sample(wrong.to_list())})",
                )
            )
    return issues


def table_row_counts(output_dir: Path) -> pl.DataFrame:
    """Rows per existing OMOP table, in OMOP_SCHEMAS order, then the era tables."""
    counts = [
        (table_name, pl.scan_parquet(path).select(pl.len()).collect().item())
        for table_name in [*OMOP_SCHEMAS, *ERA_SCHEMAS]
        for path in [table_path(output_dir, table_name)]
        if path.exists()
    ]
    return pl.DataFrame(
        counts, schema={OMOP_TABLE: pl.Utf8, col.rows: pl.Int64}, orient="row"
    )


def excluded_row_counts(output_dir: Path) -> pl.DataFrame:
    """Rows per TriNetX table and exclusion_reason from <output>/excluded/*.parquet.

    Each excluded file is named after its TriNetX table.
    """
    schema = {col.tnx_table: pl.Utf8, col.exclusion_reason: pl.Utf8, col.rows: pl.Int64}
    frames = [
        pl.scan_parquet(path)
        .group_by(col.exclusion_reason)
        .agg(pl.len().alias(col.rows))
        .select(
            pl.lit(path.stem, dtype=pl.Utf8).alias(col.tnx_table),
            pl.col(col.exclusion_reason).cast(pl.Utf8),
            pl.col(col.rows).cast(pl.Int64),
        )
        for path in sorted((output_dir / tables.excluded).glob("*.parquet"))
    ]
    if not frames:
        return pl.DataFrame(schema=schema)
    return (
        pl.concat(frames)
        .collect()
        .sort([col.tnx_table, col.rows], descending=[False, True])
    )


def coverage_report(output_dir: Path) -> str:
    """Text report of rows per OMOP table and excluded rows per TriNetX table and reason."""
    lines = ["Rows per OMOP table:"]
    lines += [
        f"  {name}: {rows:,}" for name, rows in table_row_counts(output_dir).iter_rows()
    ]
    lines += ["", "Excluded rows per TriNetX table and reason:"]
    lines += [
        f"  {tnx_table}: {reason} {rows:,}"
        for tnx_table, reason, rows in excluded_row_counts(output_dir).iter_rows()
    ]
    return "\n".join(lines)


def check_coverage_consistency(output_dir: Path) -> list[Issue]:
    """excluded/*.parquet and the table row counts agree with coverage.csv."""
    excluded_dir = output_dir / tables.excluded
    if not excluded_dir.exists():
        raise CheckSkipped(f"{excluded_dir} not found")
    coverage_path = output_dir / COVERAGE_FILE
    if not coverage_path.exists():
        raise CheckSkipped(f"{coverage_path} not found")

    table_rows = table_row_counts(output_dir)
    excluded = excluded_row_counts(output_dir)
    coverage = (
        pl.read_csv(coverage_path, schema=COVERAGE_SCHEMA)
        .group_by(col.tnx_table, col.outcome)
        .agg(pl.col(col.rows).sum())
    )

    issues = []
    expected_excluded = (
        coverage.filter(pl.col(col.outcome).is_in(EXCLUSION_REASONS))
        .rename({col.outcome: col.exclusion_reason})
        .sort(col.tnx_table, col.exclusion_reason)
    )
    if not excluded.sort(col.tnx_table, col.exclusion_reason).equals(
        expected_excluded.select(excluded.columns)
    ):
        issues.append(
            Issue(
                None,
                "COVERAGE_EXCLUDED_MISMATCH",
                f"excluded/*.parquet row counts differ from {COVERAGE_FILE}",
            )
        )

    mapped = dict(
        coverage.filter(pl.col(col.outcome).is_in(list(DOMAIN_TABLES.values())))
        .group_by(col.outcome)
        .agg(pl.col(col.rows).sum())
        .iter_rows()
    )
    written = dict(table_rows.iter_rows())
    mismatched = {
        name: (rows, written.get(name))
        for name, rows in mapped.items()
        if written.get(name) != rows
    }
    if mismatched:
        issues.append(
            Issue(
                None,
                "COVERAGE_TABLE_MISMATCH",
                f"{COVERAGE_FILE} rows vs table rows: {mismatched}",
            )
        )
    return issues


def _table_check(schema: TableSchema) -> Check:
    return Check(f"{schema.name} schema", lambda d: check_table(d, schema))


def check_optional_table(output_dir: Path, schema: TableSchema) -> list[Issue]:
    """Check a table that is only written on request, such as an exported vocabulary.

    Raises:
        CheckSkipped: when the table has no file
    """
    path = table_path(output_dir, schema.name)
    if not path.exists():
        raise CheckSkipped(f"{path} not found (optional table)")
    return check_table(output_dir, schema)


def _optional_table_check(schema: TableSchema) -> Check:
    return Check(f"{schema.name} schema", lambda d: check_optional_table(d, schema))


CHECKS: list[Check] = [
    Check("all tables present", check_all_tables_present),
    *(_table_check(schema) for schema in OMOP_SCHEMAS.values()),
    Check("CDM_SOURCE row count", check_cdm_source_row),
    Check("person references", check_person_references),
    Check("concepts exist", check_concepts_exist),
    Check("standard or zero", check_standard_or_zero),
    Check("event domains", check_event_domains),
    Check("coverage consistency", check_coverage_consistency),
]


# Checks of tables that only exist after `convert --export-vocabulary`; `validate`
# runs one when the table has a file and leaves it out otherwise.
OPTIONAL_CHECKS: dict[str, Check] = {
    name: _optional_table_check(schema)
    for name, schema in {**VOCABULARY_SCHEMAS, **ERA_SCHEMAS}.items()
}
ERA_CONCEPTS_CHECK = Check("era concepts", check_era_concepts)


def checks_for(output_dir: Path) -> list[Check]:
    """CHECKS plus the checks of the optional tables present in output_dir.

    The optional tables are the exported vocabulary tables and the era tables; the
    era concept check is added when an era table exists.
    """
    optional = [
        check
        for name, check in OPTIONAL_CHECKS.items()
        if table_path(output_dir, name).exists()
    ]
    if any(table_path(output_dir, name).exists() for name in ERA_SCHEMAS):
        optional.append(ERA_CONCEPTS_CHECK)
    return CHECKS + optional


def run_check(check: Check, output_dir: Path) -> CheckResult:
    """Run one check; a skip or an unexpected error becomes part of the result."""
    try:
        return CheckResult(check.name, check.run(output_dir))
    except CheckSkipped as skip:
        return CheckResult(check.name, skipped=str(skip))
    except Exception as e:
        issue = Issue(None, "CHECK_ERROR", f"{type(e).__name__}: {e}")
        return CheckResult(check.name, [issue])


def validate(output_dir: Path) -> ValidationReport:
    """Run every check in `CHECKS` and of the optional tables present on a folder."""
    return ValidationReport(
        [run_check(check, output_dir) for check in checks_for(output_dir)]
    )
