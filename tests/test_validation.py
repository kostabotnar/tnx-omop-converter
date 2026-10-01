"""Unit tests for tnx_omop.validation on small hand-written and converted outputs."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tnx_omop import validation
from tnx_omop.schema.domains import NO_SOURCE_CONCEPT, NO_STANDARD_MAPPING
from tnx_omop.merge import COVERAGE_FILE
from tnx_omop.schema.omop import (
    CONCEPT_SCHEMA,
    ERA_SCHEMAS,
    OMOP_SCHEMAS,
    PERSON_SCHEMA,
    VOCABULARY_SCHEMAS,
)
from tnx_omop.transformers.clinical import COVERAGE_SCHEMA
from tnx_omop.util import columns as col
from tnx_omop.util import tables
from tnx_omop.validation import (
    STANDARD_CONCEPT,
    Issue,
    check_all_tables_present,
    check_cdm_source_row,
    check_concepts_exist,
    check_coverage_consistency,
    check_era_concepts,
    check_event_domains,
    check_person_references,
    check_standard_or_zero,
    check_table,
    coverage_report,
    excluded_row_counts,
    table_row_counts,
)


def _write_test_parquet(tmp_path: Path, table_name: str, df: pl.DataFrame) -> None:
    """Write a test parquet file with subdirectory structure."""
    table_lower = table_name.lower()
    table_dir = tmp_path / table_lower
    table_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(table_dir / f"{table_lower}.parquet")


# Concept IDs for the concept check unit tests
_CONDITION = 201826  # standard, Condition domain
_CONDITION_SOURCE = 45576876  # non-standard ICD10CM, Condition domain
_OBSERVATION = 4144272  # standard, Observation domain
_EHR = 32817  # standard type concept


def _write_concepts(tmp_path: Path) -> None:
    """CONCEPT with only the columns the concept checks read."""
    rows = [
        (_CONDITION, "Condition", STANDARD_CONCEPT),
        (_CONDITION_SOURCE, "Condition", None),
        (_OBSERVATION, "Observation", STANDARD_CONCEPT),
        (_EHR, "Type Concept", STANDARD_CONCEPT),
    ]
    schema = {
        col.concept_id: pl.Int64,
        col.domain_id: pl.Utf8,
        col.standard_concept: pl.Utf8,
    }
    _write_test_parquet(
        tmp_path,
        tables.omop_concept,
        pl.DataFrame(rows, schema=schema, orient="row"),
    )


def _write_conditions(
    tmp_path: Path, concept_ids: list[int], source_concept_ids: list[int]
) -> None:
    """CONDITION_OCCURRENCE with only its ID and concept columns."""
    n = len(concept_ids)
    _write_test_parquet(
        tmp_path,
        tables.omop_condition_occurrence,
        pl.DataFrame(
            {
                col.condition_occurrence_id: list(range(1, n + 1)),
                col.condition_concept_id: concept_ids,
                col.condition_type_concept_id: [_EHR] * n,
                col.condition_source_concept_id: source_concept_ids,
            },
            schema_overrides={
                col.condition_concept_id: pl.Int64,
                col.condition_source_concept_id: pl.Int64,
            },
        ),
    )


def _error_types(errors: list[Issue]) -> list[str]:
    return [e.error_type for e in errors]


class TestChecks:
    """Unit tests for the checks that do not need a converted output."""

    def test_validate_columns_present_detects_missing(self, tmp_path: Path) -> None:
        """Test that missing columns are detected."""
        # Create a DataFrame missing some columns
        df = pl.DataFrame(
            {
                "person_id": [1, 2],
                "gender_concept_id": [0, 0],
                "year_of_birth": [1990, 1980],
            }
        )
        _write_test_parquet(tmp_path, "PERSON", df)

        errors = check_table(tmp_path, PERSON_SCHEMA)

        missing_errors = [e for e in errors if e.error_type == "MISSING_COLUMNS"]
        assert len(missing_errors) == 1
        assert "race_concept_id" in missing_errors[0].message

    def test_validate_extra_columns_detected(self, tmp_path: Path) -> None:
        """Test that extra columns are detected."""
        # Create DataFrame with all required columns plus extra
        df = pl.DataFrame({spec.name: [] for spec in PERSON_SCHEMA.columns})
        df = df.with_columns(pl.lit(None).alias("unexpected_column"))
        _write_test_parquet(tmp_path, "PERSON", df)

        errors = check_table(tmp_path, PERSON_SCHEMA)

        extra_errors = [e for e in errors if e.error_type == "EXTRA_COLUMNS"]
        assert len(extra_errors) == 1
        assert "unexpected_column" in extra_errors[0].message

    def test_validate_type_mismatch_detected(self, tmp_path: Path) -> None:
        """Test that type mismatches are detected."""
        # Create DataFrame with wrong type for person_id
        df = pl.DataFrame(
            {
                "person_id": ["one", "two"],  # Should be Int64
                "gender_concept_id": [0, 0],
                "year_of_birth": [1990, 1980],
                "month_of_birth": [None, None],
                "day_of_birth": [None, None],
                "birth_datetime": [None, None],
                "race_concept_id": [0, 0],
                "ethnicity_concept_id": [0, 0],
                "location_id": [None, None],
                "provider_id": [None, None],
                "care_site_id": [None, None],
                "person_source_value": [None, None],
                "gender_source_value": [None, None],
                "gender_source_concept_id": [0, 0],
                "race_source_value": [None, None],
                "race_source_concept_id": [0, 0],
                "ethnicity_source_value": [None, None],
                "ethnicity_source_concept_id": [0, 0],
            }
        )
        _write_test_parquet(tmp_path, "PERSON", df)

        errors = check_table(tmp_path, PERSON_SCHEMA)

        type_errors = [e for e in errors if e.error_type == "TYPE_MISMATCH"]
        assert len(type_errors) == 1
        assert "person_id" in type_errors[0].message

    def test_validate_duplicate_primary_key_detected(self, tmp_path: Path) -> None:
        """Test that duplicate primary keys are detected."""
        df = pl.DataFrame(
            {
                "person_id": [1, 1, 2],  # Duplicate!
                "gender_concept_id": [0, 0, 0],
                "year_of_birth": [1990, 1990, 1980],
                "month_of_birth": [None, None, None],
                "day_of_birth": [None, None, None],
                "birth_datetime": [None, None, None],
                "race_concept_id": [0, 0, 0],
                "ethnicity_concept_id": [0, 0, 0],
                "location_id": [None, None, None],
                "provider_id": [None, None, None],
                "care_site_id": [None, None, None],
                "person_source_value": [None, None, None],
                "gender_source_value": [None, None, None],
                "gender_source_concept_id": [0, 0, 0],
                "race_source_value": [None, None, None],
                "race_source_concept_id": [0, 0, 0],
                "ethnicity_source_value": [None, None, None],
                "ethnicity_source_concept_id": [0, 0, 0],
            }
        )
        _write_test_parquet(tmp_path, "PERSON", df)

        errors = check_table(tmp_path, PERSON_SCHEMA)

        pk_errors = [e for e in errors if e.error_type == "DUPLICATE_PRIMARY_KEY"]
        assert len(pk_errors) == 1
        assert "1 duplicate" in pk_errors[0].message

    def test_validate_null_required_column_detected(self, tmp_path: Path) -> None:
        """Test that null values in required columns are detected."""
        df = pl.DataFrame(
            {
                "person_id": [1, 2],
                "gender_concept_id": [0, None],  # Null in required column!
                "year_of_birth": [1990, 1980],
                "month_of_birth": [None, None],
                "day_of_birth": [None, None],
                "birth_datetime": [None, None],
                "race_concept_id": [0, 0],
                "ethnicity_concept_id": [0, 0],
                "location_id": [None, None],
                "provider_id": [None, None],
                "care_site_id": [None, None],
                "person_source_value": [None, None],
                "gender_source_value": [None, None],
                "gender_source_concept_id": [0, 0],
                "race_source_value": [None, None],
                "race_source_concept_id": [0, 0],
                "ethnicity_source_value": [None, None],
                "ethnicity_source_concept_id": [0, 0],
            }
        )
        _write_test_parquet(tmp_path, "PERSON", df)

        errors = check_table(tmp_path, PERSON_SCHEMA)

        null_errors = [e for e in errors if e.error_type == "NULL_REQUIRED_COLUMN"]
        assert len(null_errors) == 1
        assert "gender_concept_id" in null_errors[0].message

    def test_validate_referential_integrity(self, tmp_path: Path) -> None:
        """Test that invalid person_id references are detected."""
        # Create PERSON with ID 1
        person_df = pl.DataFrame(
            {
                "person_id": [1],
                "gender_concept_id": [0],
                "year_of_birth": [1990],
                "month_of_birth": [None],
                "day_of_birth": [None],
                "birth_datetime": [None],
                "race_concept_id": [0],
                "ethnicity_concept_id": [0],
                "location_id": [None],
                "provider_id": [None],
                "care_site_id": [None],
                "person_source_value": [None],
                "gender_source_value": [None],
                "gender_source_concept_id": [0],
                "race_source_value": [None],
                "race_source_concept_id": [0],
                "ethnicity_source_value": [None],
                "ethnicity_source_concept_id": [0],
            }
        )
        _write_test_parquet(tmp_path, "PERSON", person_df)

        # Create VISIT_OCCURRENCE with person_id 999 (doesn't exist)
        visit_df = pl.DataFrame(
            {
                "visit_occurrence_id": [100],
                "person_id": [999],  # Invalid reference!
                "visit_concept_id": [0],
                "visit_start_date": [date(2023, 1, 1)],
                "visit_start_datetime": [None],
                "visit_end_date": [None],
                "visit_end_datetime": [None],
                "visit_type_concept_id": [32817],
                "provider_id": [None],
                "care_site_id": [None],
                "visit_source_value": [None],
                "visit_source_concept_id": [0],
                "admitted_from_concept_id": [0],
                "admitted_from_source_value": [None],
                "discharged_to_concept_id": [0],
                "discharged_to_source_value": [None],
                "preceding_visit_occurrence_id": [None],
            }
        )
        _write_test_parquet(tmp_path, "VISIT_OCCURRENCE", visit_df)

        errors = check_person_references(tmp_path)

        ref_errors = [e for e in errors if e.error_type == "INVALID_PERSON_ID"]
        assert len(ref_errors) == 1
        assert "999" in ref_errors[0].message

    def test_validate_valid_table_passes(self, tmp_path: Path) -> None:
        """Test that a valid table passes all validations."""
        df = pl.DataFrame(
            {
                "person_id": [1, 2],
                "gender_concept_id": [0, 0],
                "year_of_birth": [1990, 1980],
                "month_of_birth": [None, None],
                "day_of_birth": [None, None],
                "birth_datetime": [None, None],
                "race_concept_id": [0, 0],
                "ethnicity_concept_id": [0, 0],
                "location_id": [None, None],
                "provider_id": [None, None],
                "care_site_id": [None, None],
                "person_source_value": ["P1", "P2"],
                "gender_source_value": [None, None],
                "gender_source_concept_id": [0, 0],
                "race_source_value": [None, None],
                "race_source_concept_id": [0, 0],
                "ethnicity_source_value": [None, None],
                "ethnicity_source_concept_id": [0, 0],
            }
        )
        _write_test_parquet(tmp_path, "PERSON", df)

        errors = check_table(tmp_path, PERSON_SCHEMA)

        assert not errors, f"Expected no errors, got: {[str(e) for e in errors]}"

    def test_validate_empty_table_passes(self, tmp_path: Path) -> None:
        """Test that an empty table with correct schema passes."""
        df = pl.DataFrame(
            {
                spec.name: pl.Series([], dtype=spec.dtype)
                for spec in PERSON_SCHEMA.columns
            }
        )
        _write_test_parquet(tmp_path, "PERSON", df)

        errors = check_table(tmp_path, PERSON_SCHEMA)

        assert not errors, f"Expected no errors, got: {[str(e) for e in errors]}"

    def test_validate_missing_file(self, tmp_path: Path) -> None:
        """Test that missing file is reported."""
        errors = check_table(tmp_path, PERSON_SCHEMA)

        assert len(errors) == 1
        assert errors[0].error_type == "FILE_MISSING"

    @staticmethod
    def _concept_df(valid_start_date) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "concept_id": [1, 2],
                "concept_name": [
                    "Type 2 diabetes mellitus without complications",
                    "Essential hypertension",
                ],
                "domain_id": ["Condition", "Condition"],
                "vocabulary_id": ["ICD-10-CM", "ICD-10-CM"],
                "concept_class_id": ["ICD-10-CM", "ICD-10-CM"],
                "standard_concept": pl.Series([None, None], dtype=pl.Utf8),
                "concept_code": ["ICD-10-CM:E11.9", "ICD-10-CM:I10"],
                "valid_start_date": pl.Series(valid_start_date, dtype=pl.Date),
                "valid_end_date": [date(2099, 12, 31)] * 2,
                "invalid_reason": pl.Series([None, None], dtype=pl.Utf8),
            }
        )

    def test_validate_concept_table(self, tmp_path: Path) -> None:
        """Test concept table validation."""
        _write_test_parquet(
            tmp_path, "concept", self._concept_df([date(1970, 1, 1)] * 2)
        )

        errors = check_table(tmp_path, CONCEPT_SCHEMA)

        assert not errors, f"Expected no errors, got: {[str(e) for e in errors]}"

    def test_validate_concept_table_requires_valid_dates(self, tmp_path: Path) -> None:
        """Null valid_start_date is reported, as CDM 5.4 requires it."""
        _write_test_parquet(tmp_path, "concept", self._concept_df([None, None]))

        errors = check_table(tmp_path, CONCEPT_SCHEMA)

        assert [e.error_type for e in errors] == ["NULL_REQUIRED_COLUMN"]

    def test_concepts_exist_passes(self, tmp_path: Path) -> None:
        """Source, type and event concepts all in CONCEPT; zero is ignored."""
        _write_concepts(tmp_path)
        _write_conditions(tmp_path, [_CONDITION, _CONDITION], [_CONDITION_SOURCE, 0])

        assert check_concepts_exist(tmp_path) == []

    def test_concepts_exist_detects_missing(self, tmp_path: Path) -> None:
        """A concept ID absent from CONCEPT is reported with its column."""
        _write_concepts(tmp_path)
        _write_conditions(tmp_path, [_CONDITION, 999], [_CONDITION_SOURCE, 998])

        errors = check_concepts_exist(tmp_path)

        assert _error_types(errors) == ["MISSING_CONCEPT", "MISSING_CONCEPT"]
        messages = " ".join(e.message for e in errors)
        assert "condition_concept_id: 1 concept IDs" in messages
        assert "[999]" in messages
        assert "condition_source_concept_id: 1 concept IDs" in messages

    def test_concept_checks_skip_without_concept_file(self, tmp_path: Path) -> None:
        """Without CONCEPT every concept check is skipped."""
        _write_conditions(tmp_path, [_CONDITION], [0])

        for check in (
            check_concepts_exist,
            check_standard_or_zero,
            check_event_domains,
        ):
            with pytest.raises(validation.CheckSkipped):
                check(tmp_path)

    def test_standard_or_zero_passes(self, tmp_path: Path) -> None:
        """Zero and standard concepts pass; source columns may be non-standard."""
        _write_concepts(tmp_path)
        _write_conditions(tmp_path, [_CONDITION, 0], [_CONDITION_SOURCE, 0])

        assert check_standard_or_zero(tmp_path) == []

    def test_standard_or_zero_detects_non_standard(self, tmp_path: Path) -> None:
        """A non-standard concept in a non-source column is reported."""
        _write_concepts(tmp_path)
        _write_conditions(tmp_path, [_CONDITION_SOURCE], [_CONDITION_SOURCE])

        errors = check_standard_or_zero(tmp_path)

        assert _error_types(errors) == ["NON_STANDARD_CONCEPT"]
        assert errors[0].table == tables.omop_condition_occurrence
        assert errors[0].message.startswith("condition_concept_id:")
        assert str(_CONDITION_SOURCE) in errors[0].message

    def test_event_domains_passes(self, tmp_path: Path) -> None:
        """Non-zero Condition concepts in CONDITION_OCCURRENCE pass."""
        _write_concepts(tmp_path)
        _write_conditions(tmp_path, [_CONDITION, _CONDITION], [0, 0])

        assert check_event_domains(tmp_path) == []

    def test_event_domains_detects_zero(self, tmp_path: Path) -> None:
        """Rows with event concept 0 are reported."""
        _write_concepts(tmp_path)
        _write_conditions(tmp_path, [_CONDITION, 0], [0, 0])

        errors = check_event_domains(tmp_path)

        assert _error_types(errors) == ["ZERO_EVENT_CONCEPT"]
        assert "1 rows" in errors[0].message

    def test_event_domains_detects_domain_mismatch(self, tmp_path: Path) -> None:
        """An Observation concept in CONDITION_OCCURRENCE is reported."""
        _write_concepts(tmp_path)
        _write_conditions(tmp_path, [_CONDITION, _OBSERVATION], [0, 0])

        errors = check_event_domains(tmp_path)

        assert _error_types(errors) == ["DOMAIN_MISMATCH"]
        assert "not in domain Condition" in errors[0].message
        assert f"({_OBSERVATION}, 'Observation')" in errors[0].message

    def test_coverage_row_counts(self, tmp_path: Path) -> None:
        """Rows per OMOP table and excluded rows per TriNetX table and reason."""
        _write_conditions(tmp_path, [_CONDITION, _CONDITION], [0, 0])
        excluded_dir = tmp_path / tables.excluded
        excluded_dir.mkdir()
        pl.DataFrame(
            {
                col.code: ["C79.51", "C79.52", "X"],
                col.exclusion_reason: [
                    NO_STANDARD_MAPPING,
                    NO_STANDARD_MAPPING,
                    NO_SOURCE_CONCEPT,
                ],
            }
        ).write_parquet(excluded_dir / f"{tables.tnx_diagnosis}.parquet")

        assert table_row_counts(tmp_path).rows() == [
            (tables.omop_condition_occurrence, 2)
        ]
        assert excluded_row_counts(tmp_path).rows() == [
            (tables.tnx_diagnosis, NO_STANDARD_MAPPING, 2),
            (tables.tnx_diagnosis, NO_SOURCE_CONCEPT, 1),
        ]

    def test_excluded_row_counts_without_files(self, tmp_path: Path) -> None:
        """No excluded folder gives an empty, typed report."""
        report = excluded_row_counts(tmp_path)

        assert report.is_empty()
        assert report.columns == [col.tnx_table, col.exclusion_reason, col.rows]

    def test_all_tables_present(self, tmp_path: Path) -> None:
        """Every missing table is listed in one issue."""
        _write_concepts(tmp_path)

        issues = check_all_tables_present(tmp_path)

        assert _error_types(issues) == ["TABLES_MISSING"]
        assert tables.omop_person in issues[0].message
        assert tables.omop_concept not in issues[0].message

    def test_cdm_source_row_count(self, tmp_path: Path) -> None:
        """CDM_SOURCE needs exactly one row; a missing file is skipped."""
        with pytest.raises(validation.CheckSkipped):
            check_cdm_source_row(tmp_path)
        _write_test_parquet(
            tmp_path, tables.omop_cdm_source, pl.DataFrame({"x": [1, 2]})
        )

        assert _error_types(check_cdm_source_row(tmp_path)) == ["ROW_COUNT"]

    def test_person_references_skipped_without_person(self, tmp_path: Path) -> None:
        with pytest.raises(validation.CheckSkipped):
            check_person_references(tmp_path)

    def test_coverage_consistency(self, tmp_path: Path) -> None:
        """Excluded files and table rows are compared with coverage.csv."""
        with pytest.raises(validation.CheckSkipped):
            check_coverage_consistency(tmp_path)
        _write_conditions(tmp_path, [_CONDITION, _CONDITION], [0, 0])
        excluded_dir = tmp_path / tables.excluded
        excluded_dir.mkdir()
        pl.DataFrame({col.exclusion_reason: [NO_SOURCE_CONCEPT]}).write_parquet(
            excluded_dir / f"{tables.tnx_diagnosis}.parquet"
        )
        rows = [
            (tables.tnx_diagnosis, "x", NO_SOURCE_CONCEPT, 1, 25.0, 1, 50.0),
            (
                tables.tnx_diagnosis,
                "x",
                tables.omop_condition_occurrence,
                3,
                75.0,
                1,
                50.0,
            ),
        ]
        pl.DataFrame(
            rows,
            schema=COVERAGE_SCHEMA,
            orient="row",
        ).write_csv(tmp_path / COVERAGE_FILE)

        issues = check_coverage_consistency(tmp_path)

        assert _error_types(issues) == ["COVERAGE_TABLE_MISMATCH"]
        assert "(3, 2)" in issues[0].message

    def test_coverage_report_text(self, tmp_path: Path) -> None:
        """The text report lists the row counts."""
        _write_conditions(tmp_path, [_CONDITION], [0])

        report = coverage_report(tmp_path)

        assert f"{tables.omop_condition_occurrence}: 1" in report
        assert "Excluded rows per TriNetX table and reason:" in report

    def test_run_check_reports_skip_and_error(self, tmp_path: Path) -> None:
        """A skipped check is neither passed nor failed; a crash is a failure."""
        skipped = validation.run_check(
            validation.Check("s", check_cdm_source_row), tmp_path
        )
        assert skipped.skipped is not None and not skipped.passed and not skipped.issues

        def crash(output_dir: Path) -> list[Issue]:
            raise ValueError("boom")

        crashed = validation.run_check(validation.Check("c", crash), tmp_path)
        assert [i.error_type for i in crashed.issues] == ["CHECK_ERROR"]
        assert "boom" in crashed.issues[0].message

    def test_validate_empty_folder_fails(self, tmp_path: Path) -> None:
        """Every schema check and the table presence check fail without output."""
        report = validation.validate(tmp_path)

        assert not report.ok
        assert len(report.failed) == len(OMOP_SCHEMAS) + 1
        assert report.passed == []
        assert len(report.skipped) == len(validation.CHECKS) - len(report.failed)

    def test_optional_vocabulary_tables_are_checked_only_when_present(
        self, tmp_path: Path
    ) -> None:
        names = {c.name for c in validation.checks_for(tmp_path)}
        assert not names & {f"{t} schema" for t in VOCABULARY_SCHEMAS}

        _write_test_parquet(
            tmp_path,
            tables.omop_domain,
            pl.DataFrame({col.domain_id: ["Condition", "Condition"]}),
        )
        checks = {c.name: c for c in validation.checks_for(tmp_path)}
        issues = validation.run_check(
            checks[f"{tables.omop_domain} schema"], tmp_path
        ).issues

        assert {i.error_type for i in issues} == {
            "MISSING_COLUMNS",
            "DUPLICATE_PRIMARY_KEY",
        }
        assert not validation.run_check(
            validation.OPTIONAL_CHECKS[tables.omop_vocabulary], tmp_path
        ).passed  # absent: skipped, not passed

    def test_era_tables_are_checked_only_when_present(self, tmp_path: Path) -> None:
        names = {c.name for c in validation.checks_for(tmp_path)}
        assert not names & ({f"{t} schema" for t in ERA_SCHEMAS} | {"era concepts"})

        _write_test_parquet(
            tmp_path, tables.omop_drug_era, pl.DataFrame({col.drug_era_id: [1, 1]})
        )
        checks = {c.name: c for c in validation.checks_for(tmp_path)}

        assert f"{tables.omop_drug_era} schema" in checks
        assert f"{tables.omop_condition_era} schema" not in checks
        assert "era concepts" in checks
        issues = validation.run_check(
            checks[f"{tables.omop_drug_era} schema"], tmp_path
        ).issues
        assert "DUPLICATE_PRIMARY_KEY" in {i.error_type for i in issues}

    def test_era_persons_are_checked_against_person(self, tmp_path: Path) -> None:
        _write_test_parquet(
            tmp_path,
            tables.omop_person,
            pl.DataFrame({col.person_id: [1]}),
        )
        _write_test_parquet(
            tmp_path,
            tables.omop_condition_era,
            pl.DataFrame({col.person_id: [1, 7]}),
        )

        errors = check_person_references(tmp_path)

        assert [(e.table, e.error_type) for e in errors] == [
            (tables.omop_condition_era, "INVALID_PERSON_ID")
        ]

    def test_era_concepts_must_exist_in_concept(self, tmp_path: Path) -> None:
        _write_concepts(tmp_path)
        _write_test_parquet(
            tmp_path,
            tables.omop_drug_era,
            pl.DataFrame({col.drug_concept_id: [_CONDITION, 999]}),
        )

        errors = check_concepts_exist(tmp_path)

        assert [(e.table, e.error_type) for e in errors] == [
            (tables.omop_drug_era, "MISSING_CONCEPT")
        ]
        assert "999" in errors[0].message

    def test_era_concepts_kind(self, tmp_path: Path) -> None:
        concepts = pl.DataFrame(
            [
                (_CONDITION, "Condition", "Clinical Finding"),
                (_OBSERVATION, "Observation", "Clinical Observation"),
                (1191, "Drug", "Ingredient"),
                (197361, "Drug", "Clinical Drug"),
            ],
            schema={
                col.concept_id: pl.Int64,
                col.domain_id: pl.Utf8,
                col.concept_class_id: pl.Utf8,
            },
            orient="row",
        )
        _write_test_parquet(tmp_path, tables.omop_concept, concepts)
        _write_test_parquet(
            tmp_path,
            tables.omop_condition_era,
            pl.DataFrame({col.condition_concept_id: [_CONDITION, _OBSERVATION]}),
        )
        _write_test_parquet(
            tmp_path,
            tables.omop_drug_era,
            pl.DataFrame({col.drug_concept_id: [1191, 197361, 0]}),
        )

        errors = check_era_concepts(tmp_path)

        assert [(e.table, e.error_type) for e in errors] == [
            (tables.omop_condition_era, "ERA_CONCEPT_MISMATCH"),
            (tables.omop_drug_era, "ERA_CONCEPT_MISMATCH"),
        ]
        assert str(_OBSERVATION) in errors[0].message
        assert "197361" in errors[1].message

    def test_era_concepts_check_skips_without_era_tables(self, tmp_path: Path) -> None:
        result = validation.run_check(validation.ERA_CONCEPTS_CHECK, tmp_path)

        assert result.skipped is not None

    def test_row_counts_include_era_tables(self, tmp_path: Path) -> None:
        _write_test_parquet(
            tmp_path, tables.omop_drug_era, pl.DataFrame({col.drug_era_id: [1, 2]})
        )

        counts = table_row_counts(tmp_path)

        assert counts.rows() == [(tables.omop_drug_era, 2)]

    def test_issue_str(self) -> None:
        assert str(Issue("PERSON", "X", "m")) == "[PERSON] X: m"
        assert str(Issue(None, "X", "m")) == "X: m"
