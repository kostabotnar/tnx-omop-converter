"""Tests for tnx_omop/omop_vocab/concept_table.py."""

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import pytest

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.omop_vocab.concept_table import (
    create_concept_table,
    load_terminology,
    referenced_concept_ids,
    table_path,
)
from tnx_omop.omop_schema import CONCEPT_SCHEMA
from tests import athena_fixture as fx
from tests.schema_asserts import assert_conforms

MISSING_ID = 987654321


def _write_table(output_dir: Path, table_name: str, data: Dict[str, List]) -> None:
    path = table_path(output_dir, table_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_parquet(path)


def _terminology(description: Optional[str] = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            col.code_system: ["CPT", "CPT", "HCPCS"],
            col.code: ["80053", "99214", "80053"],
            col.code_description: [
                description or "Comprehensive metabolic panel",
                "TriNetX name that must not replace the Athena name",
                "Same code in another code system",
            ],
        }
    )


def _write_cpt4_procedures(output_dir: Path) -> None:
    _write_table(
        output_dir,
        tbl.omop_procedure_occurrence,
        {
            col.procedure_concept_id: [fx.CPT4_80053, fx.CPT4_99213, fx.CPT4_99214],
            col.procedure_source_value: ["CPT:80053", "CPT:99213", "CPT:99214"],
        },
    )


def _names(concepts: pl.DataFrame) -> Dict[int, str]:
    return dict(concepts.select(col.concept_id, col.concept_name).iter_rows())


class TestReferencedConceptIds:
    def test_collects_every_concept_id_column(self, tmp_path: Path):
        _write_table(
            tmp_path,
            tbl.omop_person,
            {
                col.person_id: [1, 2],
                col.gender_concept_id: [8507, 8532],
                col.race_concept_id: [8527, 0],
                col.ethnicity_concept_id: [38003564, 0],
                col.gender_source_concept_id: [0, 0],
            },
        )
        _write_table(
            tmp_path,
            tbl.omop_visit_occurrence,
            {col.visit_concept_id: [9201], col.visit_type_concept_id: [fx.EHR_TYPE]},
        )
        _write_table(
            tmp_path,
            tbl.omop_measurement,
            {
                col.measurement_concept_id: [fx.LOINC_2160_0],
                col.measurement_source_concept_id: [fx.LOINC_2160_0],
                col.unit_concept_id: [fx.UCUM_MG_DL],
                col.value_as_concept_id: [9191],
                col.value_as_number: [1.0],
            },
        )
        _write_table(
            tmp_path,
            tbl.omop_drug_exposure,
            {col.route_concept_id: [4132161], col.drug_source_concept_id: [None]},
        )
        _write_table(
            tmp_path,
            tbl.omop_condition_occurrence,
            {col.condition_status_concept_id: [32902]},
        )
        _write_table(
            tmp_path, tbl.omop_cdm_source, {col.cdm_version_concept_id: [756265]}
        )
        # CONCEPT itself is not a source of references.
        _write_table(tmp_path, tbl.omop_concept, {col.concept_id: [MISSING_ID]})

        ids = referenced_concept_ids(tmp_path)

        assert ids.to_list() == sorted(
            [
                8507,
                8532,
                8527,
                38003564,
                9201,
                fx.EHR_TYPE,
                fx.LOINC_2160_0,
                fx.UCUM_MG_DL,
                9191,
                4132161,
                32902,
                756265,
            ]
        )

    def test_era_tables_are_included(self, tmp_path: Path):
        _write_table(
            tmp_path, tbl.omop_condition_era, {col.condition_concept_id: [7, 0]}
        )
        _write_table(
            tmp_path, tbl.omop_drug_era, {col.drug_concept_id: [fx.RXNORM_1191]}
        )

        assert referenced_concept_ids(tmp_path).to_list() == sorted([7, fx.RXNORM_1191])

    def test_no_tables(self, tmp_path: Path):
        ids = referenced_concept_ids(tmp_path)
        assert ids.is_empty()
        assert ids.dtype == pl.Int64


class TestCreateConceptTable:
    def test_rows_are_referenced_athena_concepts(self, tmp_path: Path, athena_dir):
        _write_table(
            tmp_path,
            tbl.omop_condition_occurrence,
            {
                col.condition_concept_id: [fx.COND_T2DM, fx.COND_T2DM],
                col.condition_source_concept_id: [fx.ICD10_E11_65, 0],
                col.condition_type_concept_id: [fx.EHR_TYPE, fx.EHR_TYPE],
            },
        )

        result = create_concept_table(tmp_path, athena_dir, _terminology())

        assert_conforms(result, CONCEPT_SCHEMA)
        assert result.get_column(col.concept_id).to_list() == sorted(
            [fx.COND_T2DM, fx.ICD10_E11_65, fx.EHR_TYPE]
        )
        row = result.filter(pl.col(col.concept_id) == fx.COND_T2DM).row(0, named=True)
        assert row[col.concept_name] == "Type 2 diabetes mellitus"
        assert row[col.domain_id] == "Condition"
        assert row[col.vocabulary_id] == "SNOMED"
        assert row[col.standard_concept] == "S"
        assert row[col.concept_code] == "44054006"
        assert row[col.valid_start_date] == date(1970, 1, 1)
        assert row[col.invalid_reason] is None
        written = pl.read_parquet(table_path(tmp_path, tbl.omop_concept))
        assert written.equals(result)

    def test_cpt4_names_from_terminology(self, tmp_path: Path, athena_dir):
        _write_cpt4_procedures(tmp_path)

        result = create_concept_table(tmp_path, athena_dir, _terminology())

        assert result.height == 3
        assert _names(result) == {
            fx.CPT4_80053: "Comprehensive metabolic panel",
            fx.CPT4_99213: "99213",  # no TriNetX description: concept code
            fx.CPT4_99214: "Office visit, established patient",  # Athena name kept
        }

    def test_without_terminology_empty_names_are_codes(
        self, tmp_path: Path, athena_dir
    ):
        _write_cpt4_procedures(tmp_path)

        result = create_concept_table(tmp_path, athena_dir, None)

        assert _names(result)[fx.CPT4_80053] == "80053"
        assert result.get_column(col.concept_name).null_count() == 0

    def test_long_names_are_truncated(self, tmp_path: Path, athena_dir):
        _write_cpt4_procedures(tmp_path)

        result = create_concept_table(tmp_path, athena_dir, _terminology("x" * 300))

        assert _names(result)[fx.CPT4_80053] == "x" * 255

    def test_missing_ids_warn_and_are_left_out(
        self, tmp_path: Path, athena_dir, caplog: pytest.LogCaptureFixture
    ):
        _write_table(
            tmp_path,
            tbl.omop_observation,
            {col.observation_concept_id: [fx.OBS_EX_SMOKER, MISSING_ID]},
        )

        with caplog.at_level(
            logging.WARNING, logger="tnx_omop.omop_vocab.concept_table"
        ):
            result = create_concept_table(tmp_path, athena_dir, None)

        assert result.get_column(col.concept_id).to_list() == [fx.OBS_EX_SMOKER]
        assert str(MISSING_ID) in caplog.text

    def test_no_references_writes_empty_table(self, tmp_path: Path, athena_dir):
        result = create_concept_table(tmp_path, athena_dir, None)

        assert result.is_empty()
        assert_conforms(result, CONCEPT_SCHEMA)
        assert table_path(tmp_path, tbl.omop_concept).exists()


class TestLoadTerminology:
    """Tests for load_terminology."""

    def test_merges_sources_and_keeps_codes_as_strings(self, tmp_path):
        for name, rows in [
            ("a", "RxNorm,0012,Drug A,,\nLOINC,2885-2,Cholesterol,,mg/dL\n"),
            ("b", "RxNorm,0012,Drug A,,\nCPT,99213,Office visit,,\n"),
        ]:
            source = tmp_path / name
            source.mkdir()
            (source / f"{tbl.tnx_standardized_terminology}.csv").write_text(
                "code_system,code,code_description,path,unit\n" + rows
            )

        result = load_terminology([tmp_path / "a", tmp_path / "b", tmp_path / "c"])

        assert result is not None
        assert len(result) == 3
        assert "0012" in result[col.code].to_list()

    def test_returns_none_when_file_missing(self, tmp_path):
        assert load_terminology([tmp_path]) is None
