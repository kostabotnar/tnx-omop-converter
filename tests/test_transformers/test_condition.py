"""Tests for CONDITION_OCCURRENCE built through transform_clinical."""

from datetime import date

import polars as pl
import pytest

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.util.concept_mappings import (
    CONDITION_STATUS_CONCEPT_MAP,
    DEFAULT_CONCEPT_ID,
)
from tnx_omop.schema.omop import CONDITION_OCCURRENCE_SCHEMA
from tnx_omop.transformers.clinical import transform_clinical
from tests import athena_fixture as fx
from tests.schema_asserts import assert_conforms

DX = tbl.tnx_diagnosis
CODES = ["I10", "E78.5", "K21.0"]


@pytest.fixture
def conditions(make_lookup):
    def run(df: pl.DataFrame) -> pl.DataFrame:
        return transform_clinical(df, DX, make_lookup({DX: df}))[
            tbl.omop_condition_occurrence
        ]

    return run


def diagnosis(n=3, **columns):
    data = {
        col._source_id: ["default"] * n,
        col.patient_id: ["P001"] * n,
        col.person_id: [1] * n,
        col.encounter_id: ["E001"] * n,
        col.visit_occurrence_id: [1] * n,
        col.date: [date(2023, 1, 1)] * n,
        col.code: CODES[:n],
        col.code_system: ["ICD-10-CM"] * n,
    }
    data.update(columns)
    return pl.DataFrame(data)


class TestBuildCondition:
    def test_transforms_basic_diagnosis(self, conditions, sample_diagnosis_df):
        result = conditions(sample_diagnosis_df)

        assert len(result) == 3
        assert_conforms(result, CONDITION_OCCURRENCE_SCHEMA, numbered=False)

    def test_maps_concepts_and_source_values(self, conditions):
        result = conditions(diagnosis(1, **{col.date: [date(2023, 2, 15)]}))
        row = result.row(0, named=True)

        assert row[col.condition_concept_id] == fx.COND_HYPERTENSION
        assert row[col.condition_source_concept_id] == fx.ICD10_I10
        assert row[col.condition_source_value] == "ICD-10-CM:I10"
        assert row[col.condition_start_date] == date(2023, 2, 15)
        assert row[col.condition_end_date] is None
        assert row[col.condition_type_concept_id] == fx.EHR_TYPE

    def test_one_to_many(self, conditions):
        result = conditions(diagnosis(1, **{col.code: ["E11.65"]}))

        assert sorted(result[col.condition_concept_id]) == [
            fx.COND_T2DM,
            fx.COND_HYPERGLYCEMIA,
        ]
        assert result.height == 2

    def test_maps_condition_status_concept_ids(self, conditions):
        result = conditions(
            diagnosis(**{col.principal_diagnosis_indicator: ["P", "S", "Unknown"]})
        )

        status_map = dict(
            result.select(
                col.condition_status_source_value, col.condition_status_concept_id
            ).iter_rows()
        )
        assert status_map["P"] == CONDITION_STATUS_CONCEPT_MAP["P"] == 32902
        assert status_map["S"] == CONDITION_STATUS_CONCEPT_MAP["S"] == 32908
        assert status_map["Unknown"] == DEFAULT_CONCEPT_ID

    def test_unmapped_indicator_defaults_to_zero(self, conditions):
        result = conditions(
            diagnosis(
                **{
                    col.encounter_id: ["E001", "E002", "E003"],
                    col.principal_diagnosis_indicator: ["Unknown", "Y", None],
                }
            )
        )

        assert result[col.condition_status_concept_id].to_list() == [0, 0, 0]

    def test_handles_missing_indicator_column(self, conditions):
        result = conditions(diagnosis(1))

        assert len(result) == 1
        assert result[col.condition_status_concept_id][0] == DEFAULT_CONCEPT_ID
        assert result[col.condition_status_source_value][0] is None

    def test_keeps_derived_records(self, conditions):
        result = conditions(diagnosis(2, **{col.derived_by_TriNetX: [None, "T"]}))

        assert len(result) == 2
        assert col.derived_by_TriNetX not in result.columns

    def test_leaves_record_id_for_the_merge(self, conditions):
        result = conditions(diagnosis(1))

        assert result[col.condition_occurrence_id].to_list() == [None]

    def test_keeps_duplicate_rows(self, conditions):
        df = diagnosis(1)
        assert len(conditions(pl.concat([df, df]))) == 2
