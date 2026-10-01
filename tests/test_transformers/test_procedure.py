"""Tests for PROCEDURE_OCCURRENCE built through transform_clinical."""

from datetime import date

import polars as pl

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.schema.omop import PROCEDURE_OCCURRENCE_SCHEMA
from tnx_omop.transformers.clinical import transform_clinical
from tests import athena_fixture as fx
from tests.schema_asserts import assert_conforms

PX = tbl.tnx_procedure


def procedures(df: pl.DataFrame, make_lookup) -> pl.DataFrame:
    return transform_clinical(df, PX, make_lookup({PX: df}))[
        tbl.omop_procedure_occurrence
    ]


class TestBuildProcedure:
    def test_transforms_basic_procedure(self, make_lookup, sample_procedure_df):
        result = procedures(sample_procedure_df, make_lookup)

        assert_conforms(result, PROCEDURE_OCCURRENCE_SCHEMA, numbered=False)
        rows = {r[col.procedure_source_value]: r for r in result.iter_rows(named=True)}
        assert set(rows) == {"CPT:99213", "CPT:99214"}
        first = rows["CPT:99213"]
        assert first[col.procedure_concept_id] == fx.CPT4_99213
        assert first[col.procedure_source_concept_id] == fx.CPT4_99213
        assert first[col.procedure_date] == date(2023, 1, 1)
        assert first[col.procedure_end_date] is None
        assert first[col.modifier_concept_id] == 0
        assert first[col.modifier_source_value] == "25"
        assert first[col.quantity] is None
        assert rows["CPT:99214"][col.modifier_source_value] is None

    def test_icd9_uses_procedure_vocabulary(self, make_lookup):
        df = pl.DataFrame(
            {
                col._source_id: ["default"],
                col.patient_id: ["P001"],
                col.person_id: [1],
                col.encounter_id: ["E001"],
                col.visit_occurrence_id: [1],
                col.date: [date(2023, 1, 1)],
                col.code_system: ["ICD-9-CM"],
                col.code: ["38.93"],
            }
        )
        result = procedures(df, make_lookup)

        assert result[col.procedure_concept_id].to_list() == [fx.PROC_ICD9_TARGET]
        assert result[col.procedure_source_concept_id].to_list() == [fx.ICD9PROC_38_93]
