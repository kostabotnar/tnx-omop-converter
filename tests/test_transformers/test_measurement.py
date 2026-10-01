"""Tests for MEASUREMENT built through transform_clinical."""

from datetime import date

import polars as pl

from tnx_omop.util import columns as col
from tests.id_helpers import with_ids
from tnx_omop.util import tables as tbl
from tnx_omop.omop_schema import MEASUREMENT_SCHEMA
from tnx_omop.transformers.clinical import transform_clinical
from tests import athena_fixture as fx
from tests.schema_asserts import assert_conforms

LAB = tbl.tnx_lab_result
VITALS = tbl.tnx_vitals_signs


def measurements(df: pl.DataFrame, table: str, make_lookup) -> pl.DataFrame:
    return transform_clinical(df, table, make_lookup({table: df}))[tbl.omop_measurement]


class TestBuildMeasurement:
    def test_lab_results(self, make_lookup, sample_lab_result_df):
        result = measurements(sample_lab_result_df, LAB, make_lookup)

        assert_conforms(result, MEASUREMENT_SCHEMA, numbered=False)
        rows = {
            r[col.measurement_source_value]: r for r in result.iter_rows(named=True)
        }
        creatinine = rows["2160-0"]
        assert creatinine[col.measurement_concept_id] == fx.LOINC_2160_0
        assert creatinine[col.measurement_source_concept_id] == fx.LOINC_2160_0
        assert creatinine[col.measurement_date] == date(2023, 1, 1)
        assert creatinine[col.value_as_number] == 1.2
        assert creatinine[col.unit_concept_id] == fx.UCUM_MG_DL
        assert creatinine[col.unit_source_value] == "mg/dL"
        assert creatinine[col.value_as_concept_id] == 0
        assert creatinine[col.operator_concept_id] == 0
        assert creatinine[col.meas_event_field_concept_id] == 0
        assert rows["2345-7"][col.value_source_value] == "Normal"

    def test_text_values(self, make_lookup):
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default"] * 3,
                    col.patient_id: ["P001"] * 3,
                    col.encounter_id: ["E001"] * 3,
                    col.date: [date(2023, 1, 1)] * 3,
                    col.code_system: ["LOINC"] * 3,
                    col.code: ["2345-7"] * 3,
                    col.lab_result_num_val: [None, None, None],
                    col.lab_result_text_val: ["Negative", "Positive", "Unknown"],
                    col.units_of_measure: [None, None, None],
                }
            )
        )
        result = measurements(df, LAB, make_lookup)

        values = dict(
            result.select(col.value_source_value, col.value_as_concept_id).iter_rows()
        )
        assert values == {"Negative": 9189, "Positive": 9191, "Unknown": 0}

    def test_vitals(self, make_lookup, sample_vitals_df):
        result = measurements(sample_vitals_df, VITALS, make_lookup)

        assert_conforms(result, MEASUREMENT_SCHEMA, numbered=False)
        values = dict(
            result.select(col.measurement_concept_id, col.value_as_number).iter_rows()
        )
        assert values == {fx.LOINC_8480_6: 120.0, fx.LOINC_8462_4: 80.0}

    def test_same_row_in_lab_and_vitals_gives_two_records(self, make_lookup):
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default"],
                    col.patient_id: ["P001"],
                    col.encounter_id: ["E001"],
                    col.date: [date(2023, 1, 1)],
                    col.code_system: ["LOINC"],
                    col.code: ["8480-6"],
                    col.units_of_measure: ["mmHg"],
                }
            )
        )
        lookup = make_lookup({LAB: df, VITALS: df})
        lab = transform_clinical(df, LAB, lookup)[tbl.omop_measurement]
        vitals = transform_clinical(df, VITALS, lookup)[tbl.omop_measurement]

        assert lab.height == vitals.height == 1
