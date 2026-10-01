"""Tests for domain routing, OBSERVATION, DEVICE_EXPOSURE and cross-domain defaults."""

from datetime import date

import polars as pl
import pytest

from tnx_omop.util import columns as col
from tests.id_helpers import with_ids
from tnx_omop.util import tables as tbl
from tnx_omop.schema.domains import UNSUPPORTED_DOMAIN
from tnx_omop.schema.omop import OMOP_SCHEMAS
from tnx_omop.transformers import build_condition, build_procedure
from tnx_omop.transformers.clinical import clinical_events, transform_clinical
from tnx_omop.omop_vocab.vocabulary import CODES_SCHEMA, UNITS_SCHEMA, VocabularyLookup
from tests import athena_fixture as fx
from tests.schema_asserts import assert_conforms

DX = tbl.tnx_diagnosis
PX = tbl.tnx_procedure
LAB = tbl.tnx_lab_result


def records(code_system, codes, **columns):
    n = len(codes)
    data = {
        col._source_id: ["default"] * n,
        col.patient_id: ["P001"] * n,
        col.encounter_id: ["E001"] * n,
        col.date: [date(2023, 3, 1)] * n,
        col.code_system: [code_system] * n,
        col.code: codes,
    }
    data.update(columns)
    return with_ids(pl.DataFrame(data))


def run(df, table, make_lookup):
    return transform_clinical(df, table, make_lookup({table: df}))


def assert_tables_conform(result):
    for table, frame in result.items():
        if table != tbl.excluded:
            assert_conforms(frame, OMOP_SCHEMAS[table], numbered=False)


class TestRouting:
    def test_diagnosis_routes_by_domain(self, make_lookup):
        df = records("ICD-10-CM", ["I10", "Z87.891", "C81.90", "C79.51"])
        result = run(df, DX, make_lookup)

        assert set(result) == {
            tbl.omop_condition_occurrence,
            tbl.omop_observation,
            tbl.excluded,
        }
        assert_tables_conform(result)
        assert sorted(result[tbl.excluded][col.code]) == ["C79.51", "C81.90"]

    def test_empty_input_returns_only_excluded(self, make_lookup):
        df = records("ICD-10-CM", []).with_columns(pl.all().cast(pl.Utf8))
        result = run(df, DX, make_lookup)

        assert list(result) == [tbl.excluded]
        assert result[tbl.excluded].is_empty()

    def test_condition_plus_episode(self):
        rows = [
            (DX, "ICD-10-CM", "C81.90", 7, 70, "Condition", None),
            (DX, "ICD-10-CM", "C81.90", 7, None, "Episode", UNSUPPORTED_DOMAIN),
        ]
        lookup = VocabularyLookup(
            codes=pl.DataFrame(rows, schema=CODES_SCHEMA, orient="row"),
            units=pl.DataFrame(schema=UNITS_SCHEMA),
        )
        result = transform_clinical(records("ICD-10-CM", ["C81.90"]), DX, lookup)

        conditions = result[tbl.omop_condition_occurrence]
        assert conditions[col.condition_concept_id].to_list() == [70]
        assert result[tbl.excluded].select(
            col.exclusion_reason, col.target_domain_id
        ).rows() == [(UNSUPPORTED_DOMAIN, "Episode")]

    def test_condition_plus_episode_from_athena(self, make_lookup):
        # One record per target concept: the Condition target is kept and the
        # Episode target is excluded, so one TriNetX row appears in both outputs.
        result = run(records("ICD-10-CM", ["C91.00"]), DX, make_lookup)

        assert set(result) == {tbl.omop_condition_occurrence, tbl.excluded}
        assert_tables_conform(result)
        conditions = result[tbl.omop_condition_occurrence]
        assert conditions.select(
            col.condition_concept_id, col.condition_source_concept_id
        ).rows() == [(fx.COND_ALL, fx.ICD10_C91_00)]
        assert result[tbl.excluded].select(
            col.code, col.exclusion_reason, col.target_domain_id, col.source_concept_id
        ).rows() == [("C91.00", UNSUPPORTED_DOMAIN, "Episode", fx.ICD10_C91_00)]


class TestObservation:
    def test_diagnosis_to_observation(self, make_lookup):
        df = records("ICD-10-CM", ["Z87.891"])
        row = run(df, DX, make_lookup)[tbl.omop_observation].row(0, named=True)

        assert row[col.observation_concept_id] == fx.OBS_EX_SMOKER
        assert row[col.observation_source_concept_id] == fx.ICD10_Z87_891
        assert row[col.observation_source_value] == "ICD-10-CM:Z87.891"
        assert row[col.observation_date] == date(2023, 3, 1)
        assert row[col.observation_type_concept_id] == fx.EHR_TYPE
        assert row[col.value_as_number] is None
        assert row[col.value_as_string] is None
        assert row[col.value_as_concept_id] == 0
        assert row[col.unit_concept_id] == 0
        assert row[col.qualifier_concept_id] == 0
        assert row[col.obs_event_field_concept_id] == 0

    def test_lab_to_observation(self, make_lookup):
        df = records(
            "LOINC",
            ["72166-2"],
            **{
                col.lab_result_num_val: [2.0],
                col.lab_result_text_val: ["Positive"],
                col.units_of_measure: ["U/L"],
            },
        )
        result = run(df, LAB, make_lookup)
        assert set(result) == {tbl.omop_observation, tbl.excluded}
        assert_tables_conform(result)
        row = result[tbl.omop_observation].row(0, named=True)

        assert row[col.observation_concept_id] == fx.LOINC_72166_2
        assert row[col.observation_source_value] == "72166-2"
        assert row[col.value_as_number] == 2.0
        assert row[col.value_as_string] == "Positive"
        assert row[col.value_source_value] == "Positive"
        assert row[col.value_as_concept_id] == 9191
        assert row[col.unit_concept_id] == fx.UCUM_U_L
        assert row[col.unit_source_value] == "U/L"


class TestProcedureSources:
    @pytest.fixture
    def result(self, make_lookup):
        df = pl.concat(
            [
                records("HCPCS", ["J1100", "E0601"], **{col.modifier_1: ["JW", None]}),
                records("CPT", ["80053"], **{col.modifier_1: [None]}),
            ]
        )
        return run(df, PX, make_lookup)

    def test_routes(self, result):
        assert set(result) == {
            tbl.omop_drug_exposure,
            tbl.omop_device_exposure,
            tbl.omop_measurement,
            tbl.excluded,
        }
        assert_tables_conform(result)
        assert result[tbl.excluded].is_empty()

    def test_procedure_to_drug(self, result):
        row = result[tbl.omop_drug_exposure].row(0, named=True)

        assert row[col.drug_concept_id] == fx.DRUG_DEXAMETHASONE
        assert row[col.drug_source_concept_id] == fx.HCPCS_J1100
        assert row[col.drug_source_value] == "HCPCS:J1100"
        assert row[col.drug_exposure_start_date] == date(2023, 3, 1)
        assert row[col.drug_exposure_end_date] == date(2023, 3, 1)
        assert row[col.route_concept_id] == 0
        assert row[col.route_source_value] is None
        assert row[col.quantity] is None

    def test_procedure_to_device(self, result):
        row = result[tbl.omop_device_exposure].row(0, named=True)

        assert row[col.device_concept_id] == fx.DEVICE_CPAP
        assert row[col.device_source_concept_id] == fx.HCPCS_E0601
        assert row[col.device_source_value] == "HCPCS:E0601"
        assert row[col.device_exposure_start_date] == date(2023, 3, 1)
        assert row[col.device_exposure_end_date] is None
        assert row[col.device_type_concept_id] == fx.EHR_TYPE
        assert row[col.unit_concept_id] == 0
        assert row[col.unit_source_concept_id] == 0
        assert row[col.quantity] is None

    def test_procedure_to_measurement(self, result):
        row = result[tbl.omop_measurement].row(0, named=True)

        assert row[col.measurement_concept_id] == fx.CPT4_80053
        assert row[col.measurement_source_value] == "CPT:80053"
        assert row[col.value_as_number] is None
        assert row[col.value_source_value] is None
        assert row[col.value_as_concept_id] == 0
        assert row[col.unit_concept_id] == 0
        assert row[col.operator_concept_id] == 0
        assert row[col.meas_event_field_concept_id] == 0


class TestBuilderDefaults:
    def test_condition_from_procedure_events(self, make_lookup):
        df = records("ICD-9-CM", ["38.93"], **{col.modifier_1: ["51"]})
        events, _ = clinical_events(df, PX, make_lookup({PX: df}))
        row = build_condition(events).row(0, named=True)

        assert row[col.condition_status_concept_id] == 0
        assert row[col.condition_status_source_value] is None
        assert row[col.condition_end_date] is None

    def test_procedure_from_diagnosis_events(self, make_lookup):
        df = records("ICD-10-CM", ["I10"], **{col.principal_diagnosis_indicator: ["P"]})
        events, _ = clinical_events(df, DX, make_lookup({DX: df}))
        row = build_procedure(events).row(0, named=True)

        assert row[col.modifier_concept_id] == 0
        assert row[col.modifier_source_value] is None
        assert row[col.quantity] is None
        assert row[col.procedure_end_date] is None
