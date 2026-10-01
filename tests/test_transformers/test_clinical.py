"""Tests for tnx_omop/transformers/clinical.py and tnx_omop/transformers/sources.py."""

from datetime import date

import polars as pl
import pytest

from tnx_omop.util import columns as col
from tests.id_helpers import with_ids
from tnx_omop.util import tables as tbl
from tnx_omop.util.concept_mappings import (
    CONDITION_STATUS_CONCEPT_MAP,
    ROUTE_CONCEPT_MAP,
)
from tnx_omop.domains import NO_SOURCE_CONCEPT, NO_STANDARD_MAPPING, UNSUPPORTED_DOMAIN
from tnx_omop.transformers.clinical import apply_lookup, clinical_events
from tnx_omop.transformers.sources import EVENT_SCHEMA
from tnx_omop.omop_vocab.vocabulary import CODES_SCHEMA
from tests import athena_fixture as fx

DX = tbl.tnx_diagnosis


def diagnosis(codes, **extra):
    n = len(codes)
    data = {
        col._source_id: ["default"] * n,
        col.patient_id: ["P001"] * n,
        col.person_id: [1] * n,
        col.encounter_id: ["E001"] * n,
        col.visit_occurrence_id: [1] * n,
        col.date: [date(2023, 1, 1)] * n,
        col.code_system: ["ICD-10-CM"] * n,
        col.code: codes,
    }
    data.update(extra)
    return pl.DataFrame(data)


@pytest.fixture
def codes() -> pl.DataFrame:
    """Hand-made lookup rows of the diagnosis table (without _tnx_table)."""
    rows = [
        ("ICD-10-CM", "E11.65", 1, 10, "Condition", None),
        ("ICD-10-CM", "E11.65", 1, 11, "Condition", None),
        ("ICD-10-CM", "C81.90", 2, 20, "Condition", None),
        ("ICD-10-CM", "C81.90", 2, None, "Episode", UNSUPPORTED_DOMAIN),
        ("ICD-10-CM", "C79.51", 3, None, None, NO_STANDARD_MAPPING),
    ]
    schema = {k: v for k, v in CODES_SCHEMA.items() if k != col._tnx_table}
    return pl.DataFrame(rows, schema=schema, orient="row")


class TestApplyLookup:
    def test_splits_mapped_and_excluded(self, codes):
        df = diagnosis(["E11.65", "C81.90", "C79.51", "UNKNOWN"])
        mapped, excluded = apply_lookup(df, codes, DX)

        assert sorted(mapped.get_column(col._concept_id)) == [10, 11, 20]
        reasons = dict(excluded.select(col.code, col.exclusion_reason).iter_rows())
        assert reasons == {
            "C81.90": UNSUPPORTED_DOMAIN,
            "C79.51": NO_STANDARD_MAPPING,
            "UNKNOWN": NO_SOURCE_CONCEPT,
        }

    def test_condition_plus_episode(self, codes):
        mapped, excluded = apply_lookup(diagnosis(["C81.90"]), codes, DX)

        assert mapped.select(col._concept_id, col._domain_id).rows() == [
            (20, "Condition")
        ]
        assert excluded.select(
            col.exclusion_reason, col.target_domain_id, col.source_concept_id
        ).rows() == [(UNSUPPORTED_DOMAIN, "Episode", 2)]

    def test_excluded_columns(self, codes):
        df = diagnosis(["UNKNOWN"])
        _, excluded = apply_lookup(df, codes, DX)

        assert excluded.columns == df.columns + [
            col.exclusion_reason,
            col.target_domain_id,
            col.source_concept_id,
        ]
        assert excluded.row(0)[-3:] == (NO_SOURCE_CONCEPT, None, 0)

    def test_mapped_columns(self, codes):
        df = diagnosis(["E11.65"])
        mapped, _ = apply_lookup(df, codes, DX)

        assert mapped.columns == df.columns + [
            col._source_concept_id,
            col._concept_id,
            col._domain_id,
        ]

    def test_one_row_per_target_and_duplicates_kept(self, codes):
        df = diagnosis(["E11.65", "E11.65", "C81.90"])
        mapped, _ = apply_lookup(df, codes, DX)

        assert mapped.height == 5

    def test_empty_input(self, codes):
        mapped, excluded = apply_lookup(diagnosis([]), codes, DX)
        assert mapped.is_empty()
        assert excluded.is_empty()


class TestSourceAdapters:
    def test_diagnosis(self, make_lookup):
        df = diagnosis(
            ["I10", "E78.5"], **{col.principal_diagnosis_indicator: ["P", None]}
        )
        events, excluded = clinical_events(df, DX, make_lookup({DX: df}))

        assert events.drop(col._domain_id).schema == EVENT_SCHEMA
        assert excluded.is_empty()
        rows = {r[col._source_value]: r for r in events.iter_rows(named=True)}
        assert rows["ICD-10-CM:I10"][col._concept_id] == fx.COND_HYPERTENSION
        assert rows["ICD-10-CM:I10"][col._source_concept_id] == fx.ICD10_I10
        assert (
            rows["ICD-10-CM:I10"][col._status_concept_id]
            == (CONDITION_STATUS_CONCEPT_MAP["P"])
        )
        assert rows["ICD-10-CM:I10"][col._status_source_value] == "P"
        assert rows["ICD-10-CM:E78.5"][col._status_concept_id] == 0
        assert rows["ICD-10-CM:I10"][col._start_date] == date(2023, 1, 1)
        assert rows["ICD-10-CM:I10"][col._end_date] is None
        assert rows["ICD-10-CM:I10"][col._route_concept_id] == 0
        assert rows["ICD-10-CM:I10"][col.person_id] is not None
        assert rows["ICD-10-CM:I10"][col.visit_occurrence_id] is not None

    def test_null_visit_id_is_kept(self, make_lookup):
        df = diagnosis(["I10"], **{col.visit_occurrence_id: [None]}).with_columns(
            pl.col(col.visit_occurrence_id).cast(pl.Int64)
        )
        events, _ = clinical_events(df, DX, make_lookup({DX: df}))
        assert events[col.visit_occurrence_id][0] is None
        assert events[col.person_id][0] == 1

    def test_procedure(self, make_lookup, sample_procedure_df):
        table = tbl.tnx_procedure
        lookup = make_lookup({table: sample_procedure_df})
        events, _ = clinical_events(sample_procedure_df, table, lookup)

        assert events.drop(col._domain_id).schema == EVENT_SCHEMA
        modifiers = dict(
            events.select(col._source_value, col._modifier_source_value).iter_rows()
        )
        assert modifiers == {"CPT:99213": "25", "CPT:99214": None}

    def test_medication(self, make_lookup):
        table = tbl.tnx_medication_ingredient
        df = with_ids(
            with_ids(
                pl.DataFrame(
                    {
                        col._source_id: ["default"] * 2,
                        col.patient_id: ["P001"] * 2,
                        col.encounter_id: ["E001"] * 2,
                        col.unique_id: ["M001", "M002"],
                        col.code_system: ["RxNorm"] * 2,
                        col.code: ["197361", "312961"],
                        col.start_date: [date(2023, 1, 1), date(2023, 1, 5)],
                        col.end_date: [date(2023, 1, 7), None],
                        col.route: ["Oral Product", "Unknown"],
                    }
                )
            )
        )
        events, _ = clinical_events(df, table, make_lookup({table: df}))

        rows = {r[col._source_value]: r for r in events.iter_rows(named=True)}
        assert rows["197361"][col._concept_id] == fx.RXNORM_197361
        assert rows["197361"][col._end_date] == date(2023, 1, 7)
        assert rows["312961"][col._end_date] == date(2023, 1, 5)
        assert (
            rows["197361"][col._route_concept_id] == (ROUTE_CONCEPT_MAP["Oral Product"])
        )
        assert rows["312961"][col._route_concept_id] == 0
        assert rows["312961"][col._route_source_value] == "Unknown"

    def test_lab(self, make_lookup):
        table = tbl.tnx_lab_result
        df = with_ids(
            with_ids(
                pl.DataFrame(
                    {
                        col._source_id: ["default"] * 4,
                        col.patient_id: ["P001"] * 4,
                        col.encounter_id: ["E001"] * 4,
                        col.date: [date(2023, 1, 1)] * 4,
                        col.code_system: ["LOINC", "LOINC", "LOINC", "TNX"],
                        col.code: ["2160-0", "2345-7", "2345-7", "9037-0"],
                        col.lab_result_num_val: [1.2, None, None, 3.0],
                        col.lab_result_text_val: ["", "Negative", "Unknown", None],
                        col.units_of_measure: ["mg/dL", "U/L", "{INR}", None],
                    }
                )
            )
        )
        events, excluded = clinical_events(df, table, make_lookup({table: df}))

        assert events.drop(col._domain_id).schema == EVENT_SCHEMA
        assert excluded.get_column(col.code).to_list() == ["9037-0"]
        rows = events.sort(col._value_source_value, nulls_last=False).rows(named=True)
        numeric, negative, unknown = rows
        assert numeric[col._source_value] == "2160-0"
        assert numeric[col._value_as_number] == 1.2
        assert numeric[col._unit_concept_id] == fx.UCUM_MG_DL
        assert numeric[col._unit_source_value] == "mg/dL"
        assert numeric[col._value_as_concept_id] == 0
        assert numeric[col._value_source_value] is None
        assert negative[col._value_as_concept_id] == 9189
        assert negative[col._unit_concept_id] == fx.UCUM_U_L
        assert unknown[col._value_as_concept_id] == 0
        assert unknown[col._unit_concept_id] == 0
        assert unknown[col._value_source_value] == "Unknown"

    def test_vitals(self, make_lookup, sample_vitals_df):
        table = tbl.tnx_vitals_signs
        lookup = make_lookup({table: sample_vitals_df})
        events, _ = clinical_events(sample_vitals_df, table, lookup)

        assert sorted(events.get_column(col._value_as_number)) == [80.0, 120.0]
        assert set(events.get_column(col._concept_id)) == {
            fx.LOINC_8480_6,
            fx.LOINC_8462_4,
        }
        # mmHg has no UCUM override, so it stays unmapped.
        assert set(events.get_column(col._unit_concept_id)) == {0}
