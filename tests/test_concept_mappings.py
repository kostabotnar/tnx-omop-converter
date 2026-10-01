"""Tests for concept mapping config and domain routing constants."""

import polars as pl
import pytest

from tnx_omop.util import tables as tbl
from tnx_omop.util.concept_mappings import (
    DEFAULT_CONCEPT_ID,
    GENDER_CONCEPT_MAP,
    LAB_VALUE_CONCEPT_MAP,
    SOURCE_VOCABULARY_MAP,
    UNIT_UCUM_MAP,
    map_concept_id,
)
from tnx_omop.domains import DOMAIN_TABLES, EVENT_CONCEPT_COLUMNS

CLINICAL_TABLES = [
    tbl.tnx_diagnosis,
    tbl.tnx_procedure,
    tbl.tnx_medication_ingredient,
    tbl.tnx_lab_result,
    tbl.tnx_vitals_signs,
]


class TestSourceVocabularies:
    @pytest.mark.parametrize("tnx_table", CLINICAL_TABLES)
    def test_every_clinical_table_has_code_system_map(self, tnx_table: str):
        assert SOURCE_VOCABULARY_MAP.get(tnx_table)

    def test_only_known_tables(self):
        assert set(SOURCE_VOCABULARY_MAP) == set(CLINICAL_TABLES)

    def test_values_are_strings_or_null(self):
        for systems in SOURCE_VOCABULARY_MAP.values():
            for vocabulary_id in systems.values():
                assert vocabulary_id is None or isinstance(vocabulary_id, str)

    def test_icd9_depends_on_table(self):
        assert SOURCE_VOCABULARY_MAP[tbl.tnx_diagnosis]["ICD-9-CM"] == "ICD9CM"
        assert SOURCE_VOCABULARY_MAP[tbl.tnx_procedure]["ICD-9-CM"] == "ICD9Proc"

    def test_tnx_has_no_vocabulary(self):
        assert SOURCE_VOCABULARY_MAP[tbl.tnx_lab_result]["TNX"] is None


class TestValueMaps:
    def test_unit_overrides_change_the_unit(self):
        assert UNIT_UCUM_MAP
        for unit, ucum in UNIT_UCUM_MAP.items():
            assert unit != ucum

    def test_lab_values(self):
        assert LAB_VALUE_CONCEPT_MAP == {"Negative": 9189, "Positive": 9191}


class TestDomains:
    def test_every_domain_table_has_event_concept_column(self):
        assert set(DOMAIN_TABLES.values()) == set(EVENT_CONCEPT_COLUMNS)


class TestMapConceptId:
    def test_maps_known_values_and_defaults_others(self):
        df = pl.DataFrame({"sex": ["M", "F", "X", "", None]})

        result = df.select(map_concept_id("sex", GENDER_CONCEPT_MAP))

        assert result["sex"].to_list() == [
            GENDER_CONCEPT_MAP["M"],
            GENDER_CONCEPT_MAP["F"],
            DEFAULT_CONCEPT_ID,
            DEFAULT_CONCEPT_ID,
            DEFAULT_CONCEPT_ID,
        ]
        assert result["sex"].dtype == pl.Int64

    def test_all_null_column_gives_default(self):
        df = pl.DataFrame({"sex": [None, None]})

        result = df.select(map_concept_id("sex", GENDER_CONCEPT_MAP))

        assert result["sex"].to_list() == [DEFAULT_CONCEPT_ID] * 2
        assert result["sex"].dtype == pl.Int64
