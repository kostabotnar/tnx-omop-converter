"""Tests for the vocabulary lookup against the synthetic Athena fixture."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import pytest

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.util.concept_mappings import UNIT_UCUM_MAP
from tnx_omop.schema.domains import (
    NO_SOURCE_CONCEPT,
    NO_STANDARD_MAPPING,
    UNSUPPORTED_DOMAIN,
)
from tnx_omop.omop_vocab.vocabulary import (
    CODES_SCHEMA,
    UNITS_SCHEMA,
    VocabularyLookup,
    build_lookup,
    collect_source_codes,
    collect_units,
)
from tests import athena_fixture as fx

DX = tbl.tnx_diagnosis
PX = tbl.tnx_procedure
RX = tbl.tnx_medication_ingredient
LAB = tbl.tnx_lab_result
VITALS = tbl.tnx_vitals_signs

CODES: List[Tuple[str, str, str]] = [
    (DX, "ICD-10-CM", "E11.65"),
    (DX, "ICD-10-CM", "Z87.891"),
    (DX, "ICD-10-CM", "C79.51"),
    (DX, "ICD-10-CM", "C81.90"),
    (DX, "ICD-10-CM", "I10"),
    (DX, "ICD-10-CM", "R05"),
    (DX, "ICD-10-CM", "K21.0"),
    (DX, "ICD-10-CM", "E78.5"),
    (DX, "ICD-10-CM", "C91.00"),
    (DX, "ICD-10-CM", "NOT.A.CODE"),
    (DX, "ICD-9-CM", "38.93"),
    (DX, "SNOMED", "44054006"),
    (PX, "ICD-9-CM", "38.93"),
    (PX, "CPT", "80053"),
    (PX, "CPT", "99214"),
    (PX, "HCPCS", "J1100"),
    (PX, "HCPCS", "E0601"),
    (PX, "CVX", "207"),
    (RX, "RxNorm", "1191"),
    (RX, "RxNorm", "OMOP123"),
    (RX, "RxNorm", "999999"),
    (LAB, "LOINC", "2160-0"),
    (LAB, "LOINC", "72166-2"),
    (LAB, "TNX", "9037-0"),
    (VITALS, "LOINC", "8480-6"),
]

UNITS = ["mg/dL", "mg/dl", "U/L", "[U]/L", "mm[Hg]", "mmHg", "{INR}", "[ppm]"]


@pytest.fixture(scope="module")
def lookup(athena_dir: Path) -> VocabularyLookup:
    codes = pl.DataFrame(
        CODES,
        schema=[col._tnx_table, col.code_system, col.code],
        orient="row",
    )
    units = pl.DataFrame({col.units_of_measure: UNITS})
    return build_lookup(athena_dir, codes, units)


def rows_for(
    lookup: VocabularyLookup, table: str, code_system: str, code: str
) -> pl.DataFrame:
    return lookup.codes.filter(
        (pl.col(col._tnx_table) == table)
        & (pl.col(col.code_system) == code_system)
        & (pl.col(col.code) == code)
    )


def single(
    lookup: VocabularyLookup, table: str, code_system: str, code: str
) -> Tuple[int, Optional[int], Optional[str], Optional[str]]:
    df = rows_for(lookup, table, code_system, code)
    assert df.height == 1, df
    return df.select(
        col.source_concept_id, col.concept_id, col.domain_id, col.exclusion_reason
    ).row(0)


class TestShape:
    def test_schemas(self, lookup: VocabularyLookup):
        assert lookup.codes.schema == CODES_SCHEMA
        assert lookup.units.schema == UNITS_SCHEMA

    def test_every_code_has_a_row(self, lookup: VocabularyLookup):
        keys = set(
            lookup.codes.select(col._tnx_table, col.code_system, col.code).iter_rows()
        )
        assert keys == set(CODES)

    def test_mapped_rows_have_no_reason(self, lookup: VocabularyLookup):
        mapped = lookup.codes.filter(pl.col(col.concept_id).is_not_null())
        assert mapped.get_column(col.exclusion_reason).null_count() == mapped.height
        no_reason = lookup.codes.filter(pl.col(col.exclusion_reason).is_null())
        assert no_reason.get_column(col.concept_id).null_count() == 0

    def test_codes_for(self, lookup: VocabularyLookup):
        df = lookup.codes_for(RX)
        assert col._tnx_table not in df.columns
        assert set(df.get_column(col.code)) == {"1191", "OMOP123", "999999"}


class TestMapping:
    def test_one_to_many(self, lookup: VocabularyLookup):
        df = rows_for(lookup, DX, "ICD-10-CM", "E11.65")
        assert sorted(df.get_column(col.concept_id)) == [
            fx.COND_T2DM,
            fx.COND_HYPERGLYCEMIA,
        ]
        assert set(df.get_column(col.source_concept_id)) == {fx.ICD10_E11_65}
        assert set(df.get_column(col.domain_id)) == {"Condition"}

    def test_cross_domain(self, lookup: VocabularyLookup):
        assert single(lookup, DX, "ICD-10-CM", "Z87.891") == (
            fx.ICD10_Z87_891,
            fx.OBS_EX_SMOKER,
            "Observation",
            None,
        )
        assert single(lookup, LAB, "LOINC", "72166-2") == (
            fx.LOINC_72166_2,
            fx.LOINC_72166_2,
            "Observation",
            None,
        )

    def test_unsupported_domain(self, lookup: VocabularyLookup):
        assert single(lookup, DX, "ICD-10-CM", "C81.90") == (
            fx.ICD10_C81_90,
            None,
            "Episode",
            UNSUPPORTED_DOMAIN,
        )

    def test_supported_and_unsupported_targets(self, lookup: VocabularyLookup):
        df = rows_for(lookup, DX, "ICD-10-CM", "C91.00").sort(col.domain_id)
        assert df.select(
            col.source_concept_id, col.concept_id, col.domain_id, col.exclusion_reason
        ).rows() == [
            (fx.ICD10_C91_00, fx.COND_ALL, "Condition", None),
            (fx.ICD10_C91_00, None, "Episode", UNSUPPORTED_DOMAIN),
        ]

    def test_no_mapping(self, lookup: VocabularyLookup):
        assert single(lookup, DX, "ICD-10-CM", "C79.51") == (
            fx.ICD10_C79_51,
            None,
            None,
            NO_STANDARD_MAPPING,
        )
        assert single(lookup, RX, "RxNorm", "999999") == (
            fx.RXNORM_999999,
            None,
            None,
            NO_STANDARD_MAPPING,
        )

    @pytest.mark.parametrize(
        "table, code_system, code",
        [
            (DX, "ICD-10-CM", "NOT.A.CODE"),
            (DX, "SNOMED", "44054006"),  # code system not in config
            (LAB, "TNX", "9037-0"),  # TNX has no OMOP vocabulary
            (PX, "CVX", "207"),  # vocabulary missing from the download
        ],
    )
    def test_no_source_concept(
        self, lookup: VocabularyLookup, table: str, code_system: str, code: str
    ):
        assert single(lookup, table, code_system, code) == (
            0,
            None,
            None,
            NO_SOURCE_CONCEPT,
        )

    def test_icd9_vocabulary_depends_on_table(self, lookup: VocabularyLookup):
        assert single(lookup, DX, "ICD-9-CM", "38.93") == (
            fx.ICD9CM_38_93,
            fx.COND_ICD9_TARGET,
            "Condition",
            None,
        )
        assert single(lookup, PX, "ICD-9-CM", "38.93") == (
            fx.ICD9PROC_38_93,
            fx.PROC_ICD9_TARGET,
            "Procedure",
            None,
        )

    def test_cpt4_from_cpt4_file(self, lookup: VocabularyLookup):
        assert single(lookup, PX, "CPT", "80053") == (
            fx.CPT4_80053,
            fx.CPT4_80053,
            "Measurement",
            None,
        )

    def test_cpt4_in_both_files_is_one_row(self, lookup: VocabularyLookup):
        assert single(lookup, PX, "CPT", "99214")[1] == fx.CPT4_99214

    def test_hcpcs_to_drug_and_device(self, lookup: VocabularyLookup):
        assert single(lookup, PX, "HCPCS", "J1100")[1:3] == (
            fx.DRUG_DEXAMETHASONE,
            "Drug",
        )
        assert single(lookup, PX, "HCPCS", "E0601")[1:3] == (fx.DEVICE_CPAP, "Device")

    def test_rxnorm_self_map(self, lookup: VocabularyLookup):
        assert single(lookup, RX, "RxNorm", "1191")[:3] == (
            fx.RXNORM_1191,
            fx.RXNORM_1191,
            "Drug",
        )

    def test_rxnorm_extension_prefix(self, lookup: VocabularyLookup):
        assert single(lookup, RX, "RxNorm", "OMOP123")[:3] == (
            fx.RXNORM_EXT_OMOP123,
            fx.RXNORM_EXT_OMOP123,
            "Drug",
        )

    def test_invalid_maps_to_ignored(self, lookup: VocabularyLookup):
        assert single(lookup, DX, "ICD-10-CM", "I10")[1] == fx.COND_HYPERTENSION

    def test_non_standard_target_ignored(self, lookup: VocabularyLookup):
        assert single(lookup, DX, "ICD-10-CM", "R05") == (
            fx.ICD10_R05,
            None,
            None,
            NO_STANDARD_MAPPING,
        )

    def test_invalid_source_concept_kept(self, lookup: VocabularyLookup):
        assert single(lookup, DX, "ICD-10-CM", "K21.0")[:2] == (
            fx.ICD10_K21_0,
            fx.COND_GERD,
        )

    def test_valid_source_concept_preferred(self, lookup: VocabularyLookup):
        assert single(lookup, DX, "ICD-10-CM", "E78.5")[:2] == (
            fx.ICD10_E78_5,
            fx.COND_HLD,
        )

    def test_vitals(self, lookup: VocabularyLookup):
        assert single(lookup, VITALS, "LOINC", "8480-6")[1] == fx.LOINC_8480_6


class TestUnits:
    def test_unit_concepts(self, lookup: VocabularyLookup):
        units = dict(lookup.units.iter_rows())
        assert units == {
            "mg/dL": fx.UCUM_MG_DL,  # exact match
            "mg/dl": 0,  # UCUM codes are case-sensitive
            "U/L": fx.UCUM_U_L,  # override
            "[U]/L": fx.UCUM_U_L,
            "mm[Hg]": fx.UCUM_MM_HG,
            "mmHg": 0,  # unmapped
            "{INR}": 0,
            "[ppm]": 0,  # only a non-standard UCUM concept has this code
        }

    def test_ucum_match_is_case_sensitive(self, lookup: VocabularyLookup):
        units = dict(lookup.units.iter_rows())
        assert units["mg/dL"] == fx.UCUM_MG_DL
        assert units["mg/dl"] == 0

    def test_non_standard_ucum_is_unmapped(self, lookup: VocabularyLookup):
        assert dict(lookup.units.iter_rows())["[ppm]"] == 0

    def test_override_target_missing_warns(
        self, athena_dir: Path, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING, logger="tnx_omop.omop_vocab.vocabulary"):
            build_lookup(
                athena_dir,
                pl.DataFrame(
                    schema={
                        col._tnx_table: pl.String,
                        col.code_system: pl.String,
                        col.code: pl.String,
                    }
                ),
                pl.DataFrame(schema={col.units_of_measure: pl.String}),
            )
        warned = " ".join(r.getMessage() for r in caplog.records)
        assert "[iU]/L" in warned
        assert "[U]/L" not in warned
        missing = [t for t in UNIT_UCUM_MAP.values() if t != "[U]/L"]
        assert len(caplog.records) == len(missing)


class TestEmpty:
    def test_empty_inputs(self, athena_dir: Path):
        result = build_lookup(
            athena_dir,
            collect_source_codes({}),
            collect_units({}),
        )
        assert result.codes.is_empty()
        assert result.codes.schema == CODES_SCHEMA
        assert result.units.is_empty()


class TestCollect:
    def test_collect_source_codes(self):
        tables = {
            DX: pl.DataFrame(
                {
                    col.code_system: ["ICD-10-CM", "ICD-10-CM", None, "ICD-10-CM"],
                    col.code: ["I10", "I10", "X", None],
                    col.patient_id: ["P1", "P2", "P3", "P4"],
                }
            ),
            LAB: pl.DataFrame(
                {col.code_system: ["LOINC"], col.code: ["2160-0"]}
            ).lazy(),
            tbl.tnx_patient: pl.DataFrame({col.patient_id: ["P1"]}),
        }
        codes = collect_source_codes(tables).sort(col._tnx_table)
        assert codes.rows() == [(DX, "ICD-10-CM", "I10"), (LAB, "LOINC", "2160-0")]

    def test_collect_units(self):
        tables = {
            LAB: pl.DataFrame({col.units_of_measure: ["mg/dL", None, "mg/dL"]}),
            VITALS: pl.DataFrame({col.units_of_measure: ["mmHg"]}),
            DX: pl.DataFrame({col.units_of_measure: ["ignored"]}),
        }
        units = collect_units(tables).get_column(col.units_of_measure)
        assert sorted(units) == ["mg/dL", "mmHg"]
