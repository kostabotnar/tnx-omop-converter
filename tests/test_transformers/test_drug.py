"""Tests for DRUG_EXPOSURE built through transform_clinical."""

from datetime import date

import polars as pl
import pytest

from tnx_omop.util import columns as col
from tests.id_helpers import with_ids
from tnx_omop.util import tables as tbl
from tnx_omop.util.concept_mappings import DEFAULT_CONCEPT_ID, ROUTE_CONCEPT_MAP
from tnx_omop.omop_schema import DRUG_EXPOSURE_SCHEMA
from tnx_omop.transformers.clinical import transform_clinical
from tests import athena_fixture as fx
from tests.schema_asserts import assert_conforms

RX = tbl.tnx_medication_ingredient


@pytest.fixture
def drugs(make_lookup):
    def run(df: pl.DataFrame) -> pl.DataFrame:
        return transform_clinical(df, RX, make_lookup({RX: df}))[tbl.omop_drug_exposure]

    return run


def medication(routes=None, n=None, **columns):
    n = n if n is not None else len(routes)
    data = {
        col._source_id: ["default"] * n,
        col.patient_id: ["P001"] * n,
        col.encounter_id: ["E001"] * n,
        col.unique_id: [f"M{i:03d}" for i in range(n)],
        col.start_date: [date(2023, 1, 1)] * n,
        col.code_system: ["RxNorm"] * n,
        col.code: ["197361"] * n,
    }
    if routes is not None:
        data[col.route] = routes
    data.update(columns)
    return with_ids(pl.DataFrame(data))


def route_map(result: pl.DataFrame) -> dict:
    return dict(result.select(col.route_source_value, col.route_concept_id).iter_rows())


class TestBuildDrug:
    def test_transforms_basic_medication(self, drugs, sample_medication_df):
        result = drugs(sample_medication_df)

        assert len(result) == 2
        assert_conforms(result, DRUG_EXPOSURE_SCHEMA, numbered=False)
        assert set(result[col.drug_concept_id]) == {
            fx.RXNORM_197361,
            fx.RXNORM_312961,
        }
        assert set(result[col.drug_source_value]) == {"197361", "312961"}
        assert result[col.drug_exposure_end_date].to_list() == [date(2023, 1, 7)] * 2

    def test_end_date_falls_back_to_start(self, drugs):
        result = drugs(medication(n=1, **{col.start_date: [date(2023, 1, 5)]}))

        assert result[col.drug_exposure_end_date][0] == date(2023, 1, 5)

    def test_maps_route_concept_ids(self, drugs):
        routes = [
            "Oral Product",
            "Injectable Product",
            "Inhalant Product",
            "Topical Product",
            "Nasal Product",
            "Ophthalmic Product",
            "Rectal Product",
            "Otic Product",
            "Vaginal Product",
            "Urethral Product",
            "Intraperitoneal Product",
            "Drug Implant Product",
        ]
        result = route_map(drugs(medication(routes)))

        assert result == {r: ROUTE_CONCEPT_MAP[r] for r in routes}

    def test_unmapped_route_defaults_to_zero(self, drugs):
        result = drugs(medication(["Unknown", "Sublingual", None]))

        assert result[col.route_concept_id].to_list() == [DEFAULT_CONCEPT_ID] * 3

    def test_handles_missing_route_column(self, drugs):
        result = drugs(medication(n=1))

        assert len(result) == 1
        assert result[col.route_concept_id][0] == DEFAULT_CONCEPT_ID
        assert result[col.route_source_value][0] is None

    def test_keeps_derived_records(self, drugs):
        result = drugs(
            medication(
                ["Oral Product", "Injectable Product"],
                **{col.derived_by_TriNetX: [None, "T"]},
            )
        )

        assert len(result) == 2
        assert col.derived_by_TriNetX not in result.columns

    def test_rxnorm_extension_and_unmapped_codes(self, make_lookup):
        df = medication(n=2, **{col.code: ["OMOP123", "999999"]})
        result = transform_clinical(df, RX, make_lookup({RX: df}))

        assert result[tbl.omop_drug_exposure][col.drug_concept_id].to_list() == [
            fx.RXNORM_EXT_OMOP123
        ]
        assert result[tbl.excluded][col.code].to_list() == ["999999"]

    def test_keeps_duplicate_rows(self, drugs):
        df = medication(n=1)
        assert len(drugs(pl.concat([df, df]))) == 2
