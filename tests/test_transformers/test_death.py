"""Tests for tnx_omop/transformers/death.py."""

from datetime import date

import polars as pl

from tnx_omop.util import columns as col
from tests.id_helpers import with_ids
from tnx_omop.transformers.base import ConceptIds
from tnx_omop.omop_schema import DEATH_SCHEMA
from tnx_omop.transformers.death import transform_death


def _patients(deaths):
    return with_ids(
        pl.DataFrame(
            {
                col._source_id: ["default"] * len(deaths),
                col.patient_id: [f"P{i}" for i in range(len(deaths))],
                col.month_year_death: deaths,
            },
            schema_overrides={col.month_year_death: pl.Int64},
        )
    )


class TestTransformDeath:
    """Tests for transform_death function."""

    def test_one_row_per_deceased_patient(self):
        """Keeps only patients with a valid death month."""
        result = transform_death(_patients([202003, None, 201912]))

        assert result[col.death_date].to_list() == [
            date(2020, 3, 1),
            date(2019, 12, 1),
        ]
        assert result[col.person_id].to_list() == [1, 3]
        assert result[col.death_type_concept_id].to_list() == [ConceptIds.EHR_TYPE] * 2
        assert result[col.cause_concept_id].to_list() == [0, 0]

    def test_output_schema(self):
        """Output has the OMOP DEATH columns and types."""
        result = transform_death(_patients([202003]))

        assert result.columns == DEATH_SCHEMA.column_names
        assert result[col.person_id].dtype == pl.Int64
        assert result[col.death_date].dtype == pl.Date

    def test_no_deaths_returns_empty_table(self):
        """Returns a typed empty table when nobody died."""
        result = transform_death(_patients([None, None]))

        assert result.is_empty()
        assert result.columns == DEATH_SCHEMA.column_names

    def test_missing_death_column_returns_empty_table(self):
        """Returns a typed empty table without month_year_death."""
        df = with_ids(
            pl.DataFrame({col._source_id: ["default"], col.patient_id: ["P1"]})
        )

        assert transform_death(df).columns == DEATH_SCHEMA.column_names
