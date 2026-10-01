"""Tests for transformers/observation_period.py (DataFrames only, no files)."""

from datetime import date

import polars as pl

from tests.schema_asserts import assert_conforms
from tnx_omop.schema.omop import OBSERVATION_PERIOD_SCHEMA
from tnx_omop.transformers.base import ConceptIds
from tnx_omop.transformers.observation_period import build_observation_period
from tnx_omop.util import columns as col


def _dates(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            col.person_id: pl.Int64,
            col.observation_period_start_date: pl.Date,
            col.observation_period_end_date: pl.Date,
        },
        orient="row",
    )


def test_spans_all_rows_of_each_person():
    """One period per person from the earliest start to the latest end."""
    result = build_observation_period(
        _dates(
            [
                (2, date(2021, 3, 1), date(2021, 3, 1)),
                (1, date(2020, 1, 5), date(2020, 1, 10)),
                (1, date(2019, 12, 1), date(2020, 2, 1)),
                (2, date(2022, 6, 30), date(2022, 6, 30)),
            ]
        )
    )

    assert result.rows() == [
        (1, 1, date(2019, 12, 1), date(2020, 2, 1), ConceptIds.EHR_TYPE),
        (2, 2, date(2021, 3, 1), date(2022, 6, 30), ConceptIds.EHR_TYPE),
    ]
    assert_conforms(result, OBSERVATION_PERIOD_SCHEMA)


def test_null_end_dates_are_ignored():
    """A missing end date does not hide the other dates."""
    result = build_observation_period(
        _dates(
            [
                (1, date(2020, 1, 1), None),
                (1, date(2020, 5, 1), date(2020, 5, 1)),
            ]
        )
    )

    assert result.row(0)[2:4] == (date(2020, 1, 1), date(2020, 5, 1))


def test_person_without_start_date_gets_no_period():
    result = build_observation_period(_dates([(1, None, None)]))

    assert result.is_empty()


def test_empty_input_returns_typed_empty_table():
    result = build_observation_period(_dates([]))

    assert result.is_empty()
    assert_conforms(result, OBSERVATION_PERIOD_SCHEMA)
