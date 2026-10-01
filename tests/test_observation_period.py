"""Tests for tnx_omop/observation_period.py."""

from datetime import date
from pathlib import Path

import polars as pl

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.observation_period import create_observation_period
from tnx_omop.transformers.base import ConceptIds


def _write(output_dir: Path, table_name: str, data: dict) -> None:
    table_dir = output_dir / table_name.lower()
    table_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data).write_parquet(table_dir / f"{table_name.lower()}.parquet")


def test_spans_all_records_of_each_person(tmp_path):
    """One period per person from the earliest to the latest record date."""
    _write(
        tmp_path,
        tbl.omop_visit_occurrence,
        {
            col.person_id: [1, 2],
            col.visit_start_date: [date(2020, 1, 5), date(2021, 3, 1)],
            col.visit_end_date: [date(2020, 1, 10), date(2021, 3, 1)],
        },
    )
    _write(
        tmp_path,
        tbl.omop_drug_exposure,
        {
            col.person_id: [1],
            col.drug_exposure_start_date: [date(2019, 12, 1)],
            col.drug_exposure_end_date: [date(2020, 2, 1)],
        },
    )
    _write(
        tmp_path,
        tbl.omop_measurement,
        {col.person_id: [2], col.measurement_date: [date(2022, 6, 30)]},
    )

    result = create_observation_period(tmp_path)

    assert result.rows() == [
        (1, 1, date(2019, 12, 1), date(2020, 2, 1), ConceptIds.EHR_TYPE),
        (2, 2, date(2021, 3, 1), date(2022, 6, 30), ConceptIds.EHR_TYPE),
    ]
    written = pl.read_parquet(
        tmp_path / "observation_period" / "observation_period.parquet"
    )
    assert written.equals(result)


def test_null_end_dates_are_ignored(tmp_path):
    """Missing end dates do not hide the other dates of a row."""
    _write(
        tmp_path,
        tbl.omop_condition_occurrence,
        {
            col.person_id: [1, 1],
            col.condition_start_date: [date(2020, 1, 1), date(2020, 5, 1)],
            col.condition_end_date: pl.Series([None, None], dtype=pl.Date),
        },
    )

    result = create_observation_period(tmp_path)

    assert result.select(
        col.observation_period_start_date, col.observation_period_end_date
    ).row(0) == (date(2020, 1, 1), date(2020, 5, 1))


def test_no_tables_returns_empty_period_table(tmp_path):
    """Writes an empty, typed table when no domain tables exist."""
    result = create_observation_period(tmp_path)

    assert result.is_empty()
    assert col.observation_period_id in result.columns
