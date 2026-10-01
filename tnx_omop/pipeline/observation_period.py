"""Create OMOP OBSERVATION_PERIOD table from the written domain tables.

Reads the merged tables and writes the result; the table logic is in
transformers/observation_period.py.
"""

from pathlib import Path

import polars as pl

from ..util import columns as col
from ..util import tables as tbl
from ..util.parquet_io import write_parquet
from ..transformers.observation_period import build_observation_period

# Date columns that bound a person's observation time, per OMOP table
_EVENT_DATES = [
    (tbl.omop_visit_occurrence, [col.visit_start_date, col.visit_end_date]),
    (tbl.omop_condition_occurrence, [col.condition_start_date, col.condition_end_date]),
    (tbl.omop_procedure_occurrence, [col.procedure_date, col.procedure_end_date]),
    (
        tbl.omop_drug_exposure,
        [col.drug_exposure_start_date, col.drug_exposure_end_date],
    ),
    (tbl.omop_measurement, [col.measurement_date]),
    (tbl.omop_observation, [col.observation_date]),
    (
        tbl.omop_device_exposure,
        [col.device_exposure_start_date, col.device_exposure_end_date],
    ),
]


def create_observation_period(output_dir: Path) -> pl.DataFrame:
    """Create one observation period per person spanning all of their records.

    Reads output_dir/{table}/{table}.parquet and writes
    output_dir/observation_period/observation_period.parquet. Persons without
    any dated record get no observation period.

    Args:
        output_dir: Base output directory containing OMOP tables

    Returns:
        OBSERVATION_PERIOD DataFrame
    """
    ranges = []
    for table_name, date_cols in _EVENT_DATES:
        table_lower = table_name.lower()
        path = output_dir / table_lower / f"{table_lower}.parquet"
        if not path.exists():
            continue
        # Reduce each table to one row per person here, so that the large tables
        # are never held in memory as rows
        ranges.append(
            pl.scan_parquet(path)
            .select(
                pl.col(col.person_id),
                pl.min_horizontal(date_cols).alias(col.observation_period_start_date),
                pl.max_horizontal(date_cols).alias(col.observation_period_end_date),
            )
            .group_by(col.person_id)
            .agg(
                pl.col(col.observation_period_start_date).min(),
                pl.col(col.observation_period_end_date).max(),
            )
        )

    if ranges:
        dates = pl.concat(ranges).collect()
    else:
        dates = pl.DataFrame(
            schema={
                col.person_id: pl.Int64,
                col.observation_period_start_date: pl.Date,
                col.observation_period_end_date: pl.Date,
            }
        )
    periods = build_observation_period(dates)

    table_lower = tbl.omop_observation_period.lower()
    out_dir = output_dir / table_lower
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(periods, out_dir / f"{table_lower}.parquet")

    return periods
