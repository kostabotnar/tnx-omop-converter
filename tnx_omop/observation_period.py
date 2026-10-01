"""Create OMOP OBSERVATION_PERIOD table from the written domain tables."""

from pathlib import Path

import polars as pl

from .util import columns as col
from .util import tables as tbl
from .util.parquet_io import write_parquet
from .omop_schema import OBSERVATION_PERIOD_SCHEMA
from .transformers.base import ConceptIds, conform, create_empty_dataframe

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
        ranges.append(
            pl.scan_parquet(path).select(
                pl.col(col.person_id),
                pl.min_horizontal(date_cols).alias("_start"),
                pl.max_horizontal(date_cols).alias("_end"),
            )
        )

    if ranges:
        periods = (
            pl.concat(ranges)
            .group_by(col.person_id)
            .agg(
                pl.col("_start").min().alias(col.observation_period_start_date),
                pl.col("_end").max().alias(col.observation_period_end_date),
            )
            .filter(pl.col(col.observation_period_start_date).is_not_null())
            .sort(col.person_id)
            .select(
                # One period per person, so person_id is a unique period ID
                pl.col(col.person_id).alias(col.observation_period_id),
                pl.col(col.person_id),
                pl.col(col.observation_period_start_date),
                pl.col(col.observation_period_end_date),
                pl.lit(ConceptIds.EHR_TYPE).alias(col.period_type_concept_id),
            )
            .collect()
        )
        periods = conform(periods, OBSERVATION_PERIOD_SCHEMA)
    else:
        periods = create_empty_dataframe(OBSERVATION_PERIOD_SCHEMA.dtype_map)

    table_lower = tbl.omop_observation_period.lower()
    out_dir = output_dir / table_lower
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(periods, out_dir / f"{table_lower}.parquet")

    return periods
