"""Transform per-person record dates into the OMOP OBSERVATION_PERIOD table."""

import polars as pl

from ..schema.omop import OBSERVATION_PERIOD_SCHEMA
from ..util import columns as col
from .base import ConceptIds, conform


def build_observation_period(dates: pl.DataFrame) -> pl.DataFrame:
    """Build one observation period per person spanning all of their records.

    Args:
        dates: Frame with person_id, observation_period_start_date and
            observation_period_end_date. Each row is the earliest and latest
            date of a person in one source table (or of one record); a person
            may have several rows. Null dates are ignored by the min and max.

    Returns:
        OBSERVATION_PERIOD frame sorted by person_id. Persons without any
        start date get no period. person_id doubles as the period ID because
        each person has one period.
    """
    periods = (
        dates.group_by(col.person_id)
        .agg(
            pl.col(col.observation_period_start_date).min(),
            pl.col(col.observation_period_end_date).max(),
        )
        .filter(pl.col(col.observation_period_start_date).is_not_null())
        .sort(col.person_id)
        .select(
            pl.col(col.person_id).alias(col.observation_period_id),
            pl.col(col.person_id),
            pl.col(col.observation_period_start_date),
            pl.col(col.observation_period_end_date),
            pl.lit(ConceptIds.EHR_TYPE).alias(col.period_type_concept_id),
        )
    )
    return conform(periods, OBSERVATION_PERIOD_SCHEMA)
