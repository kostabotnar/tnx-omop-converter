"""Transform TriNetX patient table to OMOP DEATH table."""

import polars as pl

from ..util.date_utils import parse_month_year_death_column
from ..util import columns as col
from ..util.concept_mappings import DEFAULT_CONCEPT_ID
from ..omop_schema import DEATH_SCHEMA
from .base import ConceptIds, build_from_events, create_empty_dataframe


def transform_death(patient_df: pl.DataFrame) -> pl.DataFrame:
    """Transform patient DataFrame to OMOP DEATH format.

    death_date is the first day of the month in month_year_death. TriNetX
    reports the month after the actual death month, so the date is shifted
    by up to two months. TriNetX has no cause of death.

    Args:
        patient_df: Cleaned patient DataFrame from TriNetX with person_id
            (see ingest.build_persons)

    Returns:
        OMOP DEATH DataFrame with one row per deceased patient
    """
    if col.month_year_death not in patient_df.columns:
        return create_empty_dataframe(DEATH_SCHEMA.dtype_map)

    df = patient_df.with_columns(
        parse_month_year_death_column(pl.col(col.month_year_death)).alias(
            col.death_date
        )
    ).filter(pl.col(col.death_date).is_not_null())

    if df.is_empty():
        return create_empty_dataframe(DEATH_SCHEMA.dtype_map)

    return build_from_events(
        df,
        DEATH_SCHEMA,
        {
            col.death_type_concept_id: pl.lit(ConceptIds.EHR_TYPE),
            col.cause_concept_id: pl.lit(DEFAULT_CONCEPT_ID),
            col.cause_source_concept_id: pl.lit(DEFAULT_CONCEPT_ID),
        },
    )
