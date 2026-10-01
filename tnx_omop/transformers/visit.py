"""Transform TriNetX encounter table to OMOP VISIT_OCCURRENCE table."""

import polars as pl

from ..util import columns as col
from ..util.concept_mappings import (
    VISIT_TYPE_CONCEPT_MAP,
    DEFAULT_CONCEPT_ID,
    map_concept_id,
)
from ..omop_schema import VISIT_OCCURRENCE_SCHEMA
from .base import ConceptIds, build_from_events, create_empty_dataframe


def transform_visit(encounter_df: pl.DataFrame) -> pl.DataFrame:
    """Transform encounter DataFrame to OMOP VISIT_OCCURRENCE format.

    Args:
        encounter_df: Cleaned encounter DataFrame from TriNetX with person_id and
            visit_occurrence_id (see util/id_generator.add_visit_ids)

    Returns:
        OMOP VISIT_OCCURRENCE DataFrame
    """
    df = encounter_df

    if df.is_empty():
        return create_empty_dataframe(VISIT_OCCURRENCE_SCHEMA.dtype_map)

    df = df.with_columns(
        pl.col(col.start_date).alias(col.visit_start_date),
        pl.col(col.end_date).alias(col.visit_end_date),
    )

    return build_from_events(
        df,
        VISIT_OCCURRENCE_SCHEMA,
        {
            col.visit_concept_id: (
                map_concept_id(col.type, VISIT_TYPE_CONCEPT_MAP)
                if col.type in df.columns
                else pl.lit(DEFAULT_CONCEPT_ID)
            ),
            col.visit_type_concept_id: pl.lit(ConceptIds.EHR_TYPE),
            col.visit_source_value: pl.col(col.encounter_id),
            col.visit_source_concept_id: pl.lit(DEFAULT_CONCEPT_ID),
            col.admitted_from_concept_id: pl.lit(DEFAULT_CONCEPT_ID),
            col.discharged_to_concept_id: pl.lit(DEFAULT_CONCEPT_ID),
        },
    )
