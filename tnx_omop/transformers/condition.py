"""Build OMOP CONDITION_OCCURRENCE rows from clinical events."""

import polars as pl

from ..util import columns as col
from ..schema import omop as omop_schema
from .base import ConceptIds, build_from_events


def build_condition(events: pl.DataFrame) -> pl.DataFrame:
    """Build CONDITION_OCCURRENCE rows from clinical events (see sources.py).

    Status comes from the event, so events from non-diagnosis sources get status 0
    and a null status source value. The end date is never known.
    """
    return build_from_events(
        events,
        omop_schema.CONDITION_OCCURRENCE_SCHEMA,
        {
            col.condition_concept_id: pl.col(col._concept_id),
            col.condition_start_date: pl.col(col._start_date),
            col.condition_type_concept_id: pl.lit(ConceptIds.EHR_TYPE),
            col.condition_status_concept_id: pl.col(col._status_concept_id),
            col.condition_source_value: pl.col(col._source_value),
            col.condition_source_concept_id: pl.col(col._source_concept_id),
            col.condition_status_source_value: pl.col(col._status_source_value),
        },
    )
