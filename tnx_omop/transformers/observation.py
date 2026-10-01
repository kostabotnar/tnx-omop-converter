"""Build OMOP OBSERVATION rows from clinical events."""

import polars as pl

from ..util import columns as col
from .. import omop_schema
from .base import ConceptIds, build_from_events


def build_observation(events: pl.DataFrame) -> pl.DataFrame:
    """Build OBSERVATION rows from clinical events (see sources.py).

    Lab and vitals events carry values and units (text value in both
    value_as_string and value_source_value); other sources leave them null or 0.
    """
    return build_from_events(
        events,
        omop_schema.OBSERVATION_SCHEMA,
        {
            col.observation_concept_id: pl.col(col._concept_id),
            col.observation_date: pl.col(col._start_date),
            col.observation_type_concept_id: pl.lit(ConceptIds.EHR_TYPE),
            col.value_as_number: pl.col(col._value_as_number),
            col.value_as_string: pl.col(col._value_source_value),
            col.value_as_concept_id: pl.col(col._value_as_concept_id),
            col.qualifier_concept_id: pl.lit(0),
            col.unit_concept_id: pl.col(col._unit_concept_id),
            col.observation_source_value: pl.col(col._source_value),
            col.observation_source_concept_id: pl.col(col._source_concept_id),
            col.unit_source_value: pl.col(col._unit_source_value),
            col.value_source_value: pl.col(col._value_source_value),
            col.obs_event_field_concept_id: pl.lit(0),
        },
    )
