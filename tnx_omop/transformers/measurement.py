"""Build OMOP MEASUREMENT rows from clinical events."""

import polars as pl

from ..util import columns as col
from .. import omop_schema
from .base import ConceptIds, build_from_events


def build_measurement(events: pl.DataFrame) -> pl.DataFrame:
    """Build MEASUREMENT rows from clinical events (see sources.py).

    Values and units come from lab and vitals events; events from other sources
    have null values and 0 value and unit concepts.
    """
    return build_from_events(
        events,
        omop_schema.MEASUREMENT_SCHEMA,
        {
            col.measurement_concept_id: pl.col(col._concept_id),
            col.measurement_date: pl.col(col._start_date),
            col.measurement_type_concept_id: pl.lit(ConceptIds.EHR_TYPE),
            col.operator_concept_id: pl.lit(0),
            col.value_as_number: pl.col(col._value_as_number),
            col.value_as_concept_id: pl.col(col._value_as_concept_id),
            col.unit_concept_id: pl.col(col._unit_concept_id),
            col.measurement_source_value: pl.col(col._source_value),
            col.measurement_source_concept_id: pl.col(col._source_concept_id),
            col.unit_source_value: pl.col(col._unit_source_value),
            col.unit_source_concept_id: pl.lit(0),
            col.value_source_value: pl.col(col._value_source_value),
            col.meas_event_field_concept_id: pl.lit(0),
        },
    )
