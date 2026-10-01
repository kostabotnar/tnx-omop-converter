"""Build OMOP DEVICE_EXPOSURE rows from clinical events."""

import polars as pl

from ..util import columns as col
from .. import omop_schema
from .base import ConceptIds, build_from_events


def build_device(events: pl.DataFrame) -> pl.DataFrame:
    """Build DEVICE_EXPOSURE rows from clinical events (see sources.py).

    TriNetX has no device end date, quantity or unit, so those stay null or 0.
    """
    return build_from_events(
        events,
        omop_schema.DEVICE_EXPOSURE_SCHEMA,
        {
            col.device_concept_id: pl.col(col._concept_id),
            col.device_exposure_start_date: pl.col(col._start_date),
            col.device_type_concept_id: pl.lit(ConceptIds.EHR_TYPE),
            col.device_source_value: pl.col(col._source_value),
            col.device_source_concept_id: pl.col(col._source_concept_id),
            col.unit_concept_id: pl.lit(0),
            col.unit_source_concept_id: pl.lit(0),
        },
    )
