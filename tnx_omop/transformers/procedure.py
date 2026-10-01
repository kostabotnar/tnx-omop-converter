"""Build OMOP PROCEDURE_OCCURRENCE rows from clinical events."""

import polars as pl

from ..util import columns as col
from .. import omop_schema
from .base import ConceptIds, build_from_events


def build_procedure(events: pl.DataFrame) -> pl.DataFrame:
    """Build PROCEDURE_OCCURRENCE rows from clinical events (see sources.py).

    Only procedure events carry a modifier source value; modifier_concept_id is
    always 0, quantity and end date are unknown.
    """
    return build_from_events(
        events,
        omop_schema.PROCEDURE_OCCURRENCE_SCHEMA,
        {
            col.procedure_concept_id: pl.col(col._concept_id),
            col.procedure_date: pl.col(col._start_date),
            col.procedure_type_concept_id: pl.lit(ConceptIds.EHR_TYPE),
            col.modifier_concept_id: pl.lit(0),
            col.procedure_source_value: pl.col(col._source_value),
            col.procedure_source_concept_id: pl.col(col._source_concept_id),
            col.modifier_source_value: pl.col(col._modifier_source_value),
        },
    )
