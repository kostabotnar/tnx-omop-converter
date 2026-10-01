"""Build OMOP DRUG_EXPOSURE rows from clinical events."""

import polars as pl

from ..util import columns as col
from ..schema import omop as omop_schema
from .base import ConceptIds, build_from_events


def build_drug(events: pl.DataFrame) -> pl.DataFrame:
    """Build DRUG_EXPOSURE rows from clinical events (see sources.py).

    drug_exposure_end_date is required by the CDM: events without an end date
    (all non-medication sources) use the start date. Route is 0 for non-medication
    sources.
    """
    return build_from_events(
        events,
        omop_schema.DRUG_EXPOSURE_SCHEMA,
        {
            col.drug_concept_id: pl.col(col._concept_id),
            col.drug_exposure_start_date: pl.col(col._start_date),
            col.drug_exposure_end_date: pl.coalesce(
                pl.col(col._end_date), pl.col(col._start_date)
            ),
            col.drug_type_concept_id: pl.lit(ConceptIds.EHR_TYPE),
            col.route_concept_id: pl.col(col._route_concept_id),
            col.drug_source_value: pl.col(col._source_value),
            col.drug_source_concept_id: pl.col(col._source_concept_id),
            col.route_source_value: pl.col(col._route_source_value),
        },
    )
