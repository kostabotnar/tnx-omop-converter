"""Transform TriNetX patient table to OMOP PERSON table."""

import polars as pl

from ..schema.omop import PERSON_SCHEMA
from ..util import columns as col
from ..util.concept_mappings import (
    GENDER_CONCEPT_MAP,
    RACE_CONCEPT_MAP,
    ETHNICITY_CONCEPT_MAP,
    DEFAULT_CONCEPT_ID,
    map_concept_id,
)
from .base import build_from_events


def transform_person(patient_df: pl.DataFrame) -> pl.DataFrame:
    """Transform patient DataFrame to OMOP PERSON format.

    Args:
        patient_df: Cleaned patient DataFrame from TriNetX with person_id
            (see ingest.build_persons)

    Returns:
        OMOP PERSON DataFrame
    """
    df = patient_df

    def mapped(source_col: str, mapping: dict[str, int]) -> pl.Expr:
        if source_col in df.columns:
            return map_concept_id(source_col, mapping)
        return pl.lit(DEFAULT_CONCEPT_ID)

    def source_value(source_col: str) -> pl.Expr:
        return pl.col(source_col) if source_col in df.columns else pl.lit(None)

    return build_from_events(
        df,
        PERSON_SCHEMA,
        {
            col.gender_concept_id: mapped(col.sex, GENDER_CONCEPT_MAP),
            col.race_concept_id: mapped(col.race, RACE_CONCEPT_MAP),
            col.ethnicity_concept_id: mapped(col.ethnicity, ETHNICITY_CONCEPT_MAP),
            col.person_source_value: pl.col(col.patient_id),
            col.gender_source_value: source_value(col.sex),
            col.gender_source_concept_id: pl.lit(DEFAULT_CONCEPT_ID),
            col.race_source_value: source_value(col.race),
            col.race_source_concept_id: pl.lit(DEFAULT_CONCEPT_ID),
            col.ethnicity_source_value: source_value(col.ethnicity),
            col.ethnicity_source_concept_id: pl.lit(DEFAULT_CONCEPT_ID),
        },
    )
