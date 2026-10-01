"""Test helper that adds person_id and visit_occurrence_id to a TriNetX frame."""

import polars as pl

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.util.id_generator import add_visit_ids


def with_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Add IDs numbered over the patients (and encounters) present in df."""
    person_map = (
        df.select(col._source_id, col.patient_id)
        .unique()
        .sort(col._source_id, col.patient_id)
        .with_row_index(col.person_id, offset=1)
        .with_columns(pl.col(col.person_id).cast(pl.Int64))
    )
    df = df.join(person_map, on=[col._source_id, col.patient_id], how="left")
    if col.encounter_id in df.columns:
        return add_visit_ids({tbl.tnx_encounter: df})[0][tbl.tnx_encounter]
    return df
