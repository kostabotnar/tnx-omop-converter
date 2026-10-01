"""Sequential, deterministic ID assignment for OMOP tables.

person_id is numbered by ingest (see ingest.build_persons). visit_occurrence_id is
numbered per batch over the sorted distinct (person_id, encounter_id) pairs of the
cleaned encounter table, continuing from the previous batch. Record IDs are numbered
per OMOP table and batch after a deterministic sort, continuing from the previous
batch. Batches are contiguous person_id ranges, so the IDs run 1..N over the whole
output and do not depend on the batch size. The same input and vocabulary give the
same IDs; IDs differ between different exports.
"""

from __future__ import annotations

from typing import Dict

import polars as pl

from . import columns as col
from . import tables as tbl

_VISIT_TABLES = [
    tbl.tnx_encounter,
    tbl.tnx_diagnosis,
    tbl.tnx_procedure,
    tbl.tnx_medication_ingredient,
    tbl.tnx_lab_result,
    tbl.tnx_vitals_signs,
]

_VISIT_KEY = [col.person_id, col.encounter_id]


def build_visit_map(encounter: pl.DataFrame | None, offset: int = 0) -> pl.DataFrame:
    """Number the distinct (person_id, encounter_id) pairs offset+1.. in sorted order.

    Args:
        encounter: Cleaned encounter table of one batch with person_id, or None.
        offset: Last visit_occurrence_id of the previous batch.

    Returns:
        Frame with person_id, encounter_id (string) and visit_occurrence_id (Int64).
    """
    if encounter is None:
        return pl.DataFrame(
            schema={
                col.person_id: pl.Int64,
                col.encounter_id: pl.Utf8,
                col.visit_occurrence_id: pl.Int64,
            }
        )
    return (
        encounter.select(pl.col(col.person_id), pl.col(col.encounter_id).cast(pl.Utf8))
        .drop_nulls()
        .unique()
        .sort(_VISIT_KEY)
        .with_row_index(col.visit_occurrence_id, offset=offset + 1)
        .select(
            col.person_id,
            col.encounter_id,
            pl.col(col.visit_occurrence_id).cast(pl.Int64),
        )
    )


def add_visit_ids(
    tables: Dict[str, pl.DataFrame], offset: int = 0
) -> tuple[Dict[str, pl.DataFrame], int]:
    """Add visit_occurrence_id to the encounter and clinical tables of one batch.

    The ID comes from the encounter table and is null for rows without an
    encounter_id (and for every row when there is no encounter table).

    Args:
        tables: Cleaned tables of one batch with person_id (see
            cleaning.clean_batch).
        offset: Last visit_occurrence_id of the previous batch.

    Returns:
        (tables, last_id): a new dict with the same tables plus the ID column, and
        the last ID used (offset when the batch has no encounters). The input dict
        is left unchanged.
    """
    visit_map = build_visit_map(tables.get(tbl.tnx_encounter), offset)
    last_id = visit_map.get_column(col.visit_occurrence_id).max()

    result = dict(tables)
    for name in _VISIT_TABLES:
        df = result.get(name)
        if df is None:
            continue
        if col.encounter_id not in df.columns:
            result[name] = df.with_columns(
                pl.lit(None, dtype=pl.Int64).alias(col.visit_occurrence_id)
            )
            continue
        result[name] = df.with_columns(pl.col(col.encounter_id).cast(pl.Utf8)).join(
            visit_map, on=_VISIT_KEY, how="left", maintain_order="left"
        )
    return result, offset if last_id is None else last_id


def with_record_ids(
    lf: pl.LazyFrame, id_col: str, sort_by: list[str], offset: int = 0
) -> pl.LazyFrame:
    """Sort rows by sort_by, then number them offset+1.. in id_col (Int64).

    Args:
        lf: Rows of one OMOP table; id_col holds a placeholder (for example null).
        id_col: Primary key column of the table.
        sort_by: Sort keys. Add every remaining column at the end so the order, and
            so the IDs, do not depend on the order of the input rows.
        offset: Rows numbered before this batch, so IDs continue across batches.

    Returns:
        LazyFrame with the same columns in the same order, id_col numbered.
    """
    names = lf.collect_schema().names()
    return (
        lf.drop(id_col)
        .sort(sort_by, nulls_last=True)
        .with_row_index(id_col, offset=offset + 1)
        .select(pl.col(id_col).cast(pl.Int64), pl.exclude(id_col))
        .select(names)
    )
