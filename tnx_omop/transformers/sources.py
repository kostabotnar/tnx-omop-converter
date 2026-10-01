"""Source adapters: turn mapped TriNetX rows into domain-neutral clinical events.

Each adapter takes the mapped frame from clinical.apply_lookup (original TriNetX
columns plus _concept_id, _source_concept_id, _domain_id) and the unit
lookup, and returns a frame with exactly EVENT_SCHEMA. Domain builders then turn
events into OMOP rows, whatever TriNetX table the events came from.

Concept ID event columns that do not apply to a source are 0, so builders can pass
them through unchanged.
"""

from __future__ import annotations

from typing import Callable, Dict

import polars as pl

from ..util import columns as col
from ..util import tables as tbl
from ..util.concept_mappings import (
    CONDITION_STATUS_CONCEPT_MAP,
    DEFAULT_CONCEPT_ID,
    LAB_VALUE_CONCEPT_MAP,
    ROUTE_CONCEPT_MAP,
    map_concept_id,
)

EVENT_SCHEMA = pl.Schema(
    {
        col.person_id: pl.Int64,
        col.visit_occurrence_id: pl.Int64,
        col._start_date: pl.Date,
        col._end_date: pl.Date,
        col._concept_id: pl.Int64,
        col._source_concept_id: pl.Int64,
        col._source_value: pl.Utf8,
        col._value_as_number: pl.Float64,
        col._value_source_value: pl.Utf8,
        col._value_as_concept_id: pl.Int64,
        col._unit_concept_id: pl.Int64,
        col._unit_source_value: pl.Utf8,
        col._status_concept_id: pl.Int64,
        col._status_source_value: pl.Utf8,
        col._route_concept_id: pl.Int64,
        col._route_source_value: pl.Utf8,
        col._modifier_source_value: pl.Utf8,
    }
)

SourceAdapter = Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]

_NO_CONCEPT = pl.lit(DEFAULT_CONCEPT_ID, dtype=pl.Int64)


def _optional(df: pl.DataFrame, name: str) -> pl.Expr:
    """Column if present, else a null string."""
    return pl.col(name) if name in df.columns else pl.lit(None, dtype=pl.Utf8)


def _optional_concept(df: pl.DataFrame, name: str, mapping: Dict[str, int]) -> pl.Expr:
    return map_concept_id(name, mapping) if name in df.columns else _NO_CONCEPT


def _qualified_code() -> pl.Expr:
    """'code_system:code', or the bare code when code_system is empty."""
    system = pl.col(col.code_system)
    return (
        pl.when(system.is_not_null() & (system != ""))
        .then(pl.concat_str([system, pl.col(col.code)], separator=":"))
        .otherwise(pl.col(col.code).cast(pl.Utf8))
    )


def _events(df: pl.DataFrame, columns: Dict[str, pl.Expr]) -> pl.DataFrame:
    """Build EVENT_SCHEMA from shared columns plus source-specific expressions.

    Event columns not given in columns are null (strings, numbers, dates) or 0
    (concept IDs).
    """
    shared = {
        col.person_id: pl.col(col.person_id),
        col.visit_occurrence_id: pl.col(col.visit_occurrence_id),
        col._concept_id: pl.col(col._concept_id),
        col._source_concept_id: pl.col(col._source_concept_id),
    }
    exprs = []
    for name, dtype in EVENT_SCHEMA.items():
        if name in columns:
            expr = columns[name]
        elif name in shared:
            expr = shared[name]
        elif name.endswith("_concept_id"):
            expr = _NO_CONCEPT
        else:
            expr = pl.lit(None)
        exprs.append(expr.cast(dtype).alias(name))
    return df.select(exprs)


def diagnosis_events(df: pl.DataFrame, units: pl.DataFrame) -> pl.DataFrame:
    """Events from the diagnosis table (status from principal_diagnosis_indicator)."""
    return _events(
        df,
        {
            col._start_date: pl.col(col.date),
            col._source_value: _qualified_code(),
            col._status_concept_id: _optional_concept(
                df, col.principal_diagnosis_indicator, CONDITION_STATUS_CONCEPT_MAP
            ),
            col._status_source_value: _optional(df, col.principal_diagnosis_indicator),
        },
    )


def procedure_events(df: pl.DataFrame, units: pl.DataFrame) -> pl.DataFrame:
    """Events from the procedure table (modifier from modifier_1)."""
    return _events(
        df,
        {
            col._start_date: pl.col(col.date),
            col._source_value: _qualified_code(),
            col._modifier_source_value: _optional(df, col.modifier_1),
        },
    )


def medication_events(df: pl.DataFrame, units: pl.DataFrame) -> pl.DataFrame:
    """Events from medication_ingredient (end date falls back to start date)."""
    start = pl.col(col.start_date)
    end = (
        pl.coalesce(pl.col(col.end_date), start)
        if col.end_date in df.columns
        else start
    )
    return _events(
        df,
        {
            col._start_date: start,
            col._end_date: end,
            col._source_value: pl.col(col.code),
            col._route_concept_id: _optional_concept(df, col.route, ROUTE_CONCEPT_MAP),
            col._route_source_value: _optional(df, col.route),
        },
    )


def _result_events(
    df: pl.DataFrame, units: pl.DataFrame, number_col: str, text_col: str
) -> pl.DataFrame:
    unit = _optional(df, col.units_of_measure)
    text = _optional(df, text_col)
    return _events(
        df,
        {
            col._start_date: pl.col(col.date),
            col._source_value: pl.col(col.code),
            col._value_as_number: (
                pl.col(number_col)
                if number_col in df.columns
                else pl.lit(None, dtype=pl.Float64)
            ),
            # TriNetX writes "" when a result has no text value
            col._value_source_value: pl.when(text != "").then(text),
            col._value_as_concept_id: _optional_concept(
                df, text_col, LAB_VALUE_CONCEPT_MAP
            ),
            col._unit_concept_id: unit.replace_strict(
                units.get_column(col.units_of_measure),
                units.get_column(col.unit_concept_id),
                default=DEFAULT_CONCEPT_ID,
                return_dtype=pl.Int64,
            ),
            col._unit_source_value: unit,
        },
    )


def lab_events(df: pl.DataFrame, units: pl.DataFrame) -> pl.DataFrame:
    """Events from lab_result (numeric and text values, UCUM units)."""
    return _result_events(df, units, col.lab_result_num_val, col.lab_result_text_val)


def vitals_events(df: pl.DataFrame, units: pl.DataFrame) -> pl.DataFrame:
    """Events from vitals_signs (numeric and text values, UCUM units)."""
    return _result_events(df, units, col.value, col.text_value)


SOURCE_ADAPTERS: Dict[str, SourceAdapter] = {
    tbl.tnx_diagnosis: diagnosis_events,
    tbl.tnx_procedure: procedure_events,
    tbl.tnx_medication_ingredient: medication_events,
    tbl.tnx_lab_result: lab_events,
    tbl.tnx_vitals_signs: vitals_events,
}
