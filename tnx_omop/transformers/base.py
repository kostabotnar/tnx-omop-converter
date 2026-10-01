"""Shared utilities for OMOP transformers."""

from typing import Dict

import polars as pl

from ..schema.omop import TableSchema


class ConceptIds:
    """Centralized OMOP concept IDs."""

    EHR_TYPE = 32817  # EHR record type concept


def create_empty_dataframe(schema: Dict[str, pl.DataType]) -> pl.DataFrame:
    """Create an empty DataFrame with the specified schema.

    Args:
        schema: Dictionary mapping column names to Polars data types

    Returns:
        Empty DataFrame with correct schema
    """
    return pl.DataFrame(
        {name: pl.Series([], dtype=dtype) for name, dtype in schema.items()}
    )


def conform(df: pl.DataFrame, schema: TableSchema) -> pl.DataFrame:
    """Select exactly the schema columns, in schema order, cast to schema dtypes.

    Args:
        df: Frame holding every column of the schema (extra columns are dropped).
        schema: Target OMOP table schema.

    Returns:
        Frame whose columns and dtypes match the schema.

    Raises:
        polars.exceptions.ColumnNotFoundError: If a schema column is missing.
    """
    return df.select([pl.col(c.name).cast(c.dtype) for c in schema.columns])


def build_from_events(
    events: pl.DataFrame, schema: TableSchema, columns: Dict[str, pl.Expr]
) -> pl.DataFrame:
    """Build an OMOP table from clinical events.

    Each schema column takes its expression from columns, else the same-named
    event column (person_id, visit_occurrence_id), else null. The primary key
    stays null here; the converter numbers it when it merges the parts of a table.

    Args:
        events: Frame with EVENT_SCHEMA (see transformers/sources.py).
        schema: Target OMOP table schema.
        columns: Expressions for the table columns that are not null.

    Returns:
        Frame conformed to the schema.
    """
    exprs = []
    for name in schema.column_names:
        if name in columns:
            expr = columns[name]
        elif name in events.columns:
            expr = pl.col(name)
        else:
            expr = pl.lit(None)
        exprs.append(expr.alias(name))
    return conform(events.select(exprs), schema)
