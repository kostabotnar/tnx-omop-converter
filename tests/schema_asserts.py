"""Assertion helpers for OMOP table frames."""

import polars as pl

from tnx_omop.schema.omop import TableSchema


def assert_conforms(
    df: pl.DataFrame, schema: TableSchema, numbered: bool = True
) -> None:
    """Columns, order and dtypes match the schema; required columns have no nulls.

    numbered=False skips the primary key of event tables: the transformers leave
    it null and the converter numbers it when it merges the parts.
    """
    assert df.columns == schema.column_names
    assert dict(df.schema) == schema.dtype_map
    required = [
        c for c in schema.required_columns if numbered or c != schema.primary_key
    ]
    nulls = {c: df[c].null_count() for c in required}
    assert not any(nulls.values()), nulls
