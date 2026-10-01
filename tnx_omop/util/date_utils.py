"""Date parsing utilities for the TriNetX YYYYMM death month."""

import polars as pl


def parse_month_year_death_column(col: pl.Expr) -> pl.Expr:
    """Create a polars expression to parse a YYYYMM death month column.

    TriNetX reports the month after the actual month of death, so the first day
    of the reported month is the earliest date that is certainly after death.
    Handles both string and integer columns (e.g., "202003" or 202003).

    Args:
        col: Polars expression for the month_year_death column

    Returns:
        Polars expression with the first day of that month as Date, or null
        if the value is not a valid YYYYMM month
    """
    value = col.cast(pl.Utf8)
    return (
        pl.when(value.str.contains(r"^\d{6}$"))
        .then(value.str.to_date(format="%Y%m", strict=False))
        .otherwise(None)
    )
