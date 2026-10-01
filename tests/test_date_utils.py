"""Tests for tnx_omop/util/date_utils.py."""

from datetime import date

import polars as pl

from tnx_omop.util.date_utils import parse_month_year_death_column


class TestParseMonthYearDeathColumn:
    """Tests for parse_month_year_death_column function."""

    def test_string_and_integer_values(self):
        """Parses YYYYMM strings and integers to the first day of the month."""
        strings = pl.DataFrame({"m": ["202003", "201912", "202402"]})
        integers = pl.DataFrame({"m": [202003]})

        parsed = strings.select(parse_month_year_death_column(pl.col("m")))
        parsed_int = integers.select(parse_month_year_death_column(pl.col("m")))

        assert parsed["m"].to_list() == [
            date(2020, 3, 1),
            date(2019, 12, 1),
            date(2024, 2, 1),
        ]
        assert parsed_int["m"].to_list() == [date(2020, 3, 1)]

    def test_invalid_values_are_null(self):
        """Returns null for missing, malformed, or MMYYYY values."""
        df = pl.DataFrame({"m": [None, "", "202013", "202000", "032020", "20203"]})

        parsed = df.select(parse_month_year_death_column(pl.col("m")))

        assert parsed["m"].to_list() == [None] * 6
