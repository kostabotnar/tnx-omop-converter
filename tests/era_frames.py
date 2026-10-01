"""Small frames shared by the era tests."""

from datetime import date, timedelta

import polars as pl

from tnx_omop.util import columns as col

D0 = date(2020, 1, 1)

CONDITION = 201826
OTHER_CONDITION = 201254


def day(offset: int) -> date:
    return D0 + timedelta(days=offset)


def conditions(rows: list[tuple]) -> pl.DataFrame:
    """CONDITION_OCCURRENCE columns the eras read: person, concept, start, end."""
    return pl.DataFrame(
        rows,
        schema={
            col.person_id: pl.Int64,
            col.condition_concept_id: pl.Int64,
            col.condition_start_date: pl.Date,
            col.condition_end_date: pl.Date,
        },
        orient="row",
    )


def exposures(rows: list[tuple]) -> pl.DataFrame:
    """DRUG_EXPOSURE columns the eras read: person, concept, start, end, days supply."""
    return pl.DataFrame(
        rows,
        schema={
            col.person_id: pl.Int64,
            col.drug_concept_id: pl.Int64,
            col.drug_exposure_start_date: pl.Date,
            col.drug_exposure_end_date: pl.Date,
            col.days_supply: pl.Int64,
        },
        orient="row",
    )


def ingredient_pairs(*pairs: tuple[int, int]) -> pl.DataFrame:
    return pl.DataFrame(
        pairs,
        schema={col.drug_concept_id: pl.Int64, col.ingredient_concept_id: pl.Int64},
        orient="row",
    )
