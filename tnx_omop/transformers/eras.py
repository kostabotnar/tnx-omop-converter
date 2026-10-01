"""Build CONDITION_ERA and DRUG_ERA rows from occurrence and exposure frames.

An era is a run of records of one person and one concept that are close in time.
The logic follows the standard OHDSI era scripts (the CDM era SQL of the
OHDSI/CommonDataModel repository, "Condition Era" and "Drug Era", also used by
Achilles and ETL-CMS); the eras are built here with Polars instead of SQL.

Persistence window: two records belong to the same era when the gap between the
start of the later one and the latest end date seen so far is at most 30 days.
Gap arithmetic uses whole days and treats end dates as they are stored (no inclusive
end day is added), as the OHDSI scripts do.

CONDITION_ERA, per person_id and condition_concept_id (concept 0 is skipped):
  start = condition_start_date, end = condition_end_date, else start + 1 day.
  Occurrences are merged with the persistence window. The era runs from the
  earliest start to the latest end; condition_occurrence_count is the number of
  occurrences in it.

DRUG_ERA, per person_id and ingredient:
  1. Each drug_exposure drug_concept_id (concept 0 is skipped) becomes one row per
     ingredient (the caller passes the drug to ingredient pairs). Drugs without an
     ingredient are not in DRUG_ERA.
  2. Exposure end = drug_exposure_end_date, else start + days_supply, else
     start + 1 day. An end before the start is moved to the start.
  3. Exposures that overlap or touch (gap of 0 days) are merged into
     sub-exposures.
  4. Sub-exposures are merged with the 30 day persistence window into eras.
     drug_exposure_count is the number of exposures in the era and gap_days is the
     era length in days minus the days covered by its sub-exposures.

Where end-date conventions differ between OHDSI versions this module takes one:
drug_exposure_end_date is used as stored and days_supply is added to the start
without subtracting one day (the OHDSI scripts of CDM 5.3 and 5.4 do the same; some
older ETLs treat the end as the last covered day and add days_supply - 1). Lengths
are end minus start, so an exposure that starts and ends on one day covers 0 days.
The converter fills drug_exposure_end_date with the start date when the source has
no end date, so exposures from TriNetX normally end on their start day.

The functions return the era rows without the era ID; the caller numbers them.
"""

import polars as pl

from ..util import columns as col

PERSISTENCE_DAYS = 30

# Working columns: interval ends as days since 1970-01-01
_START = "_start"
_END = "_end"
_ERA = "_era"
_COUNT = "_count"
_COVERED = "_covered"
_NEW_KEY = "_new_key"
_SHIFTED_START = "_shifted_start"
_SHIFTED_END = "_shifted_end"


def _days(expr: pl.Expr) -> pl.Expr:
    """A date as days since 1970-01-01."""
    return expr.cast(pl.Int32)


def _merge_intervals(
    intervals: pl.DataFrame, keys: list[str], gap: int, aggregates: list[pl.Expr]
) -> pl.DataFrame:
    """Merge the intervals of each key whose gap to the earlier ones is <= gap days.

    An interval joins the running era of its key when its start is at most `gap`
    days after the latest end of the intervals before it; otherwise it opens a new
    era. The result has one row per era with the keys and the aggregates (which read
    the working columns), sorted by the keys and the era start.

    The latest end so far is a running maximum that restarts for every key. Window
    expressions per key (`cum_max().over(keys)`) are very slow with millions of small
    keys, so the days are shifted by a multiple of the day range per key: a later key
    then always has larger values than an earlier one, and one plain running maximum
    over the sorted rows gives the maximum within each key.
    """
    ordered = intervals.sort(*keys, _START)
    if ordered.is_empty():
        return ordered.group_by(keys).agg(aggregates)
    origin = ordered[_START].min()
    span = ordered[_END].max() - origin + 1
    new_key = pl.any_horizontal(
        [(pl.col(k) != pl.col(k).shift(1)).fill_null(True) for k in keys]
    )
    shift = new_key.cast(pl.Int64).cum_sum() * span - origin
    return (
        ordered.with_columns(
            new_key.alias(_NEW_KEY),
            (pl.col(_START).cast(pl.Int64) + shift).alias(_SHIFTED_START),
            (pl.col(_END).cast(pl.Int64) + shift).alias(_SHIFTED_END),
        )
        .with_columns(
            (
                pl.col(_NEW_KEY)
                | (
                    pl.col(_SHIFTED_START) - pl.col(_SHIFTED_END).cum_max().shift(1)
                    > gap
                ).fill_null(True)
            )
            .cast(pl.Int64)
            .cum_sum()
            .alias(_ERA)
        )
        .group_by(_ERA, maintain_order=True)
        .agg(*(pl.col(k).first() for k in keys), *aggregates)
        .drop(_ERA)
    )


def condition_eras(occurrences: pl.DataFrame) -> pl.DataFrame:
    """Condition eras (without IDs) of CONDITION_OCCURRENCE rows.

    Args:
        occurrences: Frame with person_id, condition_concept_id,
            condition_start_date and condition_end_date.

    Returns:
        Frame with the CONDITION_ERA columns except condition_era_id.
    """
    start = _days(pl.col(col.condition_start_date))
    intervals = occurrences.filter(pl.col(col.condition_concept_id) != 0).select(
        col.person_id,
        col.condition_concept_id,
        start.alias(_START),
        pl.max_horizontal(
            pl.coalesce(_days(pl.col(col.condition_end_date)), start + 1), start
        ).alias(_END),
        pl.lit(1, dtype=pl.Int64).alias(_COUNT),
    )
    eras = _merge_intervals(
        intervals,
        [col.person_id, col.condition_concept_id],
        PERSISTENCE_DAYS,
        [
            pl.col(_START).min(),
            pl.col(_END).max(),
            pl.col(_COUNT).sum(),
        ],
    )
    return eras.select(
        col.person_id,
        col.condition_concept_id,
        pl.col(_START).cast(pl.Date).alias(col.condition_era_start_date),
        pl.col(_END).cast(pl.Date).alias(col.condition_era_end_date),
        pl.col(_COUNT).alias(col.condition_occurrence_count),
    )


def drug_eras(exposures: pl.DataFrame, ingredients: pl.DataFrame) -> pl.DataFrame:
    """Drug eras (without IDs) of DRUG_EXPOSURE rows.

    Args:
        exposures: Frame with person_id, drug_concept_id, drug_exposure_start_date,
            drug_exposure_end_date and days_supply.
        ingredients: Frame with drug_concept_id and ingredient_concept_id: the
            ingredients of each drug concept (see eras.ingredient_map).

    Returns:
        Frame with the DRUG_ERA columns except drug_era_id.
    """
    start = _days(pl.col(col.drug_exposure_start_date))
    end = pl.coalesce(
        _days(pl.col(col.drug_exposure_end_date)),
        start + pl.col(col.days_supply).cast(pl.Int32),
        start + 1,
    )
    intervals = exposures.join(ingredients, on=col.drug_concept_id, how="inner").select(
        col.person_id,
        col.ingredient_concept_id,
        start.alias(_START),
        pl.max_horizontal(end, start).alias(_END),
        pl.lit(1, dtype=pl.Int64).alias(_COUNT),
    )
    keys = [col.person_id, col.ingredient_concept_id]
    sub_exposures = _merge_intervals(
        intervals,
        keys,
        0,
        [pl.col(_START).min(), pl.col(_END).max(), pl.col(_COUNT).sum()],
    )
    eras = _merge_intervals(
        sub_exposures,
        keys,
        PERSISTENCE_DAYS,
        [
            pl.col(_START).min(),
            pl.col(_END).max(),
            pl.col(_COUNT).sum(),
            (pl.col(_END) - pl.col(_START)).sum().alias(_COVERED),
        ],
    )
    return eras.select(
        col.person_id,
        pl.col(col.ingredient_concept_id).alias(col.drug_concept_id),
        pl.col(_START).cast(pl.Date).alias(col.drug_era_start_date),
        pl.col(_END).cast(pl.Date).alias(col.drug_era_end_date),
        pl.col(_COUNT).alias(col.drug_exposure_count),
        (pl.col(_END) - pl.col(_START) - pl.col(_COVERED)).alias(col.gap_days),
    )
