"""Build the optional CONDITION_ERA and DRUG_ERA tables from the merged output.

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
     ingredient: the ancestors in CONCEPT_ANCESTOR that are standard concepts of
     class Ingredient in RxNorm or RxNorm Extension. A drug that is an ingredient
     maps to itself (CONCEPT_ANCESTOR holds that self row with 0 levels of
     separation; the mapping adds it anyway, so a file without it still works).
     Drugs without an ingredient are not in DRUG_ERA.
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

The eras are built per range of person_id so that memory depends on the range size
(`batch_rows` source rows), not on the table size. The output tables are sorted by
person_id, so a range filter skips the Parquet row groups of other persons. IDs run
from 1 over the whole table in the order of person_id, concept ID and start date.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import polars as pl

from .ingest import DEFAULT_BATCH_ROWS
from .schema.omop import CONDITION_ERA_SCHEMA, DRUG_ERA_SCHEMA, TableSchema
from .omop_vocab.athena import athena_path, scan_athena, scan_concepts
from .omop_vocab.concept_table import table_path
from .util import columns as col
from .util import tables as tbl
from .util.parquet_io import sink_parquet, write_parquet

logger = logging.getLogger(__name__)

PERSISTENCE_DAYS = 30
# Folder of the era parts in the working directory
ERAS_DIR = "eras"
INGREDIENT_CLASS = "Ingredient"
STANDARD_CONCEPT = "S"
INGREDIENT_VOCABULARIES = ["RxNorm", "RxNorm Extension"]
ERA_TABLES = [tbl.omop_condition_era, tbl.omop_drug_era]
REQUIRED_VOCABULARY_FILES = [f"{tbl.athena_concept_ancestor}.csv"]

# Working columns: interval ends as days since 1970-01-01
_START = "_start"
_END = "_end"
_ERA = "_era"
_COUNT = "_count"
_COVERED = "_covered"
_RANGE = "_range"
_NEW_KEY = "_new_key"
_SHIFTED_START = "_shifted_start"
_SHIFTED_END = "_shifted_end"

EraBuilder = Callable[[pl.DataFrame], pl.DataFrame]


def missing_vocabulary_files(vocab_dir: Path) -> list[str]:
    """Return the Athena files that the eras need and vocab_dir lacks."""
    return [
        name for name in REQUIRED_VOCABULARY_FILES if not (vocab_dir / name).is_file()
    ]


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
            ingredients of each drug concept (see ingredient_map).

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


def ingredient_map(vocab_dir: Path, drug_concept_ids: pl.Series) -> pl.DataFrame:
    """Map drug concepts to their ingredients with the Athena vocabulary.

    CONCEPT_ANCESTOR (tens of millions of rows) is filtered while it streams from
    the file, to the given descendants and to the ingredient concepts, so only the
    matching pairs are held in memory.

    Args:
        vocab_dir: Athena folder with CONCEPT.csv and CONCEPT_ANCESTOR.csv.
        drug_concept_ids: Distinct non-zero drug_concept_id values of DRUG_EXPOSURE.

    Returns:
        Frame of unique (drug_concept_id, ingredient_concept_id) pairs; a drug that
        is an ingredient maps to itself.
    """
    schema = {col.drug_concept_id: pl.Int64, col.ingredient_concept_id: pl.Int64}
    if drug_concept_ids.is_empty():
        return pl.DataFrame(schema=schema)
    drugs = drug_concept_ids.cast(pl.Int64).rename(col.drug_concept_id).to_frame()
    ingredients = (
        scan_concepts(vocab_dir)
        .filter(
            (pl.col(col.standard_concept) == STANDARD_CONCEPT)
            & (pl.col(col.concept_class_id) == INGREDIENT_CLASS)
            & pl.col(col.vocabulary_id).is_in(INGREDIENT_VOCABULARIES)
        )
        .select(pl.col(col.concept_id).cast(pl.Int64).alias(col.ingredient_concept_id))
        .unique()
        .collect(engine="streaming")
    )
    ancestors = (
        scan_athena(athena_path(vocab_dir, tbl.athena_concept_ancestor))
        .select(
            pl.col(col.descendant_concept_id).cast(pl.Int64).alias(col.drug_concept_id),
            pl.col(col.ancestor_concept_id)
            .cast(pl.Int64)
            .alias(col.ingredient_concept_id),
        )
        .join(drugs.lazy(), on=col.drug_concept_id, how="semi")
        .join(ingredients.lazy(), on=col.ingredient_concept_id, how="semi")
        .collect(engine="streaming")
    )
    itself = drugs.join(
        ingredients,
        left_on=col.drug_concept_id,
        right_on=col.ingredient_concept_id,
        how="semi",
    ).with_columns(pl.col(col.drug_concept_id).alias(col.ingredient_concept_id))
    return pl.concat([ancestors, itself]).unique().sort(pl.all())


def person_ranges(persons: pl.LazyFrame, max_rows: int) -> list[tuple[int, int]]:
    """Split the persons of a table into ranges of about max_rows rows.

    Args:
        persons: Scan with a person_id column, one row per source row.
        max_rows: Row bound of a range; a range holds more only when one person
            alone has more rows.

    Returns:
        Inclusive (first, last) person_id of each range, in person_id order. A
        range covers every person_id between them.
    """
    per_person = (
        persons.select(pl.col(col.person_id).cast(pl.Int64))
        .group_by(col.person_id)
        .agg(pl.len().alias(col.rows))
        .sort(col.person_id)
        .collect(engine="streaming")
    )
    ranges = (
        per_person.with_columns(
            ((pl.col(col.rows).cum_sum() - pl.col(col.rows)) // max_rows).alias(_RANGE)
        )
        .group_by(_RANGE, maintain_order=True)
        .agg(pl.col(col.person_id).min().alias("_first"), pl.col(col.person_id).max())
    )
    return list(zip(ranges["_first"].to_list(), ranges[col.person_id].to_list()))


def _write_table(
    path: Path,
    parts_dir: Path,
    schema: TableSchema,
    source: pl.LazyFrame | None,
    build: EraBuilder,
    batch_rows: int,
) -> int:
    """Build one era table range by range, then concatenate the parts into path.

    Each part is numbered from the row count of the parts before it; ranges follow
    person_id order and every part is sorted, so the parts need no further sorting.

    Returns:
        Number of era rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)
    id_column = schema.primary_key
    assert id_column is not None
    offset = 0
    parts: list[Path] = []
    ranges = person_ranges(source, batch_rows) if source is not None else []
    for index, (first, last) in enumerate(ranges):
        rows = source.filter(pl.col(col.person_id).is_between(first, last)).collect(
            engine="streaming"
        )
        eras = build(rows).with_columns(
            (pl.int_range(pl.len(), dtype=pl.Int64) + offset + 1).alias(id_column)
        )
        eras = eras.select([pl.col(c.name).cast(c.dtype) for c in schema.columns])
        offset += eras.height
        part = parts_dir / f"{index:06d}.parquet"
        write_parquet(eras, part, temporary=True)
        parts.append(part)
        logger.debug(
            f"  {schema.name}: persons {first} to {last}, {eras.height:,} eras"
        )
    if parts:
        sink_parquet(pl.concat([pl.scan_parquet(p) for p in parts]), path)
    else:
        write_parquet(pl.DataFrame(schema=schema.dtype_map), path)
    shutil.rmtree(parts_dir, ignore_errors=True)
    return offset


def build_eras(
    output_dir: Path,
    vocab_dir: Path,
    work_dir: Path,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> dict[str, int]:
    """Write CONDITION_ERA and DRUG_ERA from the merged output tables.

    Reads output_dir/condition_occurrence and drug_exposure and writes
    output_dir/condition_era/condition_era.parquet and
    output_dir/drug_era/drug_era.parquet, sorted by person_id, with IDs from 1. A
    table whose source file is missing is written empty.

    Args:
        output_dir: Base output directory with the merged tables.
        vocab_dir: Athena folder with CONCEPT.csv and CONCEPT_ANCESTOR.csv.
        work_dir: Working directory; the parts of a table are written to
            <work_dir>/eras and removed after the table is complete.
        batch_rows: Upper bound of source rows per range of persons.

    Returns:
        Rows per era table, keyed by table name.

    Raises:
        FileNotFoundError: CONCEPT_ANCESTOR.csv is missing from vocab_dir.
    """
    missing = missing_vocabulary_files(vocab_dir)
    if missing:
        raise FileNotFoundError(f"{vocab_dir} is missing: {', '.join(missing)}")
    parts_dir = work_dir / ERAS_DIR
    counts: dict[str, int] = {}

    occurrence_path = table_path(output_dir, tbl.omop_condition_occurrence)
    occurrences = (
        pl.scan_parquet(occurrence_path).select(
            col.person_id,
            col.condition_concept_id,
            col.condition_start_date,
            col.condition_end_date,
        )
        if occurrence_path.exists()
        else None
    )
    counts[tbl.omop_condition_era] = _write_table(
        table_path(output_dir, tbl.omop_condition_era),
        parts_dir,
        CONDITION_ERA_SCHEMA,
        occurrences,
        condition_eras,
        batch_rows,
    )
    logger.info(f"  {tbl.omop_condition_era}: {counts[tbl.omop_condition_era]:,} rows")

    exposure_path = table_path(output_dir, tbl.omop_drug_exposure)
    exposures = (
        pl.scan_parquet(exposure_path)
        .select(
            col.person_id,
            col.drug_concept_id,
            col.drug_exposure_start_date,
            col.drug_exposure_end_date,
            col.days_supply,
        )
        .filter(pl.col(col.drug_concept_id).fill_null(0) != 0)
        if exposure_path.exists()
        else None
    )
    ingredients = pl.DataFrame(
        schema={col.drug_concept_id: pl.Int64, col.ingredient_concept_id: pl.Int64}
    )
    if exposures is not None:
        per_drug = (
            exposures.group_by(col.drug_concept_id)
            .agg(pl.len().alias(col.rows))
            .collect(engine="streaming")
        )
        ingredients = ingredient_map(vocab_dir, per_drug[col.drug_concept_id])
        total = per_drug[col.rows].sum()
        covered = per_drug.join(
            ingredients.select(col.drug_concept_id).unique(),
            on=col.drug_concept_id,
            how="semi",
        )[col.rows].sum()
        if total:
            logger.info(
                f"  {covered:,} of {total:,} drug exposures ({covered / total:.1%}) "
                "have an ingredient"
            )
    counts[tbl.omop_drug_era] = _write_table(
        table_path(output_dir, tbl.omop_drug_era),
        parts_dir,
        DRUG_ERA_SCHEMA,
        exposures,
        lambda rows: drug_eras(rows, ingredients),
        batch_rows,
    )
    logger.info(f"  {tbl.omop_drug_era}: {counts[tbl.omop_drug_era]:,} rows")
    return counts
