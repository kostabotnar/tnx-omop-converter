"""Write the optional CONDITION_ERA and DRUG_ERA tables from the merged output.

The era logic (persistence window, ingredient rules, end-date conventions) is in
transformers/eras.py. This module reads the merged tables, maps drugs to
ingredients with CONCEPT_ANCESTOR, and writes the eras range by range.

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
from .transformers.eras import condition_eras, drug_eras
from .util import columns as col
from .util import tables as tbl
from .util.parquet_io import sink_parquet, write_parquet

logger = logging.getLogger(__name__)

# Folder of the era parts in the working directory
ERAS_DIR = "eras"
INGREDIENT_CLASS = "Ingredient"
STANDARD_CONCEPT = "S"
INGREDIENT_VOCABULARIES = ["RxNorm", "RxNorm Extension"]
ERA_TABLES = [tbl.omop_condition_era, tbl.omop_drug_era]
REQUIRED_VOCABULARY_FILES = [f"{tbl.athena_concept_ancestor}.csv"]

_RANGE = "_range"

EraBuilder = Callable[[pl.DataFrame], pl.DataFrame]


def missing_vocabulary_files(vocab_dir: Path) -> list[str]:
    """Return the Athena files that the eras need and vocab_dir lacks."""
    return [
        name for name in REQUIRED_VOCABULARY_FILES if not (vocab_dir / name).is_file()
    ]


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
