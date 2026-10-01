"""Create the OMOP CONCEPT table from the Athena concepts the output references."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import polars as pl

from ..util import columns as col
from ..util import tables as tbl
from ..util.parquet_io import write_parquet
from .athena import scan_concepts, typed_concepts, unique_concepts
from ..util.concept_mappings import SOURCE_VOCABULARY_MAP
from ..schema.omop import CONCEPT_SCHEMA, ERA_SCHEMAS, OMOP_SCHEMAS

logger = logging.getLogger(__name__)

CONCEPT_ID_SUFFIX = "_concept_id"
# CDM 5.4 column limit; the OHDSI vocabulary download truncates the same way
MAX_CONCEPT_NAME_LENGTH = 255
# Number of missing concept IDs listed in the warning
_MISSING_SAMPLE_SIZE = 20

TERMINOLOGY_SCHEMA = {
    col.code_system: pl.Utf8,
    col.code: pl.Utf8,
    col.code_description: pl.Utf8,
}


def load_terminology(data_dirs: Iterable[Path]) -> Optional[pl.DataFrame]:
    """Load code descriptions from standardized_terminology.csv of each source.

    Returns:
        DataFrame with code_system, code, code_description unique on
        (code_system, code), or None if no source has the file.
    """
    frames = []
    for data_dir in data_dirs:
        path = data_dir / f"{tbl.tnx_standardized_terminology}.csv"
        if not path.exists():
            continue
        # Read as strings so codes with leading zeros keep their exact form
        frames.append(
            pl.scan_csv(path, infer_schema=False).select(list(TERMINOLOGY_SCHEMA))
        )

    if not frames:
        return None

    return (
        pl.concat(frames)
        .filter(
            pl.col(col.code).is_not_null() & pl.col(col.code_description).is_not_null()
        )
        .unique(subset=[col.code_system, col.code], keep="first")
        .collect()
    )


def table_path(output_dir: Path, table_name: str) -> Path:
    """Return output_dir/<table>/<table>.parquet (lowercase table name)."""
    table_lower = table_name.lower()
    return Path(output_dir) / table_lower / f"{table_lower}.parquet"


def referenced_concept_ids(output_dir: Path) -> pl.Series:
    """Return the sorted unique non-zero values of every *_concept_id column.

    Covers every OMOP table in OMOP_SCHEMAS except CONCEPT, and the era tables, that
    exist under output_dir, including *_source_concept_id, type, unit, route, value, status,
    visit, gender, race and ethnicity columns.
    """
    parts: List[pl.LazyFrame] = []
    for table_name in [*OMOP_SCHEMAS, *ERA_SCHEMAS]:
        path = table_path(output_dir, table_name)
        if table_name == tbl.omop_concept or not path.exists():
            continue
        lf = pl.scan_parquet(path)
        for name in lf.collect_schema().names():
            if name.endswith(CONCEPT_ID_SUFFIX):
                parts.append(
                    lf.select(pl.col(name).cast(pl.Int64).alias(col.concept_id))
                )
    if not parts:
        return pl.Series(col.concept_id, [], dtype=pl.Int64)
    return (
        pl.concat(parts)
        .filter(pl.col(col.concept_id).is_not_null() & (pl.col(col.concept_id) != 0))
        .unique()
        .sort(col.concept_id)
        .collect(engine="streaming")
        .get_column(col.concept_id)
    )


def _vocabulary_code_systems() -> pl.DataFrame:
    """Invert the source vocabulary config: vocabulary_id -> TriNetX code_system."""
    pairs = sorted(
        {
            (vocabulary_id, code_system)
            for systems in SOURCE_VOCABULARY_MAP.values()
            for code_system, vocabulary_id in systems.items()
            if vocabulary_id is not None
        }
    )
    return pl.DataFrame(
        pairs,
        schema={col.vocabulary_id: pl.Utf8, col.code_system: pl.Utf8},
        orient="row",
    )


def _code_descriptions(
    empty: pl.LazyFrame, terminology: Optional[pl.DataFrame]
) -> pl.LazyFrame:
    """Name for each concept of `empty` (rows with an empty concept_name).

    Uses the TriNetX standardized_terminology description of the concept code in
    the code system the vocabulary maps from. Concepts without a description are
    not in the result.
    """
    if terminology is None or terminology.is_empty():
        return pl.LazyFrame(
            schema={col.concept_id: pl.Int64, col.code_description: pl.Utf8}
        )
    return (
        empty.select(col.concept_id, col.vocabulary_id, col.concept_code)
        .join(_vocabulary_code_systems().lazy(), on=col.vocabulary_id, how="inner")
        .join(
            terminology.lazy().select(
                col.code_system,
                pl.col(col.code).alias(col.concept_code),
                col.code_description,
            ),
            on=[col.code_system, col.concept_code],
            how="inner",
        )
        .sort(col.concept_id, col.code_system)
        .unique(col.concept_id, keep="first", maintain_order=True)
        .select(col.concept_id, col.code_description)
    )


def build_concepts(
    raw: pl.LazyFrame, terminology: Optional[pl.DataFrame] = None
) -> pl.LazyFrame:
    """Turn Athena concept rows (all strings, see athena.scan_concepts) into CONCEPT.

    Keeps one row per concept_id (unique_concepts), casts to the CONCEPT schema
    and names the concepts with an empty concept_name (CPT4 in Athena): the
    TriNetX standardized_terminology description of the concept code in the code
    system the vocabulary maps from, else the concept code. The rows are not
    sorted (sorting the full table needs gigabytes of memory). Every CONCEPT table
    of the output is built here, so a concept has the same row in the reduced and in
    the exported full table.

    Args:
        raw: Athena concept rows, possibly filtered to some concept IDs.
        terminology: Output of load_terminology(), or None.
    """
    raw = raw.with_columns(pl.col(col.concept_id).cast(pl.Int64, strict=False)).filter(
        pl.col(col.concept_id).is_not_null()
    )
    name = pl.col(col.concept_name)
    empty = raw.filter(name.fill_null("") == "")
    # The descriptions are few (empty names are CPT4 only), so they are collected and
    # looked up by value: a join with the full table would cost gigabytes of memory.
    descriptions = _code_descriptions(empty, terminology).collect(engine="streaming")
    fallback = pl.col(col.concept_code)
    if not descriptions.is_empty():
        fallback = pl.coalesce(
            pl.col(col.concept_id).replace_strict(
                descriptions[col.concept_id],
                descriptions[col.code_description],
                default=None,
            ),
            fallback,
        )
    return (
        typed_concepts(unique_concepts(raw))
        .with_columns(
            pl.when(name == "")
            .then(fallback)
            .otherwise(name)
            .str.slice(0, MAX_CONCEPT_NAME_LENGTH)
            .alias(col.concept_name)
        )
        .select([pl.col(c.name).cast(c.dtype) for c in CONCEPT_SCHEMA.columns])
    )


def create_concept_table(
    output_dir: Path, vocab_dir: Path, terminology: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    """Write the CONCEPT rows of every concept ID referenced by the output tables.

    Reads output_dir/<table>/<table>.parquet for every OMOP table, looks the IDs up
    in the Athena CONCEPT files and writes output_dir/concept/concept.parquet.
    IDs missing from Athena are logged and left out.

    Args:
        output_dir: Base output directory containing the OMOP tables.
        vocab_dir: Folder with the Athena CSV files.
        terminology: Output of load_terminology(), used to name concepts whose
            Athena name is empty (CPT4).

    Returns:
        CONCEPT DataFrame sorted by concept_id.
    """
    ids = referenced_concept_ids(output_dir)
    id_keys = ids.to_frame().lazy()
    concepts = (
        build_concepts(
            scan_concepts(vocab_dir)
            .with_columns(pl.col(col.concept_id).cast(pl.Int64, strict=False))
            .join(id_keys, on=col.concept_id, how="semi"),
            terminology,
        )
        .sort(col.concept_id)
        .collect(engine="streaming")
    )

    missing = (
        ids.to_frame()
        .join(concepts.select(col.concept_id), on=col.concept_id, how="anti")
        .get_column(col.concept_id)
    )
    if not missing.is_empty():
        logger.warning(
            "%d referenced concept IDs are not in the Athena vocabulary: %s",
            missing.len(),
            ", ".join(str(i) for i in missing.head(_MISSING_SAMPLE_SIZE).to_list()),
        )

    path = table_path(output_dir, tbl.omop_concept)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(concepts, path)
    return concepts
