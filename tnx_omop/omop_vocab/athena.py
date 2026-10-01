"""Read the Athena OMOP vocabulary download (tab-separated CSV files)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl

from ..util import columns as col
from ..util import tables as tbl

MAPS_TO = "Maps to"
NONE_VOCABULARY_ID = "None"
ATHENA_DATE_FORMAT = "%Y%m%d"

REQUIRED_FILES: List[str] = [
    f"{tbl.athena_concept}.csv",
    f"{tbl.athena_concept_relationship}.csv",
    f"{tbl.athena_vocabulary}.csv",
]

CONCEPT_COLUMNS: List[str] = [
    col.concept_id,
    col.concept_name,
    col.domain_id,
    col.vocabulary_id,
    col.concept_class_id,
    col.standard_concept,
    col.concept_code,
    col.valid_start_date,
    col.valid_end_date,
    col.invalid_reason,
]


def athena_path(vocab_dir: Path, stem: str) -> Path:
    """Return the path of an Athena file given its stem (for example "CONCEPT")."""
    return Path(vocab_dir) / f"{stem}.csv"


def missing_files(vocab_dir: Path) -> List[str]:
    """Return the required Athena file names that are not present in vocab_dir."""
    return [name for name in REQUIRED_FILES if not (Path(vocab_dir) / name).is_file()]


def scan_athena(path: Path) -> pl.LazyFrame:
    """Lazily scan one Athena file with every column as a string."""
    # Athena files are tab separated with unescaped quotes inside names.
    return pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False)


def scan_concepts(vocab_dir: Path) -> pl.LazyFrame:
    """Scan CONCEPT.csv plus CONCEPT_CPT4.csv when present, all columns as strings.

    CPT4 concepts ship in CONCEPT_CPT4.csv until the Athena cpt tool merges them into
    CONCEPT.csv, after which they appear in both. Callers must deduplicate by
    concept_id after filtering (see unique_concepts).
    """
    frames = [
        scan_athena(athena_path(vocab_dir, tbl.athena_concept)).select(CONCEPT_COLUMNS)
    ]
    cpt4_path = athena_path(vocab_dir, tbl.athena_concept_cpt4)
    if cpt4_path.is_file():
        frames.append(scan_athena(cpt4_path).select(CONCEPT_COLUMNS))
    return pl.concat(frames)


def unique_concepts(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Keep one row per concept_id, preferring a non-empty concept_name.

    Among rows of the same concept_id with the same preference the first in file
    order is kept. The row order of the result is not defined. The named and the
    unnamed rows are deduplicated apart, which needs far less memory on the full
    CONCEPT table than sorting all rows by name presence.
    """
    has_name = pl.col(col.concept_name).fill_null("") != ""
    named = lf.filter(has_name).unique(
        col.concept_id, keep="first", maintain_order=True
    )
    unnamed = (
        lf.filter(~has_name)
        .unique(col.concept_id, keep="first", maintain_order=True)
        .join(named.select(col.concept_id), on=col.concept_id, how="anti")
    )
    return pl.concat([named, unnamed])


def typed_concepts(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Cast string CONCEPT columns to OMOP types (Int64 id, Date validity)."""
    return lf.with_columns(
        pl.col(col.concept_id).cast(pl.Int64),
        pl.col(col.concept_name).fill_null(""),
        pl.col(col.valid_start_date).str.strptime(pl.Date, ATHENA_DATE_FORMAT),
        pl.col(col.valid_end_date).str.strptime(pl.Date, ATHENA_DATE_FORMAT),
    )


def scan_maps_to(vocab_dir: Path) -> pl.LazyFrame:
    """Scan valid "Maps to" rows as (concept_id_1, concept_id_2) Int64 pairs."""
    return (
        scan_athena(athena_path(vocab_dir, tbl.athena_concept_relationship))
        .filter(
            (pl.col(col.relationship_id) == MAPS_TO)
            & (pl.col(col.invalid_reason).fill_null("") == "")
        )
        .select(
            pl.col(col.concept_id_1).cast(pl.Int64),
            pl.col(col.concept_id_2).cast(pl.Int64),
        )
    )


def read_vocabulary_version(vocab_dir: Path) -> str:
    """Return the vocabulary release from the VOCABULARY row with vocabulary_id "None".

    Raises:
        ValueError: If VOCABULARY.csv has no such row or its version is empty.
    """
    versions = (
        scan_athena(athena_path(vocab_dir, tbl.athena_vocabulary))
        .filter(pl.col(col.vocabulary_id) == NONE_VOCABULARY_ID)
        .select(col.vocabulary_version)
        .collect()
        .get_column(col.vocabulary_version)
        .drop_nulls()
    )
    if versions.is_empty() or not versions[0]:
        raise ValueError(
            f"{athena_path(vocab_dir, tbl.athena_vocabulary)} has no vocabulary_version "
            f"for vocabulary_id '{NONE_VOCABULARY_ID}'"
        )
    return str(versions[0])
