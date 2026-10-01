"""Export the full Athena vocabulary tables into the output for OHDSI tools.

ATLAS, HADES and the Data Quality Dashboard need the standardized vocabulary
tables next to the clinical tables. The converter's own CONCEPT holds only the
concepts the output references; `export_vocabulary` replaces it with the full
table and adds the other CDM 5.4 vocabulary tables, one Parquet file each in the
layout of the other output tables (`concept_table.table_path`).

Only CONCEPT is held in memory while it is deduplicated; every other table is
streamed from its Athena file to its Parquet file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl

from ..omop_schema import (
    CONCEPT_ANCESTOR_SCHEMA,
    CONCEPT_CLASS_SCHEMA,
    CONCEPT_RELATIONSHIP_SCHEMA,
    CONCEPT_SCHEMA,
    CONCEPT_SYNONYM_SCHEMA,
    DOMAIN_SCHEMA,
    DRUG_STRENGTH_SCHEMA,
    RELATIONSHIP_SCHEMA,
    VOCABULARY_SCHEMA,
    TableSchema,
)
from ..util import tables as tbl
from ..util.parquet_io import sink_parquet
from .athena import (
    ATHENA_DATE_FORMAT,
    REQUIRED_FILES,
    athena_path,
    scan_athena,
    scan_concepts,
)
from .concept_table import build_concepts, table_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VocabularyTable:
    """One exported table: its Athena file stem and its CDM schema."""

    athena_stem: str
    schema: TableSchema


# CONCEPT is listed first; it is built by build_concepts instead of a plain cast.
EXPORTED_TABLES: list[VocabularyTable] = [
    VocabularyTable(tbl.athena_concept, CONCEPT_SCHEMA),
    VocabularyTable(tbl.athena_concept_relationship, CONCEPT_RELATIONSHIP_SCHEMA),
    VocabularyTable(tbl.athena_concept_ancestor, CONCEPT_ANCESTOR_SCHEMA),
    VocabularyTable(tbl.athena_vocabulary, VOCABULARY_SCHEMA),
    VocabularyTable(tbl.athena_domain, DOMAIN_SCHEMA),
    VocabularyTable(tbl.athena_concept_class, CONCEPT_CLASS_SCHEMA),
    VocabularyTable(tbl.athena_relationship, RELATIONSHIP_SCHEMA),
    VocabularyTable(tbl.athena_concept_synonym, CONCEPT_SYNONYM_SCHEMA),
    VocabularyTable(tbl.athena_drug_strength, DRUG_STRENGTH_SCHEMA),
]


def typed_table(lf: pl.LazyFrame, schema: TableSchema) -> pl.LazyFrame:
    """Cast the string columns of an Athena file to the types of schema.

    Dates are YYYYMMDD strings in Athena. Columns missing from the file are an
    error, extra columns are dropped.
    """
    return lf.select(
        [
            pl.col(c.name).str.strptime(pl.Date, ATHENA_DATE_FORMAT)
            if c.dtype == pl.Date
            else pl.col(c.name).cast(c.dtype)
            for c in schema.columns
        ]
    )


def _write_atomically(lf: pl.LazyFrame, path: Path) -> int:
    """Stream lf to path through a temporary file and return the row count.

    The file at path is replaced only when the whole table was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        sink_parquet(lf, temp)
        rows = pl.scan_parquet(temp).select(pl.len()).collect().item()
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return rows


def export_vocabulary(
    output_dir: Path, vocab_dir: Path, terminology: Optional[pl.DataFrame] = None
) -> dict[str, int]:
    """Write the full Athena vocabulary tables into output_dir.

    CONCEPT replaces the reduced table and is built like it (concept_table.
    build_concepts: CONCEPT.csv plus CONCEPT_CPT4.csv, one row per concept_id,
    empty names filled), not sorted: sorting the full table costs gigabytes of memory. CONCEPT_RELATIONSHIP and VOCABULARY
    are required Athena files; a missing file of another table is logged and
    skipped. Every file is written through a temporary file.

    Args:
        output_dir: Base output directory.
        vocab_dir: Folder with the Athena CSV files.
        terminology: Output of concept_table.load_terminology(), used to name
            concepts whose Athena name is empty (CPT4).

    Returns:
        Rows per written table, keyed by lowercase table name.

    Raises:
        FileNotFoundError: A required Athena file is missing.
    """
    for name in REQUIRED_FILES:
        if not (Path(vocab_dir) / name).is_file():
            raise FileNotFoundError(f"{Path(vocab_dir) / name} not found")

    counts: dict[str, int] = {}
    for table in EXPORTED_TABLES:
        source = athena_path(vocab_dir, table.athena_stem)
        if table.athena_stem == tbl.athena_concept:
            lf = build_concepts(scan_concepts(vocab_dir), terminology)
        elif not source.is_file():
            logger.info(f"  {source.name} not found, {table.schema.name} skipped")
            continue
        else:
            lf = typed_table(scan_athena(source), table.schema)
        counts[table.schema.name] = _write_atomically(
            lf, table_path(output_dir, table.schema.name)
        )
        logger.info(f"  {table.schema.name}: {counts[table.schema.name]:,} rows")
    return counts
