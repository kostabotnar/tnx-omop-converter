"""Stage 6 of the bounded memory pipeline: merge the parts of all batches.

Reads <work>/parts/ (see batch_transform) and writes the OMOP tables and the
excluded records to the output folder. Parts are concatenated in batch order without
sorting: batches are contiguous person_id ranges and every part is already sorted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from .batch_transform import COVERAGE_DIR, PARTS_DIR, output_tables
from ..omop_vocab.concept_table import table_path
from ..transformers.clinical import COVERAGE_PART_SCHEMA, COVERAGE_SCHEMA
from ..transformers.sources import SOURCE_ADAPTERS
from ..util import columns as col
from ..util import tables as tbl
from ..util.parquet_io import sink_parquet, write_parquet

logger = logging.getLogger(__name__)

COVERAGE_FILE = "coverage.csv"

# Columns of an excluded file of a table that has no source rows; the source columns
# are unknown then
_EXCLUDED_ONLY_SCHEMA = pl.Schema(
    {
        col.exclusion_reason: pl.String,
        col.target_domain_id: pl.String,
        col.source_concept_id: pl.Int64,
    }
)


def _part_files(work_dir: Path, folder: str) -> list[Path]:
    """Part files of a folder in batch order."""
    return sorted((work_dir / PARTS_DIR / folder).glob("*.parquet"))


def _scan_parts(files: list[Path]) -> pl.LazyFrame:
    return pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")


def _row_count(path: Path) -> int:
    return pl.scan_parquet(path).select(pl.len()).collect().item()


def merge_tables(work_dir: Path, output_dir: Path) -> dict[str, int]:
    """Concatenate the parts of each OMOP table into its output file.

    Tables without parts are written as typed empty frames, so every OMOP table
    (including OBSERVATION and DEVICE_EXPOSURE) exists in the output.

    Returns:
        Row count per table.
    """
    row_counts: dict[str, int] = {}
    for table_name, schema in output_tables().items():
        path = table_path(output_dir, table_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        files = _part_files(work_dir, table_name)
        if files:
            sink_parquet(_scan_parts(files), path)
            row_counts[table_name] = _row_count(path)
        else:
            write_parquet(pl.DataFrame(schema=schema.dtype_map), path)
            row_counts[table_name] = 0
        logger.info(f"  {table_name}: {row_counts[table_name]:,} rows")
    return row_counts


def merge_excluded(work_dir: Path, output_dir: Path, tnx_tables: list[str]) -> None:
    """Write output_dir/excluded/<tnx table>.parquet for every clinical table.

    Args:
        work_dir: Working directory with the parts.
        output_dir: Output folder; its excluded folder must exist.
        tnx_tables: TriNetX tables present in the input.
    """
    excluded_dir = output_dir / tbl.excluded
    for tnx_table in tnx_tables:
        if tnx_table not in SOURCE_ADAPTERS:
            continue
        path = excluded_dir / f"{tnx_table}.parquet"
        files = _part_files(work_dir, f"{tbl.excluded}/{tnx_table}")
        if files:
            sink_parquet(_scan_parts(files), path)
        else:
            write_parquet(pl.DataFrame(schema=_EXCLUDED_ONLY_SCHEMA), path)


def _percent(part: pl.Expr, total: pl.Expr) -> pl.Expr:
    return (part * 100 / total).round(2)


def merge_coverage(work_dir: Path, output_dir: Path) -> pl.DataFrame:
    """Sum the coverage parts of all batches, log the summary and write coverage.csv.

    rows_pct and codes_pct are shares of the output rows and of the distinct
    (code_system, code) pairs of the TriNetX table. A code mapping to targets in
    several tables counts in each, so codes_pct of a table can add up to more than 100.
    """
    files = _part_files(work_dir, COVERAGE_DIR)
    parts = _scan_parts(files) if files else pl.LazyFrame(schema=COVERAGE_PART_SCHEMA)
    per_code = (
        parts.group_by(col.tnx_table, col.code_system, col.code, col.outcome)
        .agg(pl.col(col.rows).sum())
        .collect()
    )
    code_totals = (
        per_code.drop_nulls(col.code)
        .unique([col.tnx_table, col.code_system, col.code])
        .group_by(col.tnx_table)
        .len("_table_codes")
    )
    coverage = (
        per_code.group_by(col.tnx_table, col.code_system, col.outcome)
        .agg(
            pl.col(col.rows).sum(),
            pl.col(col.code).drop_nulls().n_unique().alias(col.codes),
        )
        .join(code_totals, on=col.tnx_table, how="left")
        .with_columns(
            _percent(
                pl.col(col.rows), pl.col(col.rows).sum().over(col.tnx_table)
            ).alias(col.rows_pct),
            _percent(pl.col(col.codes), pl.col("_table_codes"))
            .fill_null(0.0)
            .alias(col.codes_pct),
        )
        .select(COVERAGE_SCHEMA.names())
        .cast(dict(COVERAGE_SCHEMA))
        .sort(col.tnx_table, col.code_system, col.outcome, nulls_last=True)
    )
    coverage.write_csv(output_dir / COVERAGE_FILE)

    logger.info("\nVocabulary coverage (output rows by TriNetX table and outcome):")
    summary = (
        coverage.group_by(col.tnx_table, col.outcome)
        .agg(pl.col(col.rows).sum())
        .sort([col.tnx_table, col.rows, col.outcome], descending=[False, True, False])
    )
    for tnx_table, outcome, rows in summary.iter_rows():
        logger.info(f"  {tnx_table}: {outcome} {rows:,}")
    return coverage
