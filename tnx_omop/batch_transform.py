"""Stage 5 of the bounded memory pipeline: clean and transform one batch at a time.

Reads <work>/batches/<table>/<batch>.parquet (see ingest) and writes

    <work>/parts/<omop table>/<batch:05d>.parquet           sorted, record IDs numbered
    <work>/parts/excluded/<tnx table>/<batch:05d>.parquet   records without a mapping
    <work>/parts/coverage/<batch:05d>.parquet                coverage rows of the batch

Atomicity: a batch is written to <work>/scratch/<batch:05d>/ and its part files are
then moved into parts/ one by one, the coverage part last. The coverage part is the
completion marker: a batch is finished if and only if parts/coverage/<batch>.parquet
exists. A crash leaves at most a batch without its coverage part; transform_batch
removes such leftovers when it runs the batch again.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable

import polars as pl

from .cleaning import clean_batch
from .domains import EVENT_CONCEPT_COLUMNS, EVENT_DATE_COLUMNS
from .ingest import IngestResult, scan_batch
from .omop_schema import OMOP_SCHEMAS, TableSchema
from .omop_vocab.vocabulary import VocabularyLookup
from .transformers import (
    transform_clinical,
    transform_death,
    transform_person,
    transform_visit,
)
from .transformers.clinical import COVERAGE_PART_SCHEMA, record_coverage
from .transformers.sources import SOURCE_ADAPTERS
from .util import columns as col
from .util import tables as tbl
from .util.id_generator import add_visit_ids, with_record_ids
from .util.parquet_io import sink_parquet, write_parquet

logger = logging.getLogger(__name__)

TableFrames = dict[str, pl.DataFrame]

PARTS_DIR = "parts"
SCRATCH_DIR = "scratch"
COVERAGE_DIR = "coverage"
_TASKS_DIR = "tasks"


@dataclass(frozen=True)
class IdCounters:
    """ID counters carried from one batch to the next.

    Attributes:
        last_visit_id: Last visit_occurrence_id assigned.
        record_counts: OMOP event table -> number of rows written so far, which is
            the last record ID of the table.
    """

    last_visit_id: int = 0
    record_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class TransformTask:
    """One transformation: transform_fn(*inputs) returns frames keyed by OMOP table.

    The key tables.excluded holds records excluded from the OMOP tables.
    coverage_fn(*inputs), when given, returns rows with COVERAGE_PART_SCHEMA.
    """

    name: str
    transform_fn: Callable[..., TableFrames]
    inputs: list[Any]
    coverage_fn: Callable[..., pl.DataFrame] | None = None


@dataclass
class TaskResult:
    """Row counts per OMOP table (and tables.excluded) and coverage of one task."""

    name: str
    row_counts: dict[str, int] = field(default_factory=dict)
    coverage: pl.DataFrame | None = None


def part_path(parts_dir: Path, folder: str, batch: int) -> Path:
    """Path of the part file of one batch, in <parts_dir>/<folder>/."""
    return parts_dir / folder / f"{batch:05d}.parquet"


def _single_table(table_name: str, fn: Callable[[pl.DataFrame], pl.DataFrame]):
    def run(df: pl.DataFrame) -> TableFrames:
        return {table_name: fn(df)}

    return run


def _execute_task(
    task: TransformTask, tasks_dir: Path, excluded_dir: Path
) -> TaskResult:
    """Run one task and write its frames as parquet files.

    OMOP frames go to tasks_dir/<table>/<task>.parquet and excluded records to
    excluded_dir/<task>.parquet, so no task output stays in memory after the task.
    """
    result = TaskResult(task.name)
    if task.coverage_fn is not None:
        result.coverage = task.coverage_fn(*task.inputs)
    frames = task.transform_fn(*task.inputs)
    # Drop the input references so a large source table can be freed early.
    task.inputs.clear()

    for table_name, df in frames.items():
        result.row_counts[table_name] = df.height
        if table_name == tbl.excluded:
            # A fixed row order keeps the output independent of the batch size
            df = df.sort(
                [col.person_id, *(c for c in df.columns if c != col.person_id)],
                nulls_last=True,
            )
            write_parquet(df, excluded_dir / f"{task.name}.parquet", temporary=True)
        elif df.height > 0:
            table_dir = tasks_dir / table_name
            table_dir.mkdir(parents=True, exist_ok=True)
            write_parquet(df, table_dir / f"{task.name}.parquet", temporary=True)
    return result


def _run_transformations(
    tasks: list[TransformTask], tasks_dir: Path, excluded_dir: Path
) -> list[TaskResult]:
    """Execute the transformation tasks one after another.

    Each task gets all Polars threads, and only one task's tables are in memory.
    """
    results: list[TaskResult] = []
    # Pop each task so its inputs are not referenced once it has run.
    while tasks:
        results.append(_execute_task(tasks.pop(0), tasks_dir, excluded_dir))
    return sorted(results, key=lambda r: r.name)


def _build_tasks(
    tables: dict[str, pl.DataFrame], lookup: VocabularyLookup
) -> list[TransformTask]:
    """Create one task per available TriNetX table, taking the tables out of the dict."""
    patient = tables.pop(tbl.tnx_patient)
    tasks = [
        TransformTask(
            tbl.omop_person, _single_table(tbl.omop_person, transform_person), [patient]
        ),
        TransformTask(
            tbl.omop_death, _single_table(tbl.omop_death, transform_death), [patient]
        ),
    ]
    encounter = tables.pop(tbl.tnx_encounter, None)
    if encounter is not None:
        tasks.append(
            TransformTask(
                tbl.omop_visit_occurrence,
                _single_table(tbl.omop_visit_occurrence, transform_visit),
                [encounter],
            )
        )
    for tnx_table in SOURCE_ADAPTERS:
        df = tables.pop(tnx_table, None)
        if df is None:
            continue
        tasks.append(
            TransformTask(
                tnx_table,
                partial(transform_clinical, tnx_table=tnx_table, lookup=lookup),
                [df],
                coverage_fn=partial(
                    record_coverage, tnx_table=tnx_table, lookup=lookup
                ),
            )
        )
    return tasks


def order_table(
    lf: pl.LazyFrame, table_name: str, schema: TableSchema, offset: int = 0
) -> pl.LazyFrame:
    """Sort a table by person_id and number its record IDs where the table has them.

    Event tables sort by person_id, event date, main concept ID and then every
    other column, so the order (and the IDs) do not depend on the order of the
    rows. Record IDs continue after offset rows. PERSON, DEATH and VISIT_OCCURRENCE
    already carry their IDs and sort by person_id, primary key and the other columns.
    """
    primary_key = schema.primary_key
    if table_name not in EVENT_CONCEPT_COLUMNS:
        return lf.sort(
            list(dict.fromkeys([col.person_id, primary_key, *schema.column_names])),
            nulls_last=True,
        )
    leading = [
        col.person_id,
        EVENT_DATE_COLUMNS[table_name],
        EVENT_CONCEPT_COLUMNS[table_name],
    ]
    rest = [c for c in schema.column_names if c != primary_key and c not in leading]
    return with_record_ids(lf, primary_key, leading + rest, offset)


def output_tables() -> dict[str, TableSchema]:
    """OMOP tables written by the transform tasks (not the post-task tables)."""
    post_task = {tbl.omop_concept, tbl.omop_observation_period, tbl.omop_cdm_source}
    return {t: s for t, s in OMOP_SCHEMAS.items() if t not in post_task}


def _load_batch(result: IngestResult, batch: int) -> dict[str, pl.DataFrame]:
    tables = {}
    for table in result.tables:
        lf = scan_batch(result.work_dir, table, batch)
        if lf is not None:
            tables[table] = lf.collect()
    return tables


def _remove_batch_leftovers(work_dir: Path, batch: int) -> None:
    """Delete the scratch folder and part files of a batch that did not finish."""
    shutil.rmtree(work_dir / SCRATCH_DIR / f"{batch:05d}", ignore_errors=True)
    # Look in each part folder instead of globbing parts/, which walks every file
    parts_dir = work_dir / PARTS_DIR
    folders = [*output_tables(), COVERAGE_DIR]
    folders += [f"{tbl.excluded}/{table}" for table in SOURCE_ADAPTERS]
    for folder in folders:
        part_path(parts_dir, folder, batch).unlink(missing_ok=True)


def _commit_batch(work_dir: Path, batch: int) -> None:
    """Move the finished part files of a batch into parts/, coverage part last."""
    scratch = work_dir / SCRATCH_DIR / f"{batch:05d}"
    staged = scratch / PARTS_DIR
    marker = part_path(staged, COVERAGE_DIR, batch)
    for path in sorted(staged.rglob("*.parquet"), key=lambda p: p == marker):
        target = work_dir / PARTS_DIR / path.relative_to(staged)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)
    shutil.rmtree(scratch)
    try:
        scratch.parent.rmdir()
    except OSError:
        pass


def _is_finished(work_dir: Path, batch: int) -> bool:
    return part_path(work_dir / PARTS_DIR, COVERAGE_DIR, batch).exists()


def completed_batches(work_dir: Path, batch_count: int) -> int:
    """Number of leading batches that are finished, to resume stage 5 after them.

    A batch is finished if and only if its coverage part exists. Parts of batches
    after the first unfinished one (which a normal run does not leave) are removed,
    so those batches run again.

    Args:
        work_dir: Working directory with parts/.
        batch_count: Number of batches of the run.

    Returns:
        The first batch that has to run.
    """
    first = 0
    while first < batch_count and _is_finished(work_dir, first):
        first += 1
    for batch in range(first + 1, batch_count):
        if _is_finished(work_dir, batch):
            _remove_batch_leftovers(work_dir, batch)
    return first


def counters_from_parts(work_dir: Path, batch_count: int) -> IdCounters:
    """ID counters after batches 0..batch_count-1, read from their part files.

    Equal to the counters transform_batch returns for the last of those batches:
    the last visit ID is the largest visit_occurrence_id of the visit parts, and the
    count of an event table is the number of rows in its parts. Only Parquet
    metadata and one column are read.
    """
    parts_dir = work_dir / PARTS_DIR

    def files(table: str) -> list[Path]:
        paths = (part_path(parts_dir, table, b) for b in range(batch_count))
        return [p for p in paths if p.exists()]

    visit_ids = [
        pl.scan_parquet(f)
        .select(pl.col(col.visit_occurrence_id).max())
        .collect()
        .item()
        for f in files(tbl.omop_visit_occurrence)
    ]
    record_counts = {}
    for table in EVENT_CONCEPT_COLUMNS:
        paths = files(table)
        if paths:
            record_counts[table] = sum(
                pl.scan_parquet(f).select(pl.len()).collect().item() for f in paths
            )
    return IdCounters(
        last_visit_id=max((v for v in visit_ids if v is not None), default=0),
        record_counts=record_counts,
    )


def transform_batch(
    result: IngestResult, batch: int, lookup: VocabularyLookup, counters: IdCounters
) -> tuple[IdCounters, dict[str, int]]:
    """Clean and transform the persons of one batch and write its part files.

    Args:
        result: Finished ingest stages.
        batch: Batch number, 0..result.batch_count-1. Batches must run in order, each
            with the counters returned by the previous one.
        lookup: Vocabulary lookup of all batches.
        counters: ID counters after the previous batch.

    Returns:
        (counters, row_counts): the counters after this batch, and the rows written
        per OMOP table plus tables.excluded.
    """
    work_dir = result.work_dir
    _remove_batch_leftovers(work_dir, batch)
    scratch = work_dir / SCRATCH_DIR / f"{batch:05d}"
    tasks_dir = scratch / _TASKS_DIR
    staged = scratch / PARTS_DIR
    excluded_dir = scratch / tbl.excluded
    excluded_dir.mkdir(parents=True)

    tables, last_visit_id = add_visit_ids(
        clean_batch(_load_batch(result, batch)), counters.last_visit_id
    )
    tasks = _build_tasks(tables, lookup)
    del tables
    results = _run_transformations(tasks, tasks_dir, excluded_dir)

    row_counts: dict[str, int] = {}
    record_counts = dict(counters.record_counts)
    for table_name, schema in output_tables().items():
        files = sorted((tasks_dir / table_name).glob("*.parquet"))
        if not files:
            continue
        path = part_path(staged, table_name, batch)
        path.parent.mkdir(parents=True)
        offset = record_counts.get(table_name, 0)
        ordered = order_table(
            pl.concat([pl.scan_parquet(f) for f in files]), table_name, schema, offset
        )
        sink_parquet(ordered, path, temporary=True)
        rows = sum(r.row_counts.get(table_name, 0) for r in results)
        row_counts[table_name] = rows
        if table_name in EVENT_CONCEPT_COLUMNS:
            record_counts[table_name] = offset + rows

    row_counts[tbl.excluded] = sum(r.row_counts.get(tbl.excluded, 0) for r in results)
    for excluded in sorted(excluded_dir.glob("*.parquet")):
        target = part_path(staged, f"{tbl.excluded}/{excluded.stem}", batch)
        target.parent.mkdir(parents=True)
        os.replace(excluded, target)

    coverage_frames = [r.coverage for r in results if r.coverage is not None]
    coverage = (
        pl.concat(coverage_frames)
        if coverage_frames
        else pl.DataFrame(schema=COVERAGE_PART_SCHEMA)
    )
    coverage_path = part_path(staged, COVERAGE_DIR, batch)
    coverage_path.parent.mkdir(parents=True)
    write_parquet(coverage, coverage_path, temporary=True)

    _commit_batch(work_dir, batch)
    return replace(
        counters, last_visit_id=last_visit_id, record_counts=record_counts
    ), row_counts


def run_batches(
    result: IngestResult,
    lookup: VocabularyLookup,
    counters: IdCounters | None = None,
    first_batch: int = 0,
) -> IdCounters:
    """Run stage 5 for batches first_batch..batch_count-1 in order.

    Args:
        result: Finished ingest stages.
        lookup: Vocabulary lookup of all batches.
        counters: ID counters after batch first_batch - 1 (fresh counters by default).
        first_batch: First batch to run.

    Returns:
        ID counters after the last batch.
    """
    counters = counters or IdCounters()
    for batch in range(first_batch, result.batch_count):
        counters, row_counts = transform_batch(result, batch, lookup, counters)
        counts = ", ".join(f"{name} {n:,}" for name, n in sorted(row_counts.items()))
        logger.info(f"  Batch {batch + 1}/{result.batch_count}: {counts}")
    return counters
