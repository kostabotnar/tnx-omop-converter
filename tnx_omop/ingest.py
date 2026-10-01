"""Stages 1 to 3 of the bounded memory pipeline: raw Parquet, persons, batches.

Each stage reads and writes only files in the working directory:

    <work>/raw/<source>/<table>.parquet       typed copy of each TriNetX CSV
    <work>/meta/<source>/<file>.csv           dataset, cohort and terminology files
    <work>/persons.parquet                    _source_id, patient_id, person_id, batch
    <work>/batches/<table>/<batch:05d>.parquet  rows of one person_id range

    <work>/parts/<omop table>/<batch:05d>.parquet  output of stage 5, one per batch
    <work>/parts/excluded/<tnx table>/<batch:05d>.parquet  excluded rows of stage 5

so a later run can skip finished stages. The manifest records each ingest stage right
after it finishes, and raw/ is deleted only after "batches" is recorded, so an
interrupted run never loses the input of a stage that is not recorded. Batching is safe
because cleaning works per patient: encounters are matched on (_source_id, patient_id,
encounter_id), never across patients.
"""

from __future__ import annotations

import math
import os
import shutil
import zipfile
from collections.abc import Callable, Collection
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from polars.io.partition import FileProviderArgs

from .cleaning import DATE_COLUMN_MAP, TABLE_FILES
from .tnx_schema import DataDictionary, apply_data_types, drop_derived_columns
from .util import columns as col
from .util import tables as tbl
from .util.date_utils import parse_month_year_death_column
from .util.parquet_io import sink_parquet

DEFAULT_BATCH_ROWS = 5_000_000

# Streaming scans of Parquet decode many row groups ahead of the consumer (about one per
# Polars thread, all of it in memory). Four keep the reader busy at a fraction of the
# memory, with no loss of speed in these stages. An explicit setting in the environment
# wins.
os.environ.setdefault("POLARS_ROW_GROUP_PREFETCH_SIZE", "4")

RAW_DIR = "raw"
META_DIR = "meta"
BATCHES_DIR = "batches"
GROUPS_DIR = "groups"
PERSONS_FILE = "persons.parquet"

STAGE_RAW = "raw"
STAGE_PERSONS = "persons"
STAGE_BATCHES = "batches"
STAGES = (STAGE_RAW, STAGE_PERSONS, STAGE_BATCHES)

# Row group size of the intermediate files that later stages read in parallel. Polars
# decodes several row groups at once, so its memory grows with this size: 500,000 rows
# of a medication table are about 70 MB per group. The files stay the same size.
WORK_ROW_GROUP_SIZE = 20_000

# A partitioned sink keeps a write buffer per open file, so the number of files one
# sink writes is capped; more batches are written in two steps (see write_batches).
MAX_OPEN_PARTITIONS = 32

_META_TABLES = (
    tbl.tnx_dataset_details,
    tbl.tnx_cohort_details,
    tbl.tnx_standardized_terminology,
)
_DEATH_DATE = "_death_date"
_ALIVE = "_alive"
_GROUP = "_group"


@dataclass
class IngestResult:
    """What the transform stage needs from stages 1 to 3."""

    work_dir: Path
    batch_count: int  # batches are numbered 0..batch_count-1 in person_id order
    tables: list[str]  # TriNetX tables present in batches/ (patient included)
    meta_dirs: dict[str, Path]  # _source_id -> <work>/meta/<source>
    person_count: int


def batch_path(work_dir: Path, table: str, batch: int) -> Path:
    """Path of the file with the rows of one table and batch."""
    return work_dir / BATCHES_DIR / table / f"{batch:05d}.parquet"


def scan_batch(work_dir: Path, table: str, batch: int) -> pl.LazyFrame | None:
    """Lazily read one table of one batch, or None when it has no file."""
    path = batch_path(work_dir, table, batch)
    return pl.scan_parquet(path) if path.exists() else None


def scan_table(work_dir: Path, table: str) -> pl.LazyFrame | None:
    """Lazily read one table across all batches, or None when it has no file."""
    if not any((work_dir / BATCHES_DIR / table).glob("*.parquet")):
        return None
    return pl.scan_parquet(work_dir / BATCHES_DIR / table / "*.parquet")


def ingest(
    input_zips: list[Path],
    work_dir: Path,
    dictionary: DataDictionary,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    completed: Collection[str] = (),
    on_stage_done: Callable[[str], None] | None = None,
    stage_timer: Callable[[str], AbstractContextManager[object]] | None = None,
) -> IngestResult:
    """Run stages 1 to 3: raw Parquet, person map and batches.

    Args:
        input_zips: TriNetX ZIP exports; the file stem is the source ID.
        work_dir: Working directory, created when missing.
        dictionary: Column types of the TriNetX tables (see
            tnx_schema.load_data_dictionary).
        batch_rows: Upper bound of source rows per batch (a single person with
            more rows gets a batch alone).
        completed: Stages (STAGES) that finished in an earlier run on the same
            inputs. They are skipped, unless an earlier stage has to run again.
        on_stage_done: Called with the stage name right after a stage finishes.
            raw/ is deleted only after the call for the batches stage, so a
            failure in between leaves the input of that stage in place.
        stage_timer: Called with the stage name; the stage runs inside the returned
            context manager (used to time a stage).

    Returns:
        IngestResult describing the batches.
    """
    stages: dict[str, Callable[[], None]] = {
        STAGE_RAW: lambda: write_raw(input_zips, work_dir, dictionary),
        STAGE_PERSONS: lambda: build_persons(work_dir, batch_rows),
        STAGE_BATCHES: lambda: write_batches(work_dir, remove_raw=False),
    }
    rerun = False
    for name, run in stages.items():
        if name in completed and not rerun:
            continue
        rerun = True
        with stage_timer(name) if stage_timer else nullcontext():
            run()
        if on_stage_done is not None:
            on_stage_done(name)
    discard_raw(work_dir)
    return read_result(work_dir)


def read_result(work_dir: Path) -> IngestResult:
    """Describe finished stages 1 to 3 from the files in work_dir."""
    persons = pl.scan_parquet(work_dir / PERSONS_FILE)
    person_count, batch_count = (
        persons.select(pl.len(), pl.col(col.batch).max().add(1).fill_null(0))
        .collect()
        .row(0)
    )
    batches_dir = work_dir / BATCHES_DIR
    meta_dir = work_dir / META_DIR
    return IngestResult(
        work_dir=work_dir,
        batch_count=batch_count,
        tables=sorted(d.name for d in batches_dir.iterdir() if d.is_dir())
        if batches_dir.exists()
        else [],
        meta_dirs={d.name: d for d in sorted(meta_dir.iterdir()) if d.is_dir()}
        if meta_dir.exists()
        else {},
        person_count=person_count,
    )


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _members_by_name(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Map each file name to its ZIP member; the shallowest path wins."""
    members: dict[str, zipfile.ZipInfo] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name
        known = members.get(name)
        if known is None or len(Path(info.filename).parts) < len(
            Path(known.filename).parts
        ):
            members[name] = info
    return members


def _extract_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path) -> None:
    with zf.open(info) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)


def write_raw(
    input_zips: list[Path], work_dir: Path, dictionary: DataDictionary
) -> None:
    """Stage 1: convert every table CSV of every ZIP to a typed Parquet file.

    Columns are read as strings, then typed from the data dictionary (see
    tnx_schema.apply_data_types). *derived_by_TriNetX columns are dropped and
    _source_id is added. One CSV at a time is extracted and deleted after
    conversion. The metadata CSVs are copied to <work>/meta/<source>/.

    Raises:
        ValueError: If no ZIP contains patient.csv.
    """
    raw_dir = _reset_dir(work_dir / RAW_DIR)
    meta_dir = _reset_dir(work_dir / META_DIR)
    patient_file = TABLE_FILES[tbl.tnx_patient]
    has_patients = False

    for zip_path in input_zips:
        source_id = zip_path.stem
        with zipfile.ZipFile(zip_path) as zf:
            members = _members_by_name(zf)

            for table in _META_TABLES:
                info = members.get(f"{table}.csv")
                if info is not None:
                    source_meta = meta_dir / source_id
                    source_meta.mkdir(exist_ok=True)
                    _extract_member(zf, info, source_meta / f"{table}.csv")

            # A source without patients is ignored, as in cleaning.clean_data
            if patient_file not in members:
                continue
            has_patients = True
            source_raw = raw_dir / source_id
            source_raw.mkdir()
            for table, filename in TABLE_FILES.items():
                info = members.get(filename)
                if info is not None:
                    _write_raw_table(zf, info, source_id, table, source_raw, dictionary)

    if not has_patients:
        raise ValueError("patient.csv is required but not found in any source")


def _write_raw_table(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    source_id: str,
    table: str,
    source_raw: Path,
    dictionary: DataDictionary,
) -> None:
    csv_path = source_raw / f"{table}.csv"
    _extract_member(zf, info, csv_path)
    lf = apply_data_types(
        drop_derived_columns(pl.scan_csv(csv_path, infer_schema=False)),
        table,
        dictionary,
    ).with_columns(pl.lit(source_id).alias(col._source_id))
    sink_parquet(lf, source_raw / f"{table}.parquet", WORK_ROW_GROUP_SIZE)
    csv_path.unlink()


def _raw_files(work_dir: Path, table: str) -> list[Path]:
    """Raw files of a table, one per source: raw/<source>/<table>.parquet."""
    raw_dir = work_dir / RAW_DIR
    return [
        path
        for path in sorted(raw_dir.glob(f"*/{table}.parquet"))
        if path.parent.parent == raw_dir
    ]


def _scan_raw(
    work_dir: Path, table: str, columns: list[str] | None = None
) -> pl.LazyFrame | None:
    """Scan the raw files of a table across sources, or None when there are none.

    With columns, only those present in a file are read; sources that lack a
    column get nulls. Without columns, all columns are read.
    """
    frames = []
    for path in _raw_files(work_dir, table):
        lf = pl.scan_parquet(path)
        if columns is not None:
            names = set(lf.collect_schema().names())
            lf = lf.select([c for c in columns if c in names])
        frames.append(lf)
    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


def _person_key() -> list[pl.Expr]:
    return [pl.col(col._source_id), pl.col(col.patient_id).cast(pl.Utf8)]


def _clean_patients(work_dir: Path) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Patient keys with a birth year and the known death months of those patients.

    Returns:
        (keys, deaths): keys has _source_id and patient_id; deaths has the same
        columns plus _death_date and may have several rows per patient when
        duplicated patient rows disagree.
    """
    scan = _scan_raw(
        work_dir,
        tbl.tnx_patient,
        [col._source_id, col.patient_id, col.year_of_birth, col.month_year_death],
    )
    if scan is None:
        raise ValueError("patient.csv is required but not found in any source")
    names = scan.collect_schema().names()
    yob = pl.col(col.year_of_birth).cast(pl.Utf8)
    patients = scan.filter(yob.is_not_null() & (yob != "")).select(
        *_person_key(),
        (
            parse_month_year_death_column(pl.col(col.month_year_death))
            if col.month_year_death in names
            else pl.lit(None, dtype=pl.Date)
        ).alias(_DEATH_DATE),
    )
    keys = patients.select(col._source_id, col.patient_id).drop_nulls().unique()
    deaths = patients.filter(pl.col(_DEATH_DATE).is_not_null()).unique()
    return keys, deaths


def build_persons(work_dir: Path, batch_rows: int = DEFAULT_BATCH_ROWS) -> None:
    """Stage 2: number the surviving persons and cut them into batches.

    A person survives when the patient has a birth year and at least one row in a
    table of DATE_COLUMN_MAP that passes the post-death filter of cleaning. Persons
    are numbered 1..N by (_source_id, patient_id), with no gaps. Contiguous
    person_id ranges are cut so a batch holds at most batch_rows source rows; a
    person with more rows gets a batch alone.

    Writes <work>/persons.parquet with _source_id, patient_id, person_id, batch and
    person_rows (source rows of the person, before deduplication).
    """
    if batch_rows < 1:
        raise ValueError("batch_rows must be at least 1")

    keys, deaths = _clean_patients(work_dir)
    per_table = []
    for table, date_col in DATE_COLUMN_MAP.items():
        scan = _scan_raw(work_dir, table, [col._source_id, col.patient_id, date_col])
        if scan is None:
            continue
        per_table.append(
            scan.select(*_person_key(), pl.col(date_col).alias(col.date))
            .join(keys, on=[col._source_id, col.patient_id], how="semi")
            .join(deaths, on=[col._source_id, col.patient_id], how="left")
            .group_by(col._source_id, col.patient_id)
            .agg(
                pl.len().alias(col.person_rows),
                (
                    pl.col(_DEATH_DATE).is_null()
                    | (pl.col(col.date) < pl.col(_DEATH_DATE))
                )
                .fill_null(False)
                .any()
                .alias(_ALIVE),
            )
        )

    schema = {
        col._source_id: pl.Utf8,
        col.patient_id: pl.Utf8,
        col.person_id: pl.Int64,
        col.batch: pl.Int64,
        col.person_rows: pl.Int64,
    }
    if not per_table:
        persons = pl.DataFrame(schema=schema)
    else:
        persons = (
            pl.concat(per_table)
            .group_by(col._source_id, col.patient_id)
            .agg(pl.col(col.person_rows).sum(), pl.col(_ALIVE).any())
            .filter(pl.col(_ALIVE))
            .drop(_ALIVE)
            .collect(engine="streaming")
            .sort(col._source_id, col.patient_id)
            .with_row_index(col.person_id, offset=1)
        )
        persons = persons.with_columns(
            pl.col(col.person_id, col.person_rows).cast(pl.Int64),
            _cut_batches(persons.get_column(col.person_rows), batch_rows)
            .cast(pl.Int64)
            .alias(col.batch),
        ).select(list(schema))
    sink_parquet(persons.lazy(), work_dir / PERSONS_FILE)


def _cut_batches(rows: pl.Series, batch_rows: int) -> pl.Series:
    """Batch number of each person for contiguous ranges of at most batch_rows rows.

    Greedy: every batch takes as many following persons as fit, at least one.
    Loops over batches only; each end is a binary search in the cumulative rows.
    """
    cumulative = rows.cum_sum()
    count = cumulative.len()
    ends: list[int] = []
    start, base = 0, 0
    while start < count:
        end = max(cumulative.search_sorted(base + batch_rows, side="right"), start + 1)
        ends.append(end)
        start, base = end, cumulative[end - 1]
    index = pl.int_range(count, eager=True)
    return pl.Series(ends, dtype=pl.Int64).search_sorted(index, side="right")


def write_batches(work_dir: Path, remove_raw: bool = True) -> None:
    """Stage 3: split every raw table into one file per batch, then delete raw/.

    With remove_raw=False raw/ stays until discard_raw() is called, so a caller can
    record the stage as finished first.

    Rows are joined (inner) to persons.parquet on (_source_id, patient_id), which
    adds person_id and drops rows of unknown and removed patients. The batch
    column selects the file and is not stored in it.

    Raw rows are not ordered by patient, so a sink sees every batch at once and
    buffers each open file. Above MAX_OPEN_PARTITIONS batches the rows are first
    split into about sqrt(batch count) groups of consecutive batches, then each group
    is split into its batches, so at most about sqrt(batch count) files are open
    and the memory of the stage does not grow with the export size.
    """
    batches_dir = _reset_dir(work_dir / BATCHES_DIR)
    groups_dir = _reset_dir(work_dir / GROUPS_DIR)
    persons = pl.scan_parquet(work_dir / PERSONS_FILE).select(
        col._source_id, col.patient_id, col.person_id, col.batch
    )
    batch_count = persons.select(pl.col(col.batch).max().add(1)).collect().item() or 0
    group_size = (
        batch_count
        if batch_count <= MAX_OPEN_PARTITIONS
        else math.isqrt(batch_count - 1) + 1
    )
    for table in TABLE_FILES:
        scan = _scan_raw(work_dir, table)
        if scan is None:
            continue
        table_dir = batches_dir / table
        table_dir.mkdir()
        joined = scan.with_columns(pl.col(col.patient_id).cast(pl.Utf8)).join(
            persons, on=[col._source_id, col.patient_id], how="inner"
        )
        if group_size >= batch_count:
            _partition_by(joined, table_dir, col.batch, include_key=False)
            continue
        table_groups = groups_dir / table
        table_groups.mkdir()
        _partition_by(
            joined.with_columns(pl.col(col.batch).floordiv(group_size).alias(_GROUP)),
            table_groups,
            _GROUP,
            include_key=True,
        )
        for group_file in sorted(table_groups.glob("*.parquet")):
            _partition_by(
                pl.scan_parquet(group_file).drop(_GROUP),
                table_dir,
                col.batch,
                include_key=False,
            )
            group_file.unlink()
    shutil.rmtree(groups_dir)
    if remove_raw:
        discard_raw(work_dir)


def discard_raw(work_dir: Path) -> None:
    """Delete raw/ and the group files, which stage 3 no longer needs."""
    shutil.rmtree(work_dir / RAW_DIR, ignore_errors=True)
    shutil.rmtree(work_dir / GROUPS_DIR, ignore_errors=True)


def _partition_by(lf: pl.LazyFrame, out_dir: Path, key: str, include_key: bool) -> None:
    """Write one file per value of the Int64 column key: <out_dir>/<key:05d>.parquet."""
    sink_parquet(
        lf,
        pl.PartitionBy(
            out_dir,
            key=key,
            include_key=include_key,
            file_path_provider=_partition_file_name,
            approximate_bytes_per_file=None,
        ),
        WORK_ROW_GROUP_SIZE,
    )


def _partition_file_name(args: FileProviderArgs) -> str:
    # A sink that closes a file early (too many open files) would start a second file
    # of the partition, and the same name would overwrite the first one.
    if args.index_in_partition:
        raise RuntimeError(
            f"Polars split a partition into several files: {args.partition_keys.row(0)}"
        )
    return f"{args.partition_keys.item():05d}.parquet"
