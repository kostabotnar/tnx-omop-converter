"""Tests for ingest stages 1 to 3: raw Parquet, persons and batches."""

from __future__ import annotations

import zipfile
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pytest

from tnx_omop.pipeline import ingest
from tnx_omop.pipeline.cleaning import TABLE_FILES
from tnx_omop.pipeline.ingest import (
    IngestResult,
    _cut_batches,
    batch_path,
    build_persons,
    scan_batch,
    write_batches,
    write_raw,
)
from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tests.tnx_dictionary import DICTIONARY, add_dictionary

Rows = list[list[str]]
Files = dict[str, tuple[list[str], Rows]]

PATIENT_HEADER = ["patient_id", "year_of_birth", "month_year_death", "source_id"]
ENCOUNTER_HEADER = [
    "encounter_id",
    "patient_id",
    "start_date",
    "end_date",
    "type",
    "derived_by_TriNetX",
]
DIAGNOSIS_HEADER = ["patient_id", "encounter_id", "code", "date", "derived_by_TriNetX"]
LAB_HEADER = ["patient_id", "encounter_id", "code", "date", "lab_result_num_val"]


def _csv(header: list[str], rows: Rows) -> str:
    lines = [",".join(header)] + [",".join(row) for row in rows]
    return "\n".join(lines) + "\n"


def _write_zip(path: Path, files: Files, folder: str = "") -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for table, (header, rows) in files.items():
            zf.writestr(f"{folder}{table}.csv", _csv(header, rows))
        add_dictionary(zf, folder)
    return path


def _source_a() -> Files:
    return {
        tbl.tnx_patient: (
            PATIENT_HEADER,
            [
                ["P1", "1950", "", "EHR"],
                ["P2", "1960", "202401", "EHR"],
                ["P3", "", "", "EHR"],  # no birth year
                ["P4", "1970", "202001", "EHR"],  # all rows after death
                ["P5", "1980", "", "EHR"],  # no rows
                ["P1", "1950", "", "TriNetX"],  # duplicated patient row
            ],
        ),
        tbl.tnx_encounter: (
            ENCOUNTER_HEADER,
            [
                ["E1", "P1", "20230101", "20230102", "IMP", "F"],
                ["E2", "P2", "20230201", "20230201", "AMB", "F"],
                ["E3", "P4", "20200301", "20200301", "AMB", "F"],
                ["E4", "P3", "20230101", "20230101", "AMB", "F"],
            ],
        ),
        tbl.tnx_diagnosis: (
            DIAGNOSIS_HEADER,
            [
                ["P1", "E1", "E11", "20230101", "F"],
                ["P1", "", "I10", "20230105", "T"],
                ["P2", "E2", "C91", "20230201", "F"],
                ["P2", "", "I10", "20240315", "F"],  # after death, still counted
                ["P4", "E3", "I10", "20200301", "F"],
                ["P9", "", "I10", "20230101", "F"],  # unknown patient
            ],
        ),
        tbl.tnx_lab_result: (
            LAB_HEADER,
            [["P1", "E1", "2160-0", "20230101", "1.5"], ["P1", "", "x", "bad", ""]],
        ),
    }


def _source_b() -> Files:
    return {
        tbl.tnx_patient: (
            PATIENT_HEADER,
            [
                ["P1", "1955", "", "EHR"],
                ["P2", "1965", "", "EHR"],
                ["A0", "1940", "", "EHR"],
            ],
        ),
        tbl.tnx_diagnosis: (
            DIAGNOSIS_HEADER,
            [
                ["P1", "", "E11", "20230101", "F"],
                ["P2", "", "E11", "20230101", "F"],
                ["P2", "", "E12", "20230102", "F"],
                ["A0", "", "E11", "", "F"],  # null date, no death: kept
            ],
        ),
    }


@pytest.fixture()
def zips(tmp_path: Path) -> list[Path]:
    return [
        _write_zip(tmp_path / "src_a.zip", _source_a()),
        _write_zip(tmp_path / "src_b.zip", _source_b(), folder="export/inner/"),
    ]


@pytest.fixture()
def raw_work(zips: list[Path], tmp_path: Path) -> Path:
    work = tmp_path / "work"
    write_raw(zips, work, DICTIONARY)
    return work


def _read_all_batches(work: Path, table: str, count: int) -> pl.DataFrame:
    frames = [scan_batch(work, table, b) for b in range(count)]
    return pl.concat([f for f in frames if f is not None]).collect()


def _read_raw(work: Path, table: str) -> pl.DataFrame | None:
    paths = sorted((work / "raw").glob(f"*/{table}.parquet"))
    if not paths:
        return None
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed")


class TestWriteRaw:
    def test_typing_and_derived_columns(self, raw_work: Path):
        lab = pl.read_parquet(raw_work / "raw" / "src_a" / "lab_result.parquet")
        assert lab.schema["lab_result_num_val"] == pl.Float64
        assert lab.schema[col.date] == pl.Date
        assert lab.get_column("lab_result_num_val").to_list() == [1.5, None]
        assert lab.get_column(col.date).null_count() == 1
        assert lab.get_column(col._source_id).unique().to_list() == ["src_a"]

        encounter = pl.read_parquet(raw_work / "raw" / "src_a" / "encounter.parquet")
        assert "derived_by_TriNetX" not in encounter.columns
        assert encounter.schema[col.start_date] == pl.Date
        assert encounter.schema[col.encounter_id] == pl.Utf8

        patient = pl.read_parquet(raw_work / "raw" / "src_a" / "patient.parquet")
        assert patient.schema[col.year_of_birth] == pl.Int64
        assert patient.schema[col.patient_id] == pl.Utf8

    def test_csv_deleted_and_nested_zip_found(self, raw_work: Path):
        assert not list((raw_work / "raw").rglob("*.csv"))
        assert (raw_work / "raw" / "src_b" / "diagnosis.parquet").exists()
        assert not (raw_work / "raw" / "src_b" / "encounter.parquet").exists()

    def test_meta_files_copied(self, tmp_path: Path):
        files = _source_a()
        files[tbl.tnx_dataset_details] = (["network_name"], [["Net"]])
        zip_path = _write_zip(tmp_path / "src_a.zip", files)
        write_raw([zip_path], tmp_path / "work", DICTIONARY)

        meta = tmp_path / "work" / "meta" / "src_a"
        assert (meta / "dataset_details.csv").exists()
        assert not (meta / "cohort_details.csv").exists()

    def test_missing_patient_file_raises(self, tmp_path: Path):
        files = _source_a()
        del files[tbl.tnx_patient]
        zip_path = _write_zip(tmp_path / "src_a.zip", files)
        with pytest.raises(ValueError, match="patient.csv"):
            write_raw([zip_path], tmp_path / "work", DICTIONARY)


class TestBuildPersons:
    def test_person_ids_follow_sorted_source_and_patient(self, raw_work: Path):
        build_persons(raw_work)
        persons = pl.read_parquet(raw_work / "persons.parquet")

        assert persons.select(col._source_id, col.patient_id, col.person_id).rows() == [
            ("src_a", "P1", 1),
            ("src_a", "P2", 2),
            ("src_b", "A0", 3),
            ("src_b", "P1", 4),
            ("src_b", "P2", 5),
        ]

    def test_same_patient_id_in_two_sources_stays_distinct(self, raw_work: Path):
        build_persons(raw_work)
        persons = pl.read_parquet(raw_work / "persons.parquet")

        p1 = persons.filter(pl.col(col.patient_id) == "P1")
        assert p1.get_column(col._source_id).sort().to_list() == ["src_a", "src_b"]
        assert p1.get_column(col.person_id).n_unique() == 2

    def test_removed_patients(self, raw_work: Path):
        build_persons(raw_work)
        src_a = pl.read_parquet(raw_work / "persons.parquet").filter(
            pl.col(col._source_id) == "src_a"
        )

        # P3 has no birth year, P4 only rows after death, P5 no rows
        assert src_a.get_column(col.patient_id).to_list() == ["P1", "P2"]

    def test_null_date_without_death_is_kept(self, raw_work: Path):
        build_persons(raw_work)
        persons = pl.read_parquet(raw_work / "persons.parquet")

        assert "A0" in persons.get_column(col.patient_id).to_list()

    def test_null_date_with_death_does_not_count(self, tmp_path: Path):
        files = {
            tbl.tnx_patient: _source_a()[tbl.tnx_patient],
            tbl.tnx_diagnosis: (DIAGNOSIS_HEADER, [["P2", "", "I10", "bad", "F"]]),
        }
        zip_path = _write_zip(tmp_path / "src_a.zip", files)
        work = tmp_path / "work"
        write_raw([zip_path], work, DICTIONARY)
        build_persons(work)

        assert pl.read_parquet(work / "persons.parquet").height == 0

    def test_row_counts_per_person(self, raw_work: Path):
        build_persons(raw_work)
        persons = pl.read_parquet(raw_work / "persons.parquet")

        rows = {
            (r[col._source_id], r[col.patient_id]): r[col.person_rows]
            for r in persons.iter_rows(named=True)
        }
        # src_a P1: encounter 1, diagnosis 2, lab 2
        assert rows[("src_a", "P1")] == 5
        assert rows[("src_b", "P2")] == 2


class TestCutBatches:
    @pytest.mark.parametrize(
        ("rows", "limit", "expected"),
        [
            ([3, 3, 3, 3], 6, [0, 0, 1, 1]),
            ([3, 3, 3, 3], 5, [0, 1, 2, 3]),
            ([2, 2, 2, 2, 2], 5, [0, 0, 1, 1, 2]),
            ([1, 10, 1, 1], 4, [0, 1, 2, 2]),
            ([10, 1], 4, [0, 1]),
            ([5], 5, [0]),
            ([], 5, []),
        ],
    )
    def test_greedy_ranges(self, rows: list[int], limit: int, expected: list[int]):
        result = _cut_batches(pl.Series(rows, dtype=pl.Int64), limit)
        assert result.to_list() == expected

    def test_persons_file_respects_batch_rows(self, raw_work: Path):
        build_persons(raw_work, batch_rows=4)
        persons = pl.read_parquet(raw_work / "persons.parquet")

        assert persons.get_column(col.batch).is_sorted()
        assert persons.get_column(col.batch).min() == 0
        per_batch = persons.group_by(col.batch).agg(
            pl.col(col.person_rows).sum().alias("rows"),
            pl.len().alias("persons"),
            pl.col(col.person_id).min().alias("lo"),
            pl.col(col.person_id).max().alias("hi"),
        )
        # A batch is over the limit only when it holds a single person
        oversize = per_batch.filter(pl.col("rows") > 4)
        assert oversize.height >= 1
        assert oversize.get_column("persons").to_list() == [1] * oversize.height
        # Persons stay whole and batches are contiguous person_id ranges
        contiguous = per_batch.select(
            (pl.col("hi") - pl.col("lo") + 1 == pl.col("persons")).all()
        )
        assert contiguous.item()

    def test_invalid_batch_rows(self, raw_work: Path):
        with pytest.raises(ValueError, match="batch_rows"):
            build_persons(raw_work, batch_rows=0)


class TestWriteBatches:
    @pytest.mark.parametrize("batch_rows", [3, 1_000])
    def test_every_row_exactly_once(self, raw_work: Path, batch_rows: int):
        build_persons(raw_work, batch_rows)
        persons = pl.read_parquet(raw_work / "persons.parquet")
        raw = {t: _read_raw(raw_work, t) for t in TABLE_FILES}
        write_batches(raw_work)

        assert not (raw_work / "raw").exists()
        batch_count = persons.get_column(col.batch).max() + 1
        keys = persons.select(col._source_id, col.patient_id, col.person_id)
        for table, frame in raw.items():
            if frame is None:
                continue
            expected = frame.join(
                keys, on=[col._source_id, col.patient_id], how="inner"
            ).sort(pl.all())
            got = _read_all_batches(raw_work, table, batch_count)
            assert got.select(expected.columns).sort(pl.all()).equals(expected)

        # Rows of a batch belong to persons of that batch
        for batch in range(batch_count):
            lf = scan_batch(raw_work, tbl.tnx_diagnosis, batch)
            if lf is None:
                continue
            ids = lf.select(col.person_id).unique().collect().get_column(col.person_id)
            allowed = persons.filter(pl.col(col.batch) == batch).get_column(
                col.person_id
            )
            assert ids.is_in(allowed.implode()).all()

    def test_duplicated_patient_rows_keep_person_id(
        self, zips: list[Path], tmp_path: Path
    ):
        result = ingest.ingest(zips, tmp_path / "work", DICTIONARY)
        patient = _read_all_batches(
            result.work_dir, tbl.tnx_patient, result.batch_count
        )

        p1 = patient.filter(
            (pl.col(col._source_id) == "src_a") & (pl.col(col.patient_id) == "P1")
        )
        assert p1.height == 2
        assert p1.get_column(col.person_id).n_unique() == 1
        assert patient.height == 6  # P1 twice and P2 (a); P1, P2 and A0 (b)


class TestIngest:
    def test_result_and_scan_batch(self, zips: list[Path], tmp_path: Path):
        work = tmp_path / "work"
        result = ingest.ingest(zips, work, DICTIONARY, batch_rows=4)

        assert isinstance(result, IngestResult)
        assert result.work_dir == work
        assert result.person_count == 5
        assert result.batch_count == 3
        assert result.tables == sorted(
            [tbl.tnx_patient, tbl.tnx_encounter, tbl.tnx_diagnosis, tbl.tnx_lab_result]
        )
        assert result.meta_dirs == {}
        assert batch_path(work, tbl.tnx_diagnosis, 3) == (
            work / "batches" / "diagnosis" / "00003.parquet"
        )
        assert scan_batch(work, tbl.tnx_diagnosis, result.batch_count + 5) is None
        assert scan_batch(work, "no_such_table", 0) is None

    def test_rerun_replaces_stages(self, zips: list[Path], tmp_path: Path):
        work = tmp_path / "work"
        first = ingest.ingest(zips, work, DICTIONARY, batch_rows=4)
        second = ingest.ingest(zips, work, DICTIONARY, batch_rows=1_000)

        assert first.batch_count == 3
        assert second.batch_count == 1
        assert not batch_path(work, tbl.tnx_diagnosis, 1).exists()

    def test_stage_timer_wraps_each_run_stage(self, zips: list[Path], tmp_path: Path):
        events: list[str] = []

        @contextmanager
        def timer(name: str):
            events.append(f"start {name}")
            yield
            events.append(f"end {name}")

        ingest.ingest(zips, tmp_path / "work", DICTIONARY, stage_timer=timer)

        assert events == [f"{e} {s}" for s in ingest.STAGES for e in ("start", "end")]
