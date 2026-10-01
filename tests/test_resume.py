"""Manifest and resume of the working directory (stage 7)."""

import json
import logging
import os
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tnx_omop.pipeline import batch_transform as bt
from tnx_omop.pipeline import ingest
from tnx_omop.pipeline import manifest as mf
from tnx_omop.pipeline.batch_transform import IdCounters, transform_batch
from tnx_omop.pipeline.converter import build_vocab_lookup, convert
from tnx_omop.util import tables as tbl
from tests.test_converter import _read_output, _two_source_zips, _write_zip
from tests.tnx_dictionary import DICTIONARY

BATCH_ROWS = 1  # one person per batch: four batches for two sources


def _vocab(root: Path) -> Path:
    vocab = root / "vocab"
    vocab.mkdir()
    for stem in (tbl.athena_concept, tbl.athena_concept_relationship):
        (vocab / f"{stem}.csv").write_text("x")
    return vocab


def _touch(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


class TestManifest:
    @pytest.fixture()
    def inputs(self, tmp_path: Path) -> tuple[list[Path], Path]:
        zips = [_write_zip(tmp_path / "a.zip"), _write_zip(tmp_path / "b.zip")]
        return zips, _vocab(tmp_path)

    def test_same_inputs_same_fingerprint(self, inputs):
        zips, vocab = inputs
        assert mf.build_fingerprint(zips, vocab, 10) == mf.build_fingerprint(
            zips, vocab, 10
        )

    def test_zip_size_change(self, inputs):
        zips, vocab = inputs
        before = mf.build_fingerprint(zips, vocab, 10)
        with open(zips[0], "ab") as f:
            f.write(b"0")
        assert mf.build_fingerprint(zips, vocab, 10) != before

    def test_zip_mtime_change(self, inputs):
        zips, vocab = inputs
        before = mf.build_fingerprint(zips, vocab, 10)
        _touch(zips[1])
        assert mf.build_fingerprint(zips, vocab, 10) != before

    def test_zip_order_change(self, inputs):
        zips, vocab = inputs
        assert mf.build_fingerprint(zips, vocab, 10) != mf.build_fingerprint(
            zips[::-1], vocab, 10
        )

    def test_batch_rows_change(self, inputs):
        zips, vocab = inputs
        assert mf.build_fingerprint(zips, vocab, 10) != mf.build_fingerprint(
            zips, vocab, 11
        )

    def test_vocabulary_file_change(self, inputs):
        zips, vocab = inputs
        before = mf.build_fingerprint(zips, vocab, 10)
        (vocab / f"{tbl.athena_concept}.csv").write_text("xy")
        assert mf.build_fingerprint(zips, vocab, 10) != before

    def test_optional_cpt4_file_counts(self, inputs):
        zips, vocab = inputs
        before = mf.build_fingerprint(zips, vocab, 10)
        (vocab / f"{tbl.athena_concept_cpt4}.csv").write_text("x")
        assert mf.build_fingerprint(zips, vocab, 10) != before

    def test_save_load_round_trip(self, inputs, tmp_path: Path):
        zips, vocab = inputs
        work = tmp_path / "work"
        fingerprint = mf.build_fingerprint(zips, vocab, 5)
        manifest = mf.Manifest(mf.PIPELINE_VERSION, fingerprint)
        mf.save_manifest(work, manifest)
        mf.mark_stage_done(work, manifest, ingest.STAGE_RAW)

        loaded = mf.load_manifest(work)

        assert loaded == manifest
        assert loaded.completed_stages == [ingest.STAGE_RAW]
        assert mf.matches(loaded, fingerprint)
        assert [p.name for p in work.iterdir()] == [mf.MANIFEST_FILE]

    @pytest.mark.parametrize("text", [None, "", "{", "[]", '{"version": 1}'])
    def test_unreadable_manifest_is_a_mismatch(self, tmp_path: Path, text):
        if text is not None:
            (tmp_path / mf.MANIFEST_FILE).write_text(text)
        manifest = mf.load_manifest(tmp_path)
        assert manifest is None
        assert not mf.matches(manifest, {})

    def test_other_version_is_a_mismatch(self, tmp_path: Path):
        manifest = mf.Manifest(mf.PIPELINE_VERSION + 1, {"batch_rows": 1})
        mf.save_manifest(tmp_path, manifest)
        assert not mf.matches(mf.load_manifest(tmp_path), {"batch_rows": 1})


class TestCounters:
    @pytest.fixture()
    def stage_inputs(self, tmp_path: Path, athena_dir: Path):
        result = ingest.ingest(
            _two_source_zips(tmp_path),
            tmp_path / "work",
            DICTIONARY,
            batch_rows=BATCH_ROWS,
        )
        return result, build_vocab_lookup(result, athena_dir)

    def test_counters_from_parts_equal_running_counters(self, stage_inputs):
        result, lookup = stage_inputs
        assert bt.counters_from_parts(result.work_dir, 0) == IdCounters()
        counters = IdCounters()
        for batch in range(result.batch_count):
            counters, _ = transform_batch(result, batch, lookup, counters)
            assert bt.counters_from_parts(result.work_dir, batch + 1) == counters
        assert counters.last_visit_id == 8
        assert counters.record_counts

    def test_completed_batches_and_later_parts(self, stage_inputs):
        result, lookup = stage_inputs
        bt.run_batches(result, lookup)
        work = result.work_dir
        assert bt.completed_batches(work, result.batch_count) == 4

        parts = work / bt.PARTS_DIR
        bt.part_path(parts, bt.COVERAGE_DIR, 1).unlink()
        assert bt.completed_batches(work, result.batch_count) == 1
        assert not bt.part_path(parts, bt.COVERAGE_DIR, 2).exists()
        assert not bt.part_path(parts, tbl.omop_person, 3).exists()
        assert bt.part_path(parts, tbl.omop_person, 0).exists()


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory, athena_dir: Path):
    """Zips and the output of an uninterrupted run."""
    root = tmp_path_factory.mktemp("resume_reference")
    zips = _two_source_zips(root)
    convert(zips, root / "omop", athena_dir, batch_rows=BATCH_ROWS)
    return zips, _read_output(root / "omop")


def assert_same_output(out: Path, expected: dict[str, pl.DataFrame]) -> None:
    actual = _read_output(out)
    assert actual.keys() == expected.keys()
    for name, frame in expected.items():
        assert_frame_equal(actual[name], frame, check_row_order=True)


def fail_on(monkeypatch: pytest.MonkeyPatch, target: str, when=lambda *a: True):
    """Make the function at the dotted path target raise when when(*args) holds.

    Returns a function that puts the original back.
    """
    module_name, _, name = target.rpartition(".")
    module = __import__(module_name, fromlist=[name])
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        if when(*args):
            raise RuntimeError("boom")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, wrapper)
    return lambda: monkeypatch.setattr(target, original)


def count_calls(monkeypatch: pytest.MonkeyPatch, target: str) -> list[tuple]:
    """Wrap the function at the dotted path target and record its calls."""
    module_name, _, name = target.rpartition(".")
    original = getattr(__import__(module_name, fromlist=[name]), name)
    calls: list[tuple] = []

    def wrapper(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, wrapper)
    return calls


FAIL_BATCH_2 = "tnx_omop.pipeline.batch_transform.transform_batch"


class TestResume:
    def test_failure_in_middle_batch_then_resume(
        self, tmp_path: Path, athena_dir: Path, reference, monkeypatch, caplog
    ):
        zips, expected = reference
        out, work = tmp_path / "omop", tmp_path / "work"
        restore = fail_on(monkeypatch, FAIL_BATCH_2, lambda r, b, *a: b == 2)

        with pytest.raises(RuntimeError, match="boom"):
            convert(zips, out, athena_dir, work_dir=work, batch_rows=BATCH_ROWS)

        assert mf.load_manifest(work).completed_stages == list(ingest.STAGES)
        assert (work / "parts" / "coverage" / "00001.parquet").exists()
        assert not (work / "parts" / "coverage" / "00002.parquet").exists()

        restore()
        raw_calls = count_calls(monkeypatch, "tnx_omop.pipeline.ingest.write_raw")
        batch_calls = count_calls(monkeypatch, FAIL_BATCH_2)
        caplog.set_level(logging.INFO)
        caplog.clear()
        convert(zips, out, athena_dir, work_dir=work, batch_rows=BATCH_ROWS)

        text = caplog.text
        assert "stages raw, persons, batches done; 2 of 4 batches done" in text
        assert [c[1] for c in batch_calls] == [2, 3]
        assert not raw_calls
        report = json.loads((out / "run_report.json").read_text(encoding="utf-8"))
        assert report["resumed"] is True
        assert report["batches_already_done"] == 2
        assert report["ingest_stages_skipped"] == list(ingest.STAGES)
        assert not work.exists()
        assert_same_output(out, expected)

    @pytest.mark.parametrize(
        ("target", "done"),
        [
            ("tnx_omop.pipeline.ingest.build_persons", ["raw"]),
            ("tnx_omop.pipeline.ingest.write_batches", ["raw", "persons"]),
        ],
    )
    def test_failure_in_ingest_stage_then_resume(
        self, tmp_path: Path, athena_dir: Path, reference, monkeypatch, target, done
    ):
        zips, expected = reference
        out, work = tmp_path / "omop", tmp_path / "work"
        restore = fail_on(monkeypatch, target)

        with pytest.raises(RuntimeError, match="boom"):
            convert(zips, out, athena_dir, work_dir=work, batch_rows=BATCH_ROWS)
        assert mf.load_manifest(work).completed_stages == done
        assert (work / ingest.RAW_DIR).exists()

        restore()
        raw_calls = count_calls(monkeypatch, "tnx_omop.pipeline.ingest.write_raw")
        convert(zips, out, athena_dir, work_dir=work, batch_rows=BATCH_ROWS)

        assert not raw_calls
        assert_same_output(out, expected)

    def test_changed_batch_rows_discards_work_dir(
        self, tmp_path: Path, athena_dir: Path, reference, monkeypatch, caplog
    ):
        zips, expected = reference
        out, work = tmp_path / "omop", tmp_path / "work"
        work.mkdir()
        (work / "notes.txt").write_text("keep")
        restore = fail_on(monkeypatch, FAIL_BATCH_2, lambda r, b, *a: b == 2)
        with pytest.raises(RuntimeError, match="boom"):
            convert(zips, out, athena_dir, work_dir=work, batch_rows=BATCH_ROWS)

        restore()
        raw_calls = count_calls(monkeypatch, "tnx_omop.pipeline.ingest.write_raw")
        caplog.set_level(logging.INFO)
        caplog.clear()
        convert(zips, out, athena_dir, work_dir=work, batch_rows=2)

        text = caplog.text
        assert "discarding" in text and "Resuming" not in text
        assert len(raw_calls) == 1
        assert [p.name for p in work.iterdir()] == ["notes.txt"]
        assert_same_output(out, expected)

    def test_changed_zip_discards_work_dir(
        self, tmp_path: Path, athena_dir: Path, monkeypatch, caplog
    ):
        zips = _two_source_zips(tmp_path)
        out, work = tmp_path / "omop", tmp_path / "work"
        restore = fail_on(monkeypatch, "tnx_omop.pipeline.ingest.write_batches")
        with pytest.raises(RuntimeError):
            convert(zips, out, athena_dir, work_dir=work, batch_rows=BATCH_ROWS)
        restore()
        _touch(zips[0])
        caplog.set_level(logging.INFO)
        caplog.clear()

        convert(zips, out, athena_dir, work_dir=work, batch_rows=BATCH_ROWS)

        assert "discarding" in caplog.text

    def test_keep_work_dir_keeps_manifest(self, tmp_path: Path, athena_dir: Path):
        work = tmp_path / "work"
        convert(
            [_write_zip(tmp_path / "tiny_export.zip")],
            tmp_path / "omop",
            athena_dir,
            work_dir=work,
            keep_work_dir=True,
        )
        assert mf.load_manifest(work).completed_stages == list(ingest.STAGES)
        assert not (work / "raw").exists()
