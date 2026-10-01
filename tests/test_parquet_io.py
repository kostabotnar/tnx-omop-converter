"""Tests for tnx_omop/util/parquet_io.py."""

import polars as pl
import pytest

from tnx_omop.util import parquet_io


@pytest.fixture
def captured(monkeypatch):
    """Record the keyword arguments polars receives for each write."""
    calls = []

    def fake_write(self, file, **kwargs):
        calls.append(("write", kwargs))

    def fake_sink(self, path, **kwargs):
        calls.append(("sink", kwargs))

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fake_write)
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fake_sink)
    return calls


def test_write_parquet_passes_options(captured, tmp_path):
    parquet_io.write_parquet(pl.DataFrame({"a": [1]}), tmp_path / "x.parquet")

    assert captured == [
        (
            "write",
            {"compression": "zstd", "statistics": True, "row_group_size": 500_000},
        )
    ]


def test_sink_parquet_passes_options(captured, tmp_path):
    parquet_io.sink_parquet(pl.LazyFrame({"a": [1]}), tmp_path / "x.parquet")

    assert captured == [
        (
            "sink",
            {"compression": "zstd", "statistics": True, "row_group_size": 500_000},
        )
    ]


def test_temporary_files_use_fast_compression_without_statistics(captured, tmp_path):
    parquet_io.write_parquet(pl.DataFrame({"a": [1]}), tmp_path / "w.parquet", True)
    parquet_io.sink_parquet(
        pl.LazyFrame({"a": [1]}), tmp_path / "s.parquet", temporary=True
    )

    expected = {"compression": "lz4", "statistics": False, "row_group_size": 500_000}
    assert captured == [("write", expected), ("sink", expected)]


def test_files_round_trip(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})

    parquet_io.write_parquet(df, tmp_path / "w.parquet")
    parquet_io.sink_parquet(df.lazy(), tmp_path / "s.parquet")
    parquet_io.write_parquet(df, tmp_path / "t.parquet", temporary=True)

    assert pl.read_parquet(tmp_path / "w.parquet").equals(df)
    assert pl.read_parquet(tmp_path / "s.parquet").equals(df)
    assert pl.read_parquet(tmp_path / "t.parquet").equals(df)


def test_row_group_size_constant():
    assert parquet_io.ROW_GROUP_SIZE == 500_000
