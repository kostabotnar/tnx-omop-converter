"""The compression option of convert() and the command line."""

import json
from pathlib import Path

import pytest

from tests.parquet_codec import column_codecs
from tests.test_converter import _write_zip
from tnx_omop import __main__ as entry
from tnx_omop import cli
from tnx_omop.pipeline.converter import convert
from tnx_omop.pipeline.run_report import REPORT_FILE
from tnx_omop.util import parquet_io


def _codecs_by_file(output_dir: Path) -> dict[str, set[str]]:
    """Codecs per file; empty tables have no row group and are left out."""
    codecs = {
        str(path.relative_to(output_dir)): column_codecs(path)
        for path in sorted(output_dir.rglob("*.parquet"))
    }
    return {name: found for name, found in codecs.items() if found}


class TestConvertCompression:
    def test_reduced_output_uses_the_codec(self, tmp_path: Path, athena_dir: Path):
        out = tmp_path / "omop"

        convert(
            [_write_zip(tmp_path / "tiny_export.zip")],
            out,
            athena_dir,
            compression="snappy",
            export_vocabulary=False,
            eras=False,
        )

        codecs = _codecs_by_file(out)
        assert codecs
        assert {k: v for k, v in codecs.items() if v != {"snappy"}} == {}
        report = json.loads((out / REPORT_FILE).read_text(encoding="utf-8"))
        assert report["compression"] == "snappy"

    def test_vocabulary_export_and_eras_use_the_codec(
        self, tmp_path: Path, athena_dir: Path
    ):
        out = tmp_path / "omop"

        convert(
            [_write_zip(tmp_path / "tiny_export.zip")],
            out,
            athena_dir,
            compression="gzip",
        )

        codecs = _codecs_by_file(out)
        tables = {name.split("\\")[0].split("/")[0] for name in codecs}
        assert {"condition_era", "concept_relationship", "person"} <= tables
        assert {k: v for k, v in codecs.items() if v != {"gzip"}} == {}

    def test_default_is_zstd(self, tmp_path: Path, athena_dir: Path):
        out = tmp_path / "omop"

        convert(
            [_write_zip(tmp_path / "tiny_export.zip")],
            out,
            athena_dir,
            export_vocabulary=False,
            eras=False,
        )

        codecs = _codecs_by_file(out)
        assert codecs
        assert {k: v for k, v in codecs.items() if v != {"zstd"}} == {}
        report = json.loads((out / REPORT_FILE).read_text(encoding="utf-8"))
        assert report["compression"] == "zstd"

    def test_unknown_codec_fails_before_any_work(
        self, tmp_path: Path, athena_dir: Path
    ):
        out = tmp_path / "omop"

        with pytest.raises(ValueError, match="compression"):
            convert(
                [_write_zip(tmp_path / "tiny_export.zip")],
                out,
                athena_dir,
                compression="lzo",
            )

        assert not out.exists()


class TestCommandLine:
    ARGV = ["convert", "-i", "a.zip", "-o", "out", "--vocab-dir", "v"]

    def test_default(self):
        assert cli.parse_args(self.ARGV).compression == parquet_io.COMPRESSION

    @pytest.mark.parametrize("codec", parquet_io.COMPRESSIONS)
    def test_every_codec_is_accepted(self, codec: str):
        args = cli.parse_args([*self.ARGV, "--compression", codec])

        assert args.compression == codec

    def test_unknown_codec_is_a_usage_error(self):
        with pytest.raises(SystemExit) as exc:
            cli.parse_args([*self.ARGV, "--compression", "lzo"])

        assert exc.value.code == 2

    def test_main_passes_the_codec_to_convert(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        seen = {}
        monkeypatch.setattr(entry, "_check_inputs", lambda *a, **k: True)
        monkeypatch.setattr(entry, "_check_config", lambda *a, **k: True)
        monkeypatch.setattr(entry, "convert", lambda *a, **k: seen.update(k))

        code = entry.main([*self.ARGV, "--compression", "brotli"])

        assert code == 0
        assert seen["compression"] == "brotli"
