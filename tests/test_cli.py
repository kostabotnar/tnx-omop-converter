"""Argument parsing and exit codes of the `tnx-omop` subcommands."""

import logging
import shutil
from pathlib import Path

import polars as pl
import pytest

from tnx_omop import __main__ as entry
from tnx_omop import cli
from tnx_omop.converter import convert
from tnx_omop.ingest import DEFAULT_BATCH_ROWS
from tnx_omop.omop_vocab.concept_table import table_path
from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tests.test_converter import _write_zip


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory, athena_dir: Path) -> Path:
    root = tmp_path_factory.mktemp("cli")
    out = root / "omop"
    convert([_write_zip(root / "tiny_export.zip")], out, athena_dir)
    return out


class TestParsing:
    def test_convert_defaults(self):
        args = cli.parse_args(
            ["convert", "-i", "a.zip", "b.zip", "-o", "out", "--vocab-dir", "v"]
        )

        assert args.command == cli.CONVERT
        assert args.input == [Path("a.zip"), Path("b.zip")]
        assert args.output == Path("out")
        assert args.vocab_dir == Path("v")
        assert args.work_dir is None
        assert args.batch_rows == DEFAULT_BATCH_ROWS
        assert args.keep_work_dir is False
        assert args.export_vocabulary is True
        assert args.eras is True
        assert args.config is None

    def test_convert_options(self):
        args = cli.parse_args(
            ["convert", "-i", "a.zip", "-o", "out", "--vocab-dir", "v"]
            + ["--work-dir", "w", "--batch-rows", "7", "--keep-work-dir"]
        )

        assert args.work_dir == Path("w")
        assert args.batch_rows == 7
        assert args.keep_work_dir is True

    def test_export_vocabulary_flag(self):
        args = cli.parse_args(
            ["convert", "-i", "a.zip", "-o", "out", "--vocab-dir", "v"]
            + ["--export-vocabulary"]
        )

        assert args.export_vocabulary is True

    def test_opt_out_flags(self):
        args = cli.parse_args(
            ["convert", "-i", "a.zip", "-o", "out", "--vocab-dir", "v"]
            + ["--no-export-vocabulary", "--no-eras"]
        )

        assert args.export_vocabulary is False
        assert args.eras is False

    @pytest.mark.parametrize(
        "argv",
        [
            ["convert", "-i", "a.zip", "-o", "out"],
            ["convert", "-i", "a.zip", "-o", "out", "--vocab-dir", "v"]
            + ["--batch-rows", "0"],
            ["validate"],
            ["coverage", "--vocab-dir", "v"],
            ["-i", "a.zip", "-o", "out", "--vocab-dir", "v"],
            ["unknown"],
        ],
        ids=[
            "convert without vocab-dir",
            "batch-rows zero",
            "validate without folder",
            "coverage without input",
            "no subcommand, old form",
            "unknown subcommand",
        ],
    )
    def test_usage_errors_exit_2(self, argv: list[str]):
        with pytest.raises(SystemExit) as exc:
            cli.parse_args(argv)

        assert exc.value.code == 2

    def test_validate_and_coverage_arguments(self):
        validate_args = cli.parse_args(["validate", "out", "--coverage"])
        coverage_args = cli.parse_args(
            ["coverage", "-i", "a.zip", "--vocab-dir", "v", "-o", "r.md"]
        )

        assert validate_args.output_dir == Path("out") and validate_args.coverage
        assert coverage_args.output == Path("r.md")
        assert cli.parse_args(
            ["coverage", "-i", "a.zip", "--vocab-dir", "v"]
        ).output == (cli.DEFAULT_COVERAGE_OUTPUT)


class TestLoggingOptions:
    def test_verbosity_options_precede_the_command(self):
        assert cli.parse_args(["-v", "validate", "out"]).verbose
        assert cli.parse_args(["--quiet", "validate", "out"]).quiet
        args = cli.parse_args(["validate", "out"])
        assert not args.verbose and not args.quiet

    def test_verbose_and_quiet_are_exclusive(self):
        with pytest.raises(SystemExit) as exc:
            cli.parse_args(["-v", "-q", "validate", "out"])

        assert exc.value.code == 2

    @pytest.mark.parametrize(
        ("flags", "level"),
        [([], logging.INFO), (["-v"], logging.DEBUG), (["-q"], logging.WARNING)],
    )
    def test_main_sets_the_log_level(self, flags: list[str], level: int, tmp_path):
        entry.main([*flags, "validate", str(tmp_path / "nowhere")])

        assert logging.getLogger().level == level

    def test_messages_go_to_stderr_without_prefix(self, capsys):
        entry.configure_logging()
        logging.getLogger("tnx_omop.test").info("  plain message")

        captured = capsys.readouterr()
        assert captured.err == "  plain message\n"
        assert captured.out == ""

    def test_repeated_setup_keeps_one_handler(self):
        root = logging.getLogger()
        before = len(root.handlers)
        entry.configure_logging()
        entry.configure_logging(verbose=True)

        assert len(root.handlers) == before + 1


class TestNoSubcommand:
    def test_prints_help_and_exits_2(self, capsys):
        assert entry.main([]) == 2

        err = capsys.readouterr().err
        assert "usage: tnx-omop" in err
        for command in (cli.CONVERT, cli.VALIDATE, cli.COVERAGE):
            assert command in err


class TestConvertCommand:
    def test_converts(self, tmp_path: Path, athena_dir: Path):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        out = tmp_path / "omop"

        code = entry.main(
            ["convert", "-i", str(zip_path), "-o", str(out)]
            + ["--vocab-dir", str(athena_dir)]
        )

        assert code == 0
        assert table_path(out, tbl.omop_person).exists()

    def test_bad_config_exits_1_before_any_work(
        self, tmp_path: Path, athena_dir: Path, caplog
    ):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        config = tmp_path / "config.json"
        config.write_text('{"gender": {"M": "male"}}', encoding="utf-8")
        out = tmp_path / "omop"

        code = entry.main(
            ["convert", "-i", str(zip_path), "-o", str(out)]
            + ["--vocab-dir", str(athena_dir), "--config", str(config)]
        )

        assert code == 1
        assert "'gender'" in caplog.text
        assert not out.exists()

    def test_bad_config_fails_coverage_too(
        self, tmp_path: Path, athena_dir: Path, caplog
    ):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")

        code = entry.main(
            ["coverage", "-i", str(zip_path), "--vocab-dir", str(athena_dir)]
            + ["--config", str(tmp_path / "missing.json")]
        )

        assert code == 1
        assert "Cannot read config file" in caplog.text

    def test_input_error_exits_1(self, tmp_path: Path, athena_dir: Path, caplog):
        code = entry.main(
            ["convert", "-i", str(tmp_path / "missing.zip"), "-o", str(tmp_path / "o")]
            + ["--vocab-dir", str(athena_dir)]
        )

        assert code == 1
        assert "Input file not found" in caplog.text

    def test_duplicate_zip_names_exit_1(self, tmp_path: Path, athena_dir: Path, caplog):
        first = _write_zip(tmp_path / "tiny_export.zip")
        (tmp_path / "other").mkdir()
        second = _write_zip(tmp_path / "other" / "tiny_export.zip")

        code = entry.main(
            ["convert", "-i", str(first), str(second), "-o", str(tmp_path / "o")]
            + ["--vocab-dir", str(athena_dir)]
        )

        assert code == 1
        assert "Duplicate ZIP file names" in caplog.text


class TestValidateCommand:
    def test_valid_output_exits_0(self, converted: Path, capsys):
        assert entry.main(["validate", str(converted)]) == 0

        out = capsys.readouterr().out
        assert "0 failed, 0 skipped" in out
        assert "FAILED" not in out
        assert "Rows per OMOP table" not in out

    def test_coverage_flag_prints_report(self, converted: Path, capsys):
        assert entry.main(["validate", str(converted), "--coverage"]) == 0

        out = capsys.readouterr().out
        assert "Rows per OMOP table:" in out
        assert "Excluded rows per TriNetX table and reason:" in out

    def test_damaged_output_exits_1(self, converted: Path, tmp_path: Path, capsys):
        damaged = tmp_path / "damaged"
        shutil.copytree(converted, damaged)
        person = table_path(damaged, tbl.omop_person)
        pl.read_parquet(person).with_columns(
            pl.lit(1).alias(col.person_id)
        ).write_parquet(person)

        assert entry.main(["validate", str(damaged)]) == 1

        out = capsys.readouterr().out
        assert f"FAILED {tbl.omop_person} schema: [{tbl.omop_person}] DUPLICATE" in out
        assert " failed" in out and "0 failed" not in out

    def test_missing_folder_exits_1(self, tmp_path: Path, caplog):
        assert entry.main(["validate", str(tmp_path / "nowhere")]) == 1
        assert "not found" in caplog.text


class TestCoverageCommand:
    def test_writes_report(self, tmp_path: Path, athena_dir: Path, capsys):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        report = tmp_path / "reports" / "coverage.md"

        code = entry.main(
            ["coverage", "-i", str(zip_path), "--vocab-dir", str(athena_dir)]
            + ["-o", str(report)]
        )

        assert code == 0
        assert report.read_text(encoding="utf-8") == capsys.readouterr().out

    def test_missing_input_exits_1(self, tmp_path: Path, athena_dir: Path, caplog):
        code = entry.main(
            ["coverage", "-i", str(tmp_path / "missing.zip")]
            + ["--vocab-dir", str(athena_dir), "-o", str(tmp_path / "r.md")]
        )

        assert code == 1
        assert "Input file not found" in caplog.text
