"""Tests for `convert --eras`: the stage, the flag, the checks and the output files."""

import json
import logging
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tnx_omop import __main__ as entry
from tnx_omop import cli
from tnx_omop.converter import STAGE_ERAS, convert
from tnx_omop.schema.omop import CONDITION_ERA_SCHEMA, DRUG_ERA_SCHEMA, ERA_SCHEMAS
from tnx_omop.omop_vocab.concept_table import table_path
from tnx_omop.run_report import REPORT_FILE
from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.validation import validate
from tests import athena_fixture as fx
from tests.schema_asserts import assert_conforms
from tests.test_converter import _write_zip


def read(output_dir: Path, table_name: str) -> pl.DataFrame:
    return pl.read_parquet(table_path(output_dir, table_name))


@pytest.fixture(scope="module")
def outputs(
    tmp_path_factory: pytest.TempPathFactory, athena_dir: Path
) -> tuple[Path, dict[str, int]]:
    """One conversion with eras: (output folder, row counts)."""
    root = tmp_path_factory.mktemp("eras")
    zip_path = _write_zip(root / "tiny_export.zip")
    counts = convert([zip_path], root / "omop", athena_dir, eras=True)
    return root / "omop", counts


class TestEraTables:
    def test_condition_eras(self, outputs):
        result = read(outputs[0], tbl.omop_condition_era)

        assert_conforms(result, CONDITION_ERA_SCHEMA)
        # The conditions have no end date, so each era ends on the day after its start
        assert result.rows() == [
            (1, 1, fx.COND_T2DM, date(2023, 1, 1), date(2023, 1, 2), 1),
            (2, 1, fx.COND_HYPERGLYCEMIA, date(2023, 1, 1), date(2023, 1, 2), 1),
            (3, 2, fx.COND_ALL, date(2023, 2, 1), date(2023, 2, 2), 1),
            (4, 2, fx.COND_ICD9_TARGET, date(2023, 2, 1), date(2023, 2, 2), 1),
        ]

    def test_drug_eras(self, outputs):
        result = read(outputs[0], tbl.omop_drug_era)

        assert_conforms(result, DRUG_ERA_SCHEMA)
        # Person 1: aspirin on 1 January and an extension ingredient on 1 March, more
        # than 30 days apart. Person 2: dexamethasone injection, whose ingredient in
        # the fixture vocabulary is aspirin.
        assert result.rows() == [
            (1, 1, fx.RXNORM_1191, date(2023, 1, 1), date(2023, 1, 1), 1, 0),
            (2, 1, fx.RXNORM_EXT_OMOP123, date(2023, 3, 1), date(2023, 3, 1), 1, 0),
            (3, 2, fx.RXNORM_1191, date(2023, 2, 1), date(2023, 2, 1), 1, 0),
        ]

    def test_row_counts_are_returned(self, outputs):
        assert outputs[1][tbl.omop_condition_era] == 4
        assert outputs[1][tbl.omop_drug_era] == 3

    def test_concept_holds_the_era_concepts(self, outputs):
        concept_ids = set(read(outputs[0], tbl.omop_concept)[col.concept_id])

        assert {fx.RXNORM_1191, fx.RXNORM_EXT_OMOP123, fx.COND_T2DM} <= concept_ids

    def test_work_dir_is_removed(self, outputs):
        assert not (outputs[0] / "_work").exists()

    def test_validate_passes(self, outputs):
        report = validate(outputs[0])

        assert report.ok, "\n".join(str(i) for r in report.failed for i in r.issues)
        names = {r.name for r in report.passed}
        assert {f"{t} schema" for t in ERA_SCHEMAS} | {"era concepts"} <= names

    def test_run_report_has_the_stage_and_row_counts(self, outputs):
        report = json.loads((outputs[0] / REPORT_FILE).read_text(encoding="utf-8"))

        assert STAGE_ERAS in report["stages"]
        assert report["row_counts"][tbl.omop_drug_era] == 3
        assert report["row_counts"][tbl.omop_condition_era] == 4


class TestConvert:
    def test_small_batches_give_the_same_eras(
        self, tmp_path: Path, athena_dir, outputs
    ):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")

        convert([zip_path], tmp_path / "omop", athena_dir, batch_rows=1, eras=True)

        for name in ERA_SCHEMAS:
            assert read(tmp_path / "omop", name).equals(read(outputs[0], name))

    def test_no_eras_when_turned_off(self, tmp_path: Path, athena_dir):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")

        counts = convert([zip_path], tmp_path / "omop", athena_dir, eras=False)

        assert not set(ERA_SCHEMAS) & set(counts)
        assert not any(table_path(tmp_path / "omop", t).exists() for t in ERA_SCHEMAS)
        report = json.loads(
            (tmp_path / "omop" / REPORT_FILE).read_text(encoding="utf-8")
        )
        assert STAGE_ERAS not in report["stages"]

    def test_a_run_without_eras_removes_old_era_tables(
        self, tmp_path: Path, athena_dir
    ):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        out = tmp_path / "omop"
        convert([zip_path], out, athena_dir, eras=True)

        convert([zip_path], out, athena_dir, eras=False)

        assert not any(table_path(out, t).parent.exists() for t in ERA_SCHEMAS)

    def test_a_resumed_run_builds_the_eras(self, tmp_path: Path, athena_dir):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        out = tmp_path / "omop"
        convert([zip_path], out, athena_dir, keep_work_dir=True, eras=False)

        counts = convert([zip_path], out, athena_dir, eras=True)

        assert counts[tbl.omop_drug_era] == 3
        assert not (tmp_path / "omop" / "_work" / "eras").exists()

    def test_missing_concept_ancestor_fails_before_any_work(self, tmp_path: Path):
        vocab = fx.write_athena_fixture(tmp_path / "athena", optional_tables=False)
        out = tmp_path / "omop"

        with pytest.raises(FileNotFoundError, match="CONCEPT_ANCESTOR"):
            convert([_write_zip(tmp_path / "tiny_export.zip")], out, vocab, eras=True)

        assert not out.exists()


class TestCommandLine:
    def test_flag_defaults_to_on(self):
        args = cli.parse_args(
            ["convert", "-i", "a.zip", "-o", "out", "--vocab-dir", "v"]
        )

        assert args.eras is True

    def test_no_eras_turns_it_off(self):
        args = cli.parse_args(
            ["convert", "-i", "a.zip", "-o", "out", "--vocab-dir", "v", "--no-eras"]
        )

        assert args.eras is False

    def test_flag_writes_the_era_tables(self, tmp_path: Path, athena_dir: Path):
        out = tmp_path / "omop"

        code = entry.main(
            ["convert", "-i", str(_write_zip(tmp_path / "tiny_export.zip"))]
            + ["-o", str(out), "--vocab-dir", str(athena_dir), "--eras"]
        )

        assert code == 0
        assert all(table_path(out, t).exists() for t in ERA_SCHEMAS)

    def test_missing_concept_ancestor_exits_with_1(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        vocab = fx.write_athena_fixture(tmp_path / "athena", optional_tables=False)
        out = tmp_path / "omop"

        with caplog.at_level(logging.ERROR):
            code = entry.main(
                ["convert", "-i", str(_write_zip(tmp_path / "tiny_export.zip"))]
                + ["-o", str(out), "--vocab-dir", str(vocab), "--eras"]
            )

        assert code == 1
        assert "CONCEPT_ANCESTOR.csv" in caplog.text
        assert "--no-eras" in caplog.text
        assert not out.exists()

    def test_with_no_eras_concept_ancestor_is_not_needed(self, tmp_path: Path):
        vocab = fx.write_athena_fixture(tmp_path / "athena", optional_tables=False)

        code = entry.main(
            ["convert", "-i", str(_write_zip(tmp_path / "tiny_export.zip"))]
            + ["-o", str(tmp_path / "omop"), "--vocab-dir", str(vocab), "--no-eras"]
        )

        assert code == 0
