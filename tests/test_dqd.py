"""Tests for tnx_omop/dqd.py and the `dqd` subcommand.

R is replaced by a Python script that stands in for run_dqd.R, so these tests run
without R. `TestWithR` runs the real script and is skipped unless Rscript and the
R packages are installed.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tnx_omop import __main__ as entry
from tnx_omop import cli, dqd
from tnx_omop.converter import convert
from tnx_omop.util import tables as tbl
from tests.test_converter import _write_zip


def check_row(check_id: str, **values) -> dict:
    """A DQD CheckResults row that passed, with values overridden."""
    row = {
        "checkId": check_id,
        "checkName": check_id.split("_")[1],
        "cdmTableName": "PERSON",
        "cdmFieldName": "PERSON_ID",
        "category": "Conformance",
        "subcategory": "Relational",
        "numViolatedRows": 0,
        "numDenominatorRows": 10,
        "pctViolatedRows": 0,
        "thresholdValue": 0,
        "failed": 0,
        "passed": 1,
        "isError": 0,
        "notApplicable": 0,
    }
    return row | values


FAILED = {"failed": 1, "passed": 0, "numViolatedRows": 3, "pctViolatedRows": 0.3}
ERROR = {"isError": 1, "passed": 0, "error": "Catalog Error: x\nmore detail"}
NOT_APPLICABLE = {"notApplicable": 1, "passed": 0}

RESULTS = [
    check_row("field_isrequired_person_person_id"),
    check_row("field_isrequired_person_gender_concept_id", **FAILED),
    check_row("table_measurepersoncompleteness_provider", **FAILED),
    check_row("field_plausiblevalue_person_year_of_birth", **ERROR),
    check_row("table_cdmtable_note", **NOT_APPLICABLE),
]


def write_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_accepted(path: Path, entries: list[dict]) -> Path:
    return write_json(path, {"accepted": entries})


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory, athena_dir: Path) -> Path:
    root = tmp_path_factory.mktemp("dqd")
    out = root / "omop"
    # Without the vocabulary export and the eras, so that run_dqd warns about both
    convert(
        [_write_zip(root / "tiny_export.zip")],
        out,
        athena_dir,
        export_vocabulary=False,
        eras=False,
    )
    return out


@pytest.fixture
def fake_r(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Replace run_dqd.R with a Python script; returns a function that writes it.

    The script records its arguments in args.json next to itself, prints a line,
    writes RESULTS to the results path and exits with the given status.
    """

    def make(status: int = 0, write_results: bool = True) -> Path:
        script = tmp_path / "fake_run_dqd.py"
        script.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "Path(__file__).with_name('args.json').write_text(json.dumps(sys.argv[1:]))\n"
            "print('fake R output')\n"
            f"if {write_results!r}:\n"
            f"    Path(sys.argv[2]).write_text(json.dumps({{'CheckResults': {RESULTS!r}}}))\n"
            f"sys.exit({status})\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(dqd, "R_SCRIPT", script)
        return script

    return make


class TestLoadResults:
    def test_reads_the_fields(self, tmp_path: Path):
        path = write_json(tmp_path / "r.json", {"CheckResults": RESULTS})

        checks = dqd.load_results(path)

        assert [c.check_id for c in checks] == [r["checkId"] for r in RESULTS]
        failed = checks[1]
        assert failed.failed and not failed.is_error
        assert (failed.violated_rows, failed.denominator_rows) == (3, 10)
        assert failed.pct_violated == 0.3 and failed.field == "PERSON_ID"
        assert checks[3].is_error and checks[4].not_applicable

    def test_r_missing_values_become_none(self, tmp_path: Path):
        row = check_row("table_x_y", thresholdValue="NA", cdmFieldName="NA")
        del row["numViolatedRows"]
        path = write_json(tmp_path / "r.json", {"CheckResults": [row]})

        (check,) = dqd.load_results(path)

        assert check.threshold is None
        assert check.field is None
        assert check.violated_rows is None

    @pytest.mark.parametrize("content", ["not json", "[]", '{"CheckResults": 1}'])
    def test_bad_file_raises(self, tmp_path: Path, content: str):
        path = tmp_path / "r.json"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(dqd.DqdError):
            dqd.load_results(path)


class TestLoadAccepted:
    def test_bundled_file_is_valid(self):
        assert isinstance(dqd.load_accepted(dqd.ACCEPTED_FILE), dict)

    def test_patterns_are_lowercased(self, tmp_path: Path):
        path = write_accepted(
            tmp_path / "a.json", [{"check_id": "TABLE_*_Provider", "reason": "none"}]
        )

        assert dqd.load_accepted(path) == {"table_*_provider": "none"}

    @pytest.mark.parametrize(
        "entries",
        [[{"check_id": "x"}], [{"reason": "r"}], [{"check_id": "x", "reason": " "}]],
        ids=["no reason", "no check_id", "blank reason"],
    )
    def test_incomplete_entry_raises(self, tmp_path: Path, entries: list[dict]):
        path = write_accepted(tmp_path / "a.json", entries)

        with pytest.raises(dqd.DqdError, match="entry 0"):
            dqd.load_accepted(path)

    def test_missing_list_raises(self, tmp_path: Path):
        path = write_json(tmp_path / "a.json", {"accept": []})

        with pytest.raises(dqd.DqdError, match="'accepted' list"):
            dqd.load_accepted(path)


class TestSummarize:
    @pytest.fixture
    def checks(self, tmp_path: Path) -> list[dqd.DqdCheck]:
        return dqd.load_results(
            write_json(tmp_path / "r.json", {"CheckResults": RESULTS})
        )

    def test_sorts_by_outcome(self, checks):
        summary = dqd.summarize(checks, {})

        assert [c.check_id for c in summary.passed] == [RESULTS[0]["checkId"]]
        assert len(summary.failed) == 2
        assert [c.check_id for c in summary.errors] == [RESULTS[3]["checkId"]]
        assert len(summary.not_applicable) == 1
        assert not summary.ok

    def test_accepted_patterns_match_failures_and_errors(self, checks):
        accepted = {
            "table_measurepersoncompleteness_*": "no providers",
            "field_isrequired_person_gender_concept_id": "known",
            "field_plausiblevalue_*": "DuckDB",
            "field_unused_*": "stale",
        }

        summary = dqd.summarize(checks, accepted)

        assert summary.ok
        assert len(summary.accepted) == 3
        assert summary.unused_patterns == ["field_unused_*"]

    def test_passed_checks_do_not_use_patterns(self, checks):
        summary = dqd.summarize(checks, {"field_isrequired_person_person_id": "x"})

        assert summary.unused_patterns == ["field_isrequired_person_person_id"]

    def test_format(self, checks):
        text = dqd.format_summary(dqd.summarize(checks, {}))

        assert (
            "FAILED field_isrequired_person_gender_concept_id: PERSON.PERSON_ID, "
            "3 of 10 rows (30.00%), threshold 0% [Conformance/Relational]"
        ) in text
        assert (
            "ERROR field_plausiblevalue_person_year_of_birth: Catalog Error: x\n"
            in (text)
        )
        assert text.splitlines()[-1] == (
            "1 passed, 2 failed, 1 errors, 0 accepted, 1 not applicable"
        )


class TestOutputTables:
    def test_lists_present_tables_in_lowercase(self, converted: Path):
        found = dqd.output_tables(converted)

        assert found["person"].exists()
        assert {"cdm_source", "observation_period", "concept"} <= set(found)
        assert tbl.omop_concept_ancestor not in found


class TestRunDqd:
    def test_passes_database_results_and_tables(
        self, converted: Path, tmp_path: Path, fake_r, caplog
    ):
        script = fake_r()
        results_dir = tmp_path / "results"

        with caplog.at_level("INFO"):
            path = dqd.run_dqd(
                converted, results_dir, sys.executable, source_name="Src"
            )

        args = json.loads(script.with_name("args.json").read_text())
        assert path == results_dir / dqd.RESULTS_FILE and path.exists()
        assert args[:4] == [
            str(results_dir / dqd.DATABASE_FILE),
            str(path),
            "Src",
            "false",
        ]
        assert f"person={converted / 'person' / 'person.parquet'}" in args[4:]
        assert "R: fake R output" in caplog.text
        assert "--no-export-vocabulary" in caplog.text
        assert "--no-eras" in caplog.text

    def test_missing_packages(self, converted: Path, tmp_path: Path, fake_r):
        fake_r(status=dqd.EXIT_MISSING_PACKAGES)

        with pytest.raises(dqd.DqdError, match="--install-r-packages"):
            dqd.run_dqd(converted, tmp_path, sys.executable)

    def test_failure_repeats_r_output_when_quiet(
        self, converted: Path, tmp_path: Path, fake_r, caplog
    ):
        fake_r(status=1)

        with caplog.at_level("WARNING"), pytest.raises(dqd.DqdError, match="code 1"):
            dqd.run_dqd(converted, tmp_path, sys.executable)

        assert "R: fake R output" in caplog.text

    def test_no_results_file(self, converted: Path, tmp_path: Path, fake_r):
        fake_r(write_results=False)

        with pytest.raises(dqd.DqdError, match="no results file"):
            dqd.run_dqd(converted, tmp_path, sys.executable)

    def test_missing_required_table(self, tmp_path: Path, fake_r):
        fake_r()

        with pytest.raises(dqd.DqdError, match="PERSON"):
            dqd.run_dqd(tmp_path, tmp_path / "results", sys.executable)


class TestDqdCommand:
    def test_parsing_defaults(self):
        args = cli.parse_args(["dqd", "out"])

        assert args.command == cli.DQD
        assert args.output_dir == Path("out")
        assert args.results_dir is None and args.rscript is None
        assert args.accepted == dqd.ACCEPTED_FILE
        assert args.source_name == dqd.DEFAULT_SOURCE_NAME
        assert not args.keep_database and not args.no_run

    def test_runs_and_exits_1_on_failures(
        self, converted: Path, tmp_path: Path, fake_r, capsys
    ):
        fake_r()
        results_dir = tmp_path / "results"

        code = entry.main(
            ["dqd", str(converted), "--rscript", sys.executable]
            + ["--results-dir", str(results_dir), "--keep-database"]
        )

        assert code == 1
        out = capsys.readouterr().out
        assert out.splitlines()[-1].startswith("1 passed, 2 failed, 1 errors")

    def test_no_run_with_accepted_failures_exits_0(
        self, tmp_path: Path, capsys, caplog
    ):
        write_json(
            tmp_path / dqd.RESULTS_DIR / dqd.RESULTS_FILE, {"CheckResults": RESULTS}
        )
        accepted = write_accepted(
            tmp_path / "accepted.json",
            [
                {"check_id": "*_measurepersoncompleteness_*", "reason": "a"},
                {"check_id": "*_isrequired_*", "reason": "b"},
                {"check_id": "*_plausiblevalue_*", "reason": "c"},
                {"check_id": "*_nothing_*", "reason": "d"},
            ],
        )

        code = entry.main(
            ["dqd", str(tmp_path), "--no-run", "--accepted", str(accepted)]
        )

        assert code == 0
        assert "3 accepted" in capsys.readouterr().out
        assert "matches no failure: *_nothing_*" in caplog.text

    def test_no_run_without_results_exits_1(self, tmp_path: Path, caplog):
        assert entry.main(["dqd", str(tmp_path), "--no-run"]) == 1
        assert "No DQD results" in caplog.text

    def test_missing_rscript_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ):
        monkeypatch.setattr(shutil, "which", lambda name: None)

        assert entry.main(["dqd", str(tmp_path)]) == 1
        assert "Rscript not found" in caplog.text

    def test_install_r_packages_runs_the_script(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ):
        script = tmp_path / "fake_install.py"
        script.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "Path(__file__).with_name('args.json').write_text(json.dumps(sys.argv[1:]))\n"
            "print('fake install output')\n"
            "sys.exit(7)\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(dqd, "INSTALL_SCRIPT", script)

        with caplog.at_level("INFO"):
            code = entry.main(
                ["dqd", "--install-r-packages", "--rscript", sys.executable]
            )

        assert code == 7
        assert json.loads(script.with_name("args.json").read_text()) == []
        assert "R: fake install output" in caplog.text

    def test_install_r_packages_uses_rscript_on_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        script = tmp_path / "ok.py"
        script.write_text("", encoding="utf-8")
        monkeypatch.setattr(dqd, "INSTALL_SCRIPT", script)
        monkeypatch.setattr(shutil, "which", lambda name: sys.executable)

        assert entry.main(["dqd", "--install-r-packages"]) == 0

    def test_install_r_packages_without_rscript_exits_1(
        self, monkeypatch: pytest.MonkeyPatch, caplog
    ):
        monkeypatch.setattr(shutil, "which", lambda name: None)

        assert entry.main(["dqd", "--install-r-packages"]) == 1
        assert "Rscript not found" in caplog.text

    def test_missing_output_folder_argument_exits_1(self, caplog):
        assert entry.main(["dqd"]) == 1
        assert "Output folder missing" in caplog.text

    def test_missing_folder_exits_1(self, tmp_path: Path, caplog):
        assert entry.main(["dqd", str(tmp_path / "nowhere")]) == 1
        assert "Output folder not found" in caplog.text


def _r_ready() -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    packages = (
        'c("duckdb", "DatabaseConnector", "CommonDataModel", "DataQualityDashboard")'
    )
    probe = f"quit(status = !all(sapply({packages}, requireNamespace, quietly = TRUE)))"
    return subprocess.run([rscript, "-e", probe], capture_output=True).returncode == 0


@pytest.mark.skipif(not _r_ready(), reason="Rscript or the DQD R packages missing")
class TestWithR:
    def test_output_loads_and_every_check_runs(
        self, tmp_path_factory: pytest.TempPathFactory, athena_dir: Path
    ):
        root = tmp_path_factory.mktemp("dqd_r")
        out = root / "omop"
        convert([_write_zip(root / "tiny_export.zip")], out, athena_dir)

        path = dqd.run_dqd(out, root / "results", shutil.which("Rscript"))

        summary = dqd.summarize(dqd.load_results(path), {})
        assert summary.passed
        assert summary.errors == [], dqd.format_summary(summary)
