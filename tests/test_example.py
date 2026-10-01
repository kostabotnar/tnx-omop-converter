"""The `tnx-omop example` command: files, conversion with the defaults, validation."""

import zipfile
from pathlib import Path

import polars as pl
import pytest

from tnx_omop import __main__ as entry
from tnx_omop import cli
from tnx_omop.omop_vocab.concept_table import table_path
from tnx_omop.schema.trinetx import load_data_dictionary
from tnx_omop.util import tables as tbl


@pytest.fixture(scope="module")
def example_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("example")
    assert entry.main(["example", str(root)]) == 0
    return root


@pytest.fixture(scope="module")
def output_dir(example_dir: Path) -> Path:
    out = example_dir / "omop"
    code = entry.main(
        [
            "convert",
            "--input",
            str(example_dir / "example_export.zip"),
            "--output",
            str(out),
            "--vocab-dir",
            str(example_dir / "athena"),
        ]
    )
    assert code == 0
    return out


def test_parses_example_command():
    args = cli.parse_args(["example", "somewhere"])

    assert args.command == cli.EXAMPLE
    assert args.output_dir == Path("somewhere")


def test_writes_export_and_vocabulary(example_dir: Path):
    with zipfile.ZipFile(example_dir / "example_export.zip") as zf:
        names = set(zf.namelist())
    athena = {p.name for p in (example_dir / "athena").iterdir()}

    assert "datadictionary.xlsx" in names
    assert {f"{tbl.tnx_patient}.csv", f"{tbl.tnx_lab_result}.csv"} <= names
    assert {
        "CONCEPT.csv",
        "CONCEPT_RELATIONSHIP.csv",
        "VOCABULARY.csv",
        "CONCEPT_ANCESTOR.csv",
    } <= athena
    assert load_data_dictionary([example_dir / "example_export.zip"]).rows


def test_creates_missing_output_folder(tmp_path: Path):
    target = tmp_path / "a" / "b"

    assert entry.main(["example", str(target)]) == 0
    assert (target / "example_export.zip").is_file()


def test_validate_passes(output_dir: Path):
    assert entry.main(["validate", str(output_dir)]) == 0


@pytest.mark.parametrize(
    "table, rows",
    [
        (tbl.omop_person, 3),
        (tbl.omop_visit_occurrence, 5),
        (tbl.omop_condition_occurrence, 5),
        (tbl.omop_procedure_occurrence, 1),
        (tbl.omop_drug_exposure, 4),
        (tbl.omop_measurement, 5),
        (tbl.omop_death, 1),
        (tbl.omop_condition_era, 5),
        (tbl.omop_drug_era, 4),
    ],
)
def test_row_counts(output_dir: Path, table: str, rows: int):
    assert pl.read_parquet(table_path(output_dir, table)).height == rows


def test_unmapped_diagnosis_is_excluded(output_dir: Path):
    excluded = pl.read_parquet(output_dir / "excluded" / f"{tbl.tnx_diagnosis}.parquet")

    assert excluded.height == 1
