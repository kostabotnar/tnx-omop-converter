"""Tests for the data dictionary of the TriNetX exports (tnx_omop/schema/trinetx.py)."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tnx_omop import __main__ as entry
from tnx_omop.schema import trinetx
from tnx_omop.converter import convert
from tnx_omop.schema.trinetx import (
    DataDictionary,
    DataDictionaryError,
    apply_data_types,
    load_data_dictionary,
    parse_data_dictionary,
    read_data_dictionary,
)
from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.util.xlsx import write_sheet
from tests.test_converter import _write_zip
from tests.tnx_dictionary import DICTIONARY, ROWS, add_dictionary, sheet_rows

CHANGED_ROWS = [(t, c, "VARCHAR" if c == col.year_of_birth else d) for t, c, d in ROWS]


def _zip(path: Path, **kwargs) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("patient.csv", "patient_id\n")
        if kwargs.get("rows", ROWS) is not None:
            add_dictionary(zf, **kwargs)
    return path


class TestParse:
    def test_blank_and_repeated_header_rows_are_skipped(self):
        cells = sheet_rows()
        assert [] in cells and cells.count(cells[0]) > 1

        assert parse_data_dictionary(cells).rows == tuple(ROWS)

    def test_header_found_below_leading_rows_and_cells_trimmed(self):
        cells = [
            ["title"],
            [],
            ["x", "Data Type", "Column/Field Name", "Table Name"],
            ["", " BIGINT ", "year_of_birth", "patient"],
            ["", "VARCHAR"],  # no table or column
        ]
        assert parse_data_dictionary(cells).rows == (
            ("patient", "year_of_birth", "BIGINT"),
        )

    def test_no_header_raises(self):
        with pytest.raises(DataDictionaryError, match="no header row"):
            parse_data_dictionary([["a", "b"], ["c", "d"]])

    def test_no_rows_raises(self):
        with pytest.raises(DataDictionaryError, match="no columns"):
            parse_data_dictionary(sheet_rows([]))


class TestTypes:
    def test_types_match_dictionary(self):
        schema = DICTIONARY.schema(tbl.tnx_patient)
        assert schema[col.year_of_birth] == pl.Int64
        assert schema[col.patient_id] == pl.Utf8
        assert DICTIONARY.schema(tbl.tnx_lab_result)[col.lab_result_num_val] == (
            pl.Float64
        )
        assert DICTIONARY.schema("no_such_table") == {}

    def test_date_columns_in_dictionary_order(self):
        assert DICTIONARY.date_columns(tbl.tnx_encounter) == [
            col.start_date,
            col.end_date,
        ]
        assert DICTIONARY.date_columns(tbl.tnx_patient) == []

    def test_apply_data_types_follows_dictionary(self):
        lf = pl.LazyFrame(
            {col.year_of_birth: ["1950", "x"], col.start_date: ["20230102", "bad"]}
        )
        dictionary = DataDictionary(
            (
                ("t", col.year_of_birth, "BIGINT"),
                ("t", col.start_date, "DATE"),
            )
        )
        typed = apply_data_types(lf, "t", dictionary).collect()
        assert typed.schema == {col.year_of_birth: pl.Int64, col.start_date: pl.Date}
        assert typed.row(0) == (1950, date(2023, 1, 2))
        assert typed.row(1) == (None, None)

        untyped = apply_data_types(lf, "other", dictionary).collect()
        assert untyped.schema == {col.year_of_birth: pl.Utf8, col.start_date: pl.Utf8}

    def test_unknown_data_type_stays_string(self):
        dictionary = DataDictionary((("t", "c", "BLOB"),))
        assert dictionary.schema("t") == {}


class TestReadFromZip:
    def test_reads_workbook(self, tmp_path: Path):
        assert read_data_dictionary(_zip(tmp_path / "a.zip")) == DICTIONARY

    def test_member_matched_case_insensitively_in_subfolder(self, tmp_path: Path):
        path = _zip(tmp_path / "a.zip", folder="export/", name="DataDictionary.XLSX")
        assert read_data_dictionary(path) == DICTIONARY

    def test_no_member_gives_none(self, tmp_path: Path):
        assert read_data_dictionary(_zip(tmp_path / "a.zip", rows=None)) is None

    def test_unreadable_workbook_raises(self, tmp_path: Path):
        path = tmp_path / "a.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(trinetx.DATA_DICTIONARY_FILE, b"not a workbook")
        with pytest.raises(DataDictionaryError, match="a.zip"):
            read_data_dictionary(path)

    def test_missing_sheet_raises(self, tmp_path: Path):
        buffer = io.BytesIO()
        write_sheet(buffer, "Other", sheet_rows())
        path = tmp_path / "a.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(trinetx.DATA_DICTIONARY_FILE, buffer.getvalue())
        with pytest.raises(DataDictionaryError, match="Data Dictionary"):
            read_data_dictionary(path)


class TestLoad:
    def test_identical_dictionaries_accepted(self, tmp_path: Path):
        first = _zip(tmp_path / "a.zip")
        second = _zip(tmp_path / "b.zip", rows=list(reversed(ROWS)))

        assert load_data_dictionary([first, second]) == DICTIONARY

    def test_missing_dictionary_names_zip_files(self, tmp_path: Path):
        zips = [
            _zip(tmp_path / "a.zip"),
            _zip(tmp_path / "b.zip", rows=None),
            _zip(tmp_path / "c.zip", rows=None),
        ]
        with pytest.raises(
            DataDictionaryError, match="No datadictionary.xlsx in b.zip, c.zip"
        ):
            load_data_dictionary(zips)

    def test_differing_dictionaries_show_rows(self, tmp_path: Path):
        zips = [_zip(tmp_path / "a.zip"), _zip(tmp_path / "b.zip", rows=CHANGED_ROWS)]

        with pytest.raises(DataDictionaryError) as error:
            load_data_dictionary(zips)

        message = str(error.value)
        assert "a.zip and b.zip differ" in message
        assert "only in a.zip (1 rows): patient, year_of_birth, BIGINT" in message
        assert "only in b.zip (1 rows): patient, year_of_birth, VARCHAR" in message


class TestConvertChecks:
    def test_convert_raises_before_any_output(self, tmp_path: Path, athena_dir: Path):
        zips = [
            _write_zip(tmp_path / "a.zip"),
            _write_zip(tmp_path / "b.zip", dictionary=CHANGED_ROWS),
        ]
        out = tmp_path / "omop"

        with pytest.raises(DataDictionaryError, match="differ"):
            convert(zips, out, athena_dir)
        assert not out.exists()

    def test_convert_raises_without_dictionary(self, tmp_path: Path, athena_dir: Path):
        out = tmp_path / "omop"
        with pytest.raises(DataDictionaryError, match="tiny_export.zip"):
            convert([_write_zip(tmp_path / "tiny_export.zip", None)], out, athena_dir)
        assert not out.exists()

    @pytest.mark.parametrize("second_rows", [CHANGED_ROWS, None])
    def test_cli_exits_1_before_any_output(
        self, tmp_path: Path, athena_dir: Path, caplog, second_rows
    ):
        first = _write_zip(tmp_path / "a.zip")
        second = _write_zip(tmp_path / "b.zip", dictionary=second_rows)
        out = tmp_path / "omop"

        code = entry.main(
            ["convert", "-i", str(first), str(second), "-o", str(out)]
            + ["--vocab-dir", str(athena_dir)]
        )

        assert code == 1
        assert "b.zip" in caplog.text
        assert not out.exists()

    def test_convert_types_come_from_the_zip(self, tmp_path: Path, athena_dir: Path):
        out = tmp_path / "omop"
        convert(
            [_write_zip(tmp_path / "tiny_export.zip", dictionary=CHANGED_ROWS)],
            out,
            athena_dir,
            export_vocabulary=False,
            eras=False,
            keep_work_dir=True,
        )
        patient = pl.read_parquet(out / "_work" / "batches" / tbl.tnx_patient)
        assert patient.schema[col.year_of_birth] == pl.Utf8
