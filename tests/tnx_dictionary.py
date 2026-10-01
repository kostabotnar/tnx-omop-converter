"""Synthetic TriNetX data dictionary for the test exports.

Only the columns the test fixtures use, with the data types of the TriNetX data
model. Columns that are not listed stay strings, like VARCHAR columns.
"""

from __future__ import annotations

import io
import zipfile

from tnx_omop.tnx_schema import (
    DATA_DICTIONARY_FILE,
    DATA_DICTIONARY_SHEET,
    FIELD_HEADER,
    TABLE_HEADER,
    TYPE_HEADER,
    DataDictionary,
    DictionaryRow,
)
from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.util.xlsx import write_sheet

_DERIVED = (col.derived_by_TriNetX, "BOOLEAN")
_CLINICAL = [(col.patient_id, "VARCHAR"), (col.encounter_id, "VARCHAR")]
_CODED = [*_CLINICAL, (col.code_system, "VARCHAR"), (col.code, "VARCHAR")]

_COLUMNS: dict[str, list[tuple[str, str]]] = {
    tbl.tnx_patient: [
        (col.patient_id, "VARCHAR"),
        (col.sex, "VARCHAR"),
        (col.year_of_birth, "BIGINT"),
        (col.month_year_death, "BIGINT"),
    ],
    tbl.tnx_encounter: [
        (col.encounter_id, "VARCHAR"),
        (col.patient_id, "VARCHAR"),
        (col.start_date, "DATE"),
        (col.end_date, "DATE"),
        (col.type, "VARCHAR"),
        _DERIVED,
    ],
    tbl.tnx_diagnosis: [*_CODED, (col.date, "DATE"), _DERIVED],
    tbl.tnx_procedure: [*_CODED, (col.date, "DATE"), _DERIVED],
    tbl.tnx_medication_ingredient: [*_CODED, (col.start_date, "DATE"), _DERIVED],
    tbl.tnx_lab_result: [
        *_CODED,
        (col.date, "DATE"),
        (col.lab_result_num_val, "DECIMAL"),
        (col.lab_result_text_val, "VARCHAR"),
        (col.units_of_measure, "VARCHAR"),
        _DERIVED,
    ],
    tbl.tnx_vitals_signs: [
        *_CODED,
        (col.date, "DATE"),
        (col.value, "DECIMAL"),
        (col.text_value, "VARCHAR"),
        (col.units_of_measure, "VARCHAR"),
        _DERIVED,
    ],
}

ROWS: list[DictionaryRow] = [
    (table, column, data_type)
    for table, columns in _COLUMNS.items()
    for column, data_type in columns
]
DICTIONARY = DataDictionary(tuple(ROWS))

_HEADER = ["Data Category Name", TABLE_HEADER, FIELD_HEADER, TYPE_HEADER]


def sheet_rows(rows: list[DictionaryRow] = ROWS) -> list[list[str]]:
    """Sheet cells like the TriNetX workbook: a header, then each table after a
    blank row and a repeated header row."""
    cells: list[list[str]] = [_HEADER]
    table = None
    for row in rows:
        if table is not None and row[0] != table:
            cells += [[], _HEADER]
        table = row[0]
        cells.append(["Clinical", *row])
    return cells


def dictionary_xlsx(rows: list[DictionaryRow] = ROWS) -> bytes:
    buffer = io.BytesIO()
    write_sheet(buffer, DATA_DICTIONARY_SHEET, sheet_rows(rows))
    return buffer.getvalue()


def add_dictionary(
    zf: zipfile.ZipFile,
    folder: str = "",
    rows: list[DictionaryRow] = ROWS,
    name: str = DATA_DICTIONARY_FILE,
) -> None:
    """Write a data dictionary workbook into an open TriNetX ZIP."""
    zf.writestr(f"{folder}{name}", dictionary_xlsx(rows))
