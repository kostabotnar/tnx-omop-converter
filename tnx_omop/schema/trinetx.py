"""Column types of the TriNetX tables from the data dictionary of the export.

Every TriNetX export ZIP has a workbook datadictionary.xlsx whose "Data Dictionary"
sheet lists each table column with its data type. load_data_dictionary() reads it
from every input ZIP, checks that all ZIP files agree, and returns a
DataDictionary that ingest uses to type the CSV columns.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import polars as pl

from ..util.xlsx import XlsxError, read_sheet

DATA_DICTIONARY_FILE = "datadictionary.xlsx"
DATA_DICTIONARY_SHEET = "Data Dictionary"
TABLE_HEADER = "Table Name"
FIELD_HEADER = "Column/Field Name"
TYPE_HEADER = "Data Type"

# Rows shown per ZIP file when the dictionaries differ
_SHOWN_DIFFERENCES = 5

# Type mapping from data dictionary types to Polars types
_TYPE_MAP: dict[str, pl.DataType] = {
    "VARCHAR": pl.Utf8,
    "DATE": pl.Utf8,  # Read as YYYYMMDD strings; parse_date_columns makes them Date
    "BIGINT": pl.Int64,
    "DECIMAL": pl.Float64,
    "BOOLEAN": pl.Utf8,  # Values are "T"/"F" strings
}

_DATE_TYPE = "DATE"
TNX_DATE_FORMAT = "%Y%m%d"

DictionaryRow = tuple[str, str, str]  # table, column, data type


class DataDictionaryError(ValueError):
    """An input ZIP has no usable data dictionary, or the ZIP files disagree."""


@dataclass(frozen=True)
class DataDictionary:
    """The (table, column, data type) rows of a TriNetX data dictionary, in file order."""

    rows: tuple[DictionaryRow, ...]

    def schema(self, table_name: str) -> dict[str, pl.DataType]:
        """Polars types of the columns of a table; empty for an unknown table.

        Columns whose data type is not known keep no entry and stay strings.
        """
        return {
            column: _TYPE_MAP[data_type]
            for table, column, data_type in self.rows
            if table == table_name and data_type in _TYPE_MAP
        }

    def date_columns(self, table_name: str) -> list[str]:
        """Columns of a table with data type DATE (YYYYMMDD strings in the CSV)."""
        return [
            column
            for table, column, data_type in self.rows
            if table == table_name and data_type == _DATE_TYPE
        ]


def parse_data_dictionary(rows: Sequence[Sequence[str]]) -> DataDictionary:
    """Build a DataDictionary from the cell rows of the "Data Dictionary" sheet.

    The first row with the Table Name, Column/Field Name and Data Type headers is
    the header. Blank rows, rows without a table or column name, and repeated
    header rows are skipped.

    Raises:
        DataDictionaryError: There is no header row or no data row.
    """
    headers = (TABLE_HEADER, FIELD_HEADER, TYPE_HEADER)
    start = next(
        (i for i, row in enumerate(rows) if all(h in row for h in headers)), None
    )
    if start is None:
        raise DataDictionaryError(f"no header row with {', '.join(headers)}")
    positions = [list(rows[start]).index(h) for h in headers]
    entries = []
    for row in rows[start + 1 :]:
        table, column, data_type = (
            row[p].strip() if p < len(row) else "" for p in positions
        )
        if table and column and table != TABLE_HEADER:
            entries.append((table, column, data_type))
    if not entries:
        raise DataDictionaryError("no columns")
    return DataDictionary(tuple(entries))


def _dictionary_member(zf: zipfile.ZipFile) -> zipfile.ZipInfo | None:
    """The data dictionary member, matched case-insensitively; the shallowest wins."""
    found = [
        info
        for info in zf.infolist()
        if not info.is_dir()
        and PurePosixPath(info.filename).name.lower() == DATA_DICTIONARY_FILE
    ]
    return min(found, key=lambda i: len(PurePosixPath(i.filename).parts), default=None)


def read_data_dictionary(zip_path: Path) -> DataDictionary | None:
    """Read the data dictionary of one TriNetX ZIP, or None when it has none.

    Raises:
        DataDictionaryError: The ZIP or its workbook cannot be read, or the sheet
            has no usable rows.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            info = _dictionary_member(zf)
            if info is None:
                return None
            workbook = zf.read(info)
        return parse_data_dictionary(
            read_sheet(io.BytesIO(workbook), DATA_DICTIONARY_SHEET)
        )
    except (zipfile.BadZipFile, XlsxError, DataDictionaryError) as e:
        raise DataDictionaryError(
            f"Cannot read the data dictionary ({DATA_DICTIONARY_FILE}) "
            f"of {zip_path.name}: {e}"
        ) from e


def load_data_dictionary(input_zips: Sequence[Path]) -> DataDictionary:
    """Read the data dictionary of every input ZIP and return that of the first one.

    The dictionaries are compared on table, column and data type, regardless of
    row order.

    Raises:
        DataDictionaryError: A ZIP has no readable data dictionary, or the
            dictionaries of the ZIP files differ.
    """
    if not input_zips:
        raise DataDictionaryError("No input ZIP files")
    dictionaries: dict[Path, DataDictionary] = {}
    missing = []
    for path in input_zips:
        dictionary = read_data_dictionary(path)
        if dictionary is None:
            missing.append(path.name)
        else:
            dictionaries[path] = dictionary
    if missing:
        raise DataDictionaryError(
            f"No {DATA_DICTIONARY_FILE} in {', '.join(missing)}: the column types "
            "of a TriNetX export come from its data dictionary"
        )
    first_path, first = next(iter(dictionaries.items()))
    reference = set(first.rows)
    for path, dictionary in dictionaries.items():
        rows = set(dictionary.rows)
        if rows == reference:
            continue
        lines = [
            f"The data dictionaries of {first_path.name} and {path.name} differ; "
            "all input ZIP files must come from exports with the same data dictionary"
        ]
        for name, only in (
            (first_path.name, reference - rows),
            (path.name, rows - reference),
        ):
            shown = sorted(only)[:_SHOWN_DIFFERENCES]
            more = len(only) - len(shown)
            lines.append(
                f"  only in {name} ({len(only)} rows): "
                + ("; ".join(", ".join(row) for row in shown) or "none")
                + (f"; and {more} more" if more else "")
            )
        raise DataDictionaryError("\n".join(lines))
    return first


def drop_derived_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Drop TriNetX derivation flag columns, keeping all rows.

    Removes every column whose name contains "derived_by_TriNetX"
    (case-insensitive), such as derived_by_TriNetX and
    start_date_derived_by_TriNetX. Derived records are real data that other
    tables reference, so the rows stay.
    """
    names = lf.collect_schema().names()
    return lf.drop([c for c in names if "derived_by_trinetx" in c.lower()])


def parse_date_columns(
    lf: pl.LazyFrame, table_name: str, dictionary: DataDictionary
) -> pl.LazyFrame:
    """Parse the data dictionary DATE columns of a table from YYYYMMDD to Date.

    Unparseable values become null.
    """
    present = set(lf.collect_schema().names())
    date_columns = [c for c in dictionary.date_columns(table_name) if c in present]
    return lf.with_columns(
        pl.col(c).str.to_date(format=TNX_DATE_FORMAT, strict=False)
        for c in date_columns
    )


def apply_data_types(
    lf: pl.LazyFrame, table_name: str, dictionary: DataDictionary
) -> pl.LazyFrame:
    """Type a table read with every column as a string, using the data dictionary.

    BIGINT and DECIMAL columns are cast to Int64 and Float64, DATE columns are
    parsed to Date; unparseable values become null. Other columns, and columns
    missing from the dictionary, stay strings.
    """
    overrides = dictionary.schema(table_name)
    present = lf.collect_schema().names()
    numeric = [
        pl.col(c).cast(overrides[c], strict=False)
        for c in present
        if overrides.get(c) in (pl.Int64, pl.Float64)
    ]
    return parse_date_columns(lf.with_columns(numeric), table_name, dictionary)
