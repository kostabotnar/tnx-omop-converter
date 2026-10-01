"""Minimal reader and writer for one worksheet of an .xlsx file (standard library only).

The reader returns the cell values of a sheet as text: shared strings (including
rich text runs), inline strings, and the stored value of numeric, boolean and
formula cells. Styles, dates and formulas are not interpreted.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import IO
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_M = f"{{{_MAIN_NS}}}"
_WORKSHEET_TYPE = f"{_REL_NS}/worksheet"

_CELL_REF = re.compile(r"([A-Z]+)(\d+)")

Source = str | Path | IO[bytes]
Cell = str | int | float


class XlsxError(ValueError):
    """The file is not a readable .xlsx workbook or has no such sheet."""


def read_sheet(source: Source, sheet_name: str) -> list[list[str]]:
    """Cell values of one sheet as rows of text.

    Rows and cells are placed by their reference (A1 notation), so missing rows
    and cells come back as empty rows and empty strings. Trailing empty cells of a
    row are not added.

    Raises:
        XlsxError: The file is not a readable workbook or has no sheet of that name.
    """
    try:
        with zipfile.ZipFile(source) as zf:
            sheets = _sheets(zf)
            sheet = next((s for s in sheets if s.get("name") == sheet_name), None)
            if sheet is None:
                names = ", ".join(repr(s.get("name")) for s in sheets)
                raise XlsxError(f"No sheet named {sheet_name!r}; sheets: {names}")
            path = _sheet_path(zf, sheet.get(f"{{{_REL_NS}}}id", ""))
            shared = _shared_strings(zf)
            root = ET.fromstring(zf.read(path))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as e:
        raise XlsxError(f"Not a readable .xlsx file: {e}") from e
    return _rows(root, shared)


def _sheets(zf: zipfile.ZipFile) -> list[ET.Element]:
    root = ET.fromstring(zf.read("xl/workbook.xml"))
    return root.findall(f"{_M}sheets/{_M}sheet")


def _sheet_path(zf: zipfile.ZipFile, rel_id: str) -> str:
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    for rel in rels.findall(f"{{{_PKG_REL_NS}}}Relationship"):
        if rel.get("Id") == rel_id:
            target = rel.get("Target", "")
            # Targets are relative to xl/ unless they start with a slash.
            if target.startswith("/"):
                return target.lstrip("/")
            return posixpath.normpath(posixpath.join("xl", target))
    raise XlsxError(f"Workbook relationship {rel_id!r} not found")


def _text(element: ET.Element) -> str:
    """Text of a string item: a plain <t> or the <t> of every rich text run.

    Phonetic runs (<rPh>) are not part of the value.
    """
    plain = element.find(f"{_M}t")
    if plain is not None:
        return plain.text or ""
    return "".join(t.text or "" for t in element.findall(f"{_M}r/{_M}t"))


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    return [_text(si) for si in ET.fromstring(data).findall(f"{_M}si")]


def _column_index(letters: str) -> int:
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    kind = cell.get("t", "n")
    if kind == "inlineStr":
        inline = cell.find(f"{_M}is")
        return _text(inline) if inline is not None else ""
    value = cell.findtext(f"{_M}v", default="")
    if kind == "s" and value:
        return shared[int(value)]
    return value


def _rows(root: ET.Element, shared: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in root.iter(f"{_M}row"):
        number = row.get("r")
        row_index = int(number) - 1 if number else len(rows)
        while len(rows) <= row_index:
            rows.append([])
        values = rows[row_index]
        for cell in row.findall(f"{_M}c"):
            match = _CELL_REF.fullmatch(cell.get("r", ""))
            column = _column_index(match.group(1)) if match else len(values)
            while len(values) <= column:
                values.append("")
            values[column] = _cell_value(cell, shared)
    return rows


def _column_letters(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return letters


def _cell_xml(ref: str, value: Cell) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        text = escape(str(value))
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'
    return f'<c r="{ref}"><v>{value!r}</v></c>'


def write_sheet(dest: Source, sheet_name: str, rows: Sequence[Sequence[Cell]]) -> None:
    """Write a workbook with one sheet; text as inline strings, numbers as numbers.

    Empty strings are written as no cell.
    """
    row_xml = []
    for r, values in enumerate(rows, start=1):
        cells = "".join(
            _cell_xml(f"{_column_letters(c)}{r}", value)
            for c, value in enumerate(values)
            if value != ""
        )
        row_xml.append(f'<row r="{r}">{cells}</row>')
    parts = {
        "[Content_Types].xml": (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" '
            'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            f'<Relationships xmlns="{_PKG_REL_NS}">'
            f'<Relationship Id="rId1" Type="{_REL_NS}/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>'
        ),
        "xl/workbook.xml": (
            f'<workbook xmlns="{_MAIN_NS}" xmlns:r="{_REL_NS}"><sheets>'
            f'<sheet name={quoteattr(sheet_name)} sheetId="1" r:id="rId1"/>'
            "</sheets></workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            f'<Relationships xmlns="{_PKG_REL_NS}">'
            f'<Relationship Id="rId1" Type="{_WORKSHEET_TYPE}" '
            'Target="worksheets/sheet1.xml"/></Relationships>'
        ),
        "xl/worksheets/sheet1.xml": (
            f'<worksheet xmlns="{_MAIN_NS}"><sheetData>{"".join(row_xml)}'
            "</sheetData></worksheet>"
        ),
    }
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, xml in parts.items():
            zf.writestr(
                name, '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + xml
            )
