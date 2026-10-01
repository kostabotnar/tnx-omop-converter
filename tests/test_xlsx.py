"""Tests for tnx_omop/util/xlsx.py."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tnx_omop.util.xlsx import XlsxError, read_sheet, write_sheet

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def _workbook(path: Path, sheets: dict[str, str], shared: str | None = None) -> None:
    """Write a workbook by hand: sheet name -> <sheetData> content.

    The first sheet goes to sheet2.xml and the second to sheet1.xml with an
    absolute target, so the reader must follow the relationships.
    """
    entries = []
    rels = []
    files = {}
    for i, (name, data) in enumerate(sheets.items(), start=1):
        part = f"worksheets/sheet{len(sheets) - i + 1}.xml"
        target = part if i == 1 else f"/xl/{part}"
        entries.append(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>')
        rels.append(
            f'<Relationship Id="rId{i}" Type="{REL}/worksheet" Target="{target}"/>'
        )
        files[f"xl/{part}"] = (
            f'<worksheet xmlns="{MAIN}"><sheetData>{data}</sheetData></worksheet>'
        )
    files["xl/workbook.xml"] = (
        f'<workbook xmlns="{MAIN}" xmlns:r="{REL}"><sheets>{"".join(entries)}'
        "</sheets></workbook>"
    )
    files["xl/_rels/workbook.xml.rels"] = (
        f'<Relationships xmlns="{PKG}">{"".join(rels)}</Relationships>'
    )
    if shared is not None:
        files["xl/sharedStrings.xml"] = f'<sst xmlns="{MAIN}">{shared}</sst>'
    with zipfile.ZipFile(path, "w") as zf:
        for name, xml in files.items():
            zf.writestr(name, xml)


def test_shared_strings_with_rich_text_runs(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    _workbook(
        path,
        {
            "Data": '<row r="1"><c r="A1" t="s"><v>1</v></c><c r="B1" t="s"><v>0</v></c></row>'
        },
        shared=(
            "<si><t>plain</t></si>"
            "<si><r><t>ri</t></r><r><rPr><b/></rPr><t>ch</t></r>"
            "<rPh><t>x</t></rPh></si>"
        ),
    )
    assert read_sheet(path, "Data") == [["rich", "plain"]]


def test_inline_numeric_and_missing_cells(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    _workbook(
        path,
        {
            "Data": (
                '<row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c>'
                '<c r="C1"><v>42</v></c></row>'
                '<row r="3"><c r="B3" t="inlineStr"><is><r><t>b</t></r>'
                "<r><t>c</t></r></is></c></row>"
            )
        },
    )
    assert read_sheet(path, "Data") == [["a", "", "42"], [], ["", "bc"]]


def test_sheet_is_found_by_name(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    _workbook(
        path,
        {
            "Empty": "",
            "Wanted": '<row r="1"><c r="A1" t="inlineStr"><is><t>x</t></is></c></row>',
        },
    )
    assert read_sheet(path, "Wanted") == [["x"]]
    assert read_sheet(path, "Empty") == []
    with pytest.raises(XlsxError, match="Missing"):
        read_sheet(path, "Missing")


def test_not_a_workbook_raises(tmp_path: Path):
    path = tmp_path / "book.xlsx"
    path.write_bytes(b"not a zip")
    with pytest.raises(XlsxError):
        read_sheet(path, "Data")


def test_written_sheet_reads_back():
    rows = [["name", "<&> ", "", "z"], [], ["x", 7, 2.5]]
    buffer = io.BytesIO()
    write_sheet(buffer, "My Sheet", rows)
    buffer.seek(0)
    assert read_sheet(buffer, "My Sheet") == [
        ["name", "<&> ", "", "z"],
        [],
        ["x", "7", "2.5"],
    ]
