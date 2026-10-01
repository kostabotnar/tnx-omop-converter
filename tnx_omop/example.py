"""Write a tiny synthetic TriNetX export and an Athena vocabulary excerpt.

`write_example(output_dir)` creates `example_export.zip` and an `athena` folder that
`tnx-omop convert` accepts with its default options. The patients, encounters,
records and dates in the export are invented. The `athena` folder is not a full
Athena download: it holds the real Athena rows (concept IDs, names, codes) of only
the concepts the example uses, and the relationships between them.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

from .omop_vocab.athena import CONCEPT_COLUMNS
from .tnx_schema import (
    DATA_DICTIONARY_FILE,
    DATA_DICTIONARY_SHEET,
    FIELD_HEADER,
    TABLE_HEADER,
    TYPE_HEADER,
)
from .util import columns as col
from .util import tables as tbl
from .util.xlsx import write_sheet

ZIP_NAME = "example_export.zip"
ATHENA_DIR = "athena"

Table = tuple[list[tuple[str, str]], list[list[str]]]  # (column, type) list, rows

_CLINICAL = [("patient_id", "VARCHAR"), ("encounter_id", "VARCHAR")]
_CODED = [*_CLINICAL, ("code_system", "VARCHAR"), ("code", "VARCHAR")]

TNX_TABLES: dict[str, Table] = {
    tbl.tnx_patient: (
        [
            ("patient_id", "VARCHAR"),
            ("sex", "VARCHAR"),
            ("race", "VARCHAR"),
            ("ethnicity", "VARCHAR"),
            ("year_of_birth", "BIGINT"),
            ("month_year_death", "BIGINT"),
        ],
        [
            ["P1", "M", "White", "Not Hispanic or Latino", "1958", ""],
            ["P2", "F", "Black or African American", "Not Hispanic or Latino", "1971", ""],
            ["P3", "M", "Asian", "Hispanic or Latino", "1949", "202206"],
        ],
    ),
    tbl.tnx_encounter: (
        [
            ("encounter_id", "VARCHAR"),
            ("patient_id", "VARCHAR"),
            ("start_date", "DATE"),
            ("end_date", "DATE"),
            ("type", "VARCHAR"),
        ],
        [
            ["E1", "P1", "20220110", "20220110", "AMB"],
            ["E2", "P1", "20220412", "20220412", "AMB"],
            ["E3", "P2", "20220215", "20220215", "AMB"],
            ["E4", "P3", "20220301", "20220304", "IMP"],
            ["E5", "P3", "20220320", "20220320", "EMER"],
        ],
    ),
    tbl.tnx_diagnosis: (
        [*_CODED, ("date", "DATE")],
        [
            ["P1", "E1", "ICD-10-CM", "I10", "20220110"],
            ["P1", "E1", "ICD-10-CM", "E11.9", "20220110"],
            ["P2", "E3", "ICD-10-CM", "I10", "20220215"],
            ["P2", "E3", "ICD-10-CM", "E78.5", "20220215"],
            ["P3", "E4", "ICD-10-CM", "E11.9", "20220301"],
            # Not in the example vocabulary: ends up in excluded/
            ["P3", "E5", "ICD-10-CM", "Z00.00", "20220320"],
        ],
    ),
    tbl.tnx_procedure: (
        [*_CODED, ("date", "DATE")],
        [["P1", "E1", "ICD-10-PCS", "4A02X4Z", "20220110"]],
    ),
    tbl.tnx_medication_ingredient: (
        [*_CODED, ("start_date", "DATE"), ("route", "VARCHAR")],
        [
            ["P1", "E1", "RxNorm", "6809", "20220110", "Oral Product"],
            ["P1", "E2", "RxNorm", "29046", "20220412", "Oral Product"],
            ["P2", "E3", "RxNorm", "29046", "20220215", "Oral Product"],
            ["P3", "E4", "RxNorm", "6809", "20220301", "Oral Product"],
        ],
    ),
    tbl.tnx_lab_result: (
        [
            *_CODED,
            ("date", "DATE"),
            ("lab_result_num_val", "DECIMAL"),
            ("lab_result_text_val", "VARCHAR"),
            ("units_of_measure", "VARCHAR"),
        ],
        [
            ["P1", "E1", "LOINC", "2345-7", "20220110", "156", "", "mg/dL"],
            ["P2", "E3", "LOINC", "2093-3", "20220215", "214", "", "mg/dL"],
            ["P2", "E3", "LOINC", "2089-1", "20220215", "131", "", "mg/dL"],
        ],
    ),
    tbl.tnx_vitals_signs: (
        [
            *_CODED,
            ("date", "DATE"),
            ("value", "DECIMAL"),
            ("text_value", "VARCHAR"),
            ("units_of_measure", "VARCHAR"),
        ],
        [
            ["P1", "E1", "LOINC", "8480-6", "20220110", "142", "", "mm[Hg]"],
            ["P1", "E1", "LOINC", "8462-4", "20220110", "88", "", "mm[Hg]"],
        ],
    ),
    tbl.tnx_dataset_details: (
        [
            ("total_number_unique_patients", "BIGINT"),
            ("date_created", "DATE"),
            ("network_name", "VARCHAR"),
        ],
        [["3", "20220601", "Synthetic example"]],
    ),
    tbl.tnx_cohort_details: (
        [("cohort_name", "VARCHAR"), ("total_patient_records", "BIGINT")],
        [["Example cohort", "3"]],
    ),
}  # fmt: skip

# concept_id, name, domain, vocabulary, class, standard_concept, code, valid start,
# valid end: the rows of the Athena CONCEPT table for the concepts the example uses
Concept = tuple[int, str, str, str, str, str, str, str, str]

CONCEPTS: list[Concept] = [
    (756265, "OMOP CDM Version 5.4.0", "Metadata", "CDM", "CDM", "S", "CDM v5.4.0", "20210925", "20991231"),
    (38003563, "Hispanic or Latino", "Ethnicity", "Ethnicity", "Ethnicity", "S", "Hispanic", "19700101", "20991231"),
    (38003564, "Not Hispanic or Latino", "Ethnicity", "Ethnicity", "Ethnicity", "S", "Not Hispanic", "19700101", "20991231"),
    (8532, "FEMALE", "Gender", "Gender", "Gender", "S", "F", "19700101", "20991231"),
    (8507, "MALE", "Gender", "Gender", "Gender", "S", "M", "19700101", "20991231"),
    (0, "No matching concept", "Metadata", "None", "Undefined", "", "No matching concept", "19700101", "20991231"),
    (8515, "Asian", "Race", "Race", "Race", "S", "2", "19700101", "20991231"),
    (8516, "Black or African American", "Race", "Race", "Race", "S", "3", "19700101", "20991231"),
    (8527, "White", "Race", "Race", "Race", "S", "5", "19700101", "20991231"),
    (1308216, "lisinopril", "Drug", "RxNorm", "Ingredient", "S", "29046", "19700101", "20991231"),
    (1503297, "metformin", "Drug", "RxNorm", "Ingredient", "S", "6809", "19700101", "20991231"),
    (3004249, "Systolic blood pressure", "Measurement", "LOINC", "Clinical Observation", "S", "8480-6", "19960906", "20991231"),
    (3028437, "Cholesterol in LDL [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "2089-1", "19700101", "20991231"),
    (3027114, "Cholesterol [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "2093-3", "19700101", "20991231"),
    (3004501, "Glucose [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "2345-7", "19700101", "20991231"),
    (32817, "EHR", "Type Concept", "Type Concept", "Type Concept", "S", "OMOP4976890", "20200820", "20991231"),
    (8636, "gram per liter", "Unit", "UCUM", "Unit", "S", "g/L", "19700101", "20991231"),
    (8923, "international unit per liter", "Unit", "UCUM", "Unit", "S", "[iU]/L", "19700101", "20991231"),
    (8985, "international unit per milliliter", "Unit", "UCUM", "Unit", "S", "[iU]/mL", "19700101", "20991231"),
    (8840, "milligram per deciliter", "Unit", "UCUM", "Unit", "S", "mg/dL", "19700101", "20991231"),
    (720870, "milliliter per minute per 1.73 square meter", "Unit", "UCUM", "Unit", "S", "mL/min/(173.10*-2.m2)", "20220407", "20991231"),
    (8876, "millimeter mercury column", "Unit", "UCUM", "Unit", "S", "mm[Hg]", "19700101", "20991231"),
    (8482, "pH", "Unit", "UCUM", "Unit", "S", "pH", "19700101", "20991231"),
    (8523, "ratio", "Unit", "UCUM", "Unit", "S", "{ratio}", "19700101", "20991231"),
    (8906, "microgram per 24 hours", "Unit", "UCUM", "Unit", "S", "ug/(24.h)", "19700101", "20991231"),
    (8645, "unit per liter", "Unit", "UCUM", "Unit", "S", "[U]/L", "19700101", "20991231"),
    (8763, "unit per milliliter", "Unit", "UCUM", "Unit", "S", "[U]/mL", "19700101", "20991231"),
    (9203, "Emergency Room Visit", "Visit", "Visit", "Visit", "S", "ER", "19700101", "20991231"),
    (9201, "Inpatient Visit", "Visit", "Visit", "Visit", "S", "IP", "19700101", "20991231"),
    (9202, "Outpatient Visit", "Visit", "Visit", "Visit", "S", "OP", "19700101", "20991231"),
    (3012888, "Diastolic blood pressure", "Measurement", "LOINC", "Clinical Observation", "S", "8462-4", "19960906", "20991231"),
    (4132161, "Oral", "Route", "SNOMED", "Qualifier Value", "S", "26643006", "20020131", "20991231"),
    (201826, "Type 2 diabetes mellitus", "Condition", "SNOMED", "Disorder", "S", "44054006", "20020131", "20991231"),
    (432867, "Hyperlipidemia", "Condition", "SNOMED", "Disorder", "S", "55822004", "20020131", "20991231"),
    (320128, "Essential hypertension", "Condition", "SNOMED", "Disorder", "S", "59621000", "20020131", "20991231"),
    (2786741, "Measurement of Cardiac Electrical Activity, External Approach", "Procedure", "ICD10PCS", "ICD10PCS", "S", "4A02X4Z", "20140422", "20991231"),
    (35206882, "Type 2 diabetes mellitus without complications", "Condition", "ICD10CM", "4-char billing code", "", "E11.9", "20070101", "20991231"),
    (35207668, "Essential (primary) hypertension", "Condition", "ICD10CM", "3-char billing code", "", "I10", "20070101", "20991231"),
    (35207065, "Hyperlipidemia, unspecified", "Condition", "ICD10CM", "4-char billing code", "", "E78.5", "20070101", "20991231"),
]  # fmt: skip

# Standard concepts of these vocabularies are the TriNetX codes themselves and map to
# themselves.
_SELF_MAPPED_VOCABULARIES = {"RxNorm", "LOINC", "ICD10PCS"}

# ICD10CM source concepts and the standard concepts they map to
_MAPS_TO = [(35207668, 320128), (35206882, 201826), (35207065, 432867)]

_VOCABULARIES = [
    "CDM",
    "Ethnicity",
    "Gender",
    "ICD10CM",
    "ICD10PCS",
    "LOINC",
    "Race",
    "RxNorm",
    "SNOMED",
    "Type Concept",
    "UCUM",
    "Visit",
]
VOCABULARY_VERSION = "Example excerpt of the Athena vocabulary"


def _csv_text(header: list[str], rows: list[list[str]], delimiter: str = ",") -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _dictionary_xlsx() -> bytes:
    header = ["Data Category Name", TABLE_HEADER, FIELD_HEADER, TYPE_HEADER]
    cells: list[list[str]] = [header]
    for position, (table, (columns, _)) in enumerate(TNX_TABLES.items()):
        if position:
            cells += [[], header]
        cells += [["Example", table, name, kind] for name, kind in columns]
    buffer = io.BytesIO()
    write_sheet(buffer, DATA_DICTIONARY_SHEET, cells)
    return buffer.getvalue()


def write_export(path: Path) -> None:
    """Write the synthetic TriNetX ZIP with its data dictionary."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for table, (columns, rows) in TNX_TABLES.items():
            header = [name for name, _ in columns]
            zf.writestr(f"{table}.csv", _csv_text(header, rows))
        zf.writestr(DATA_DICTIONARY_FILE, _dictionary_xlsx())


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text(_csv_text(header, rows, "\t"), encoding="utf-8")


def write_athena(directory: Path) -> None:
    """Write the tab-separated Athena style files the converter reads by default."""
    directory.mkdir(parents=True, exist_ok=True)

    concepts = [[str(c[0]), *c[1:7], c[7], c[8], ""] for c in CONCEPTS]
    _write_tsv(directory / f"{tbl.athena_concept}.csv", CONCEPT_COLUMNS, concepts)

    maps_to = _MAPS_TO + [
        (c[0], c[0]) for c in CONCEPTS if c[3] in _SELF_MAPPED_VOCABULARIES
    ]
    _write_tsv(
        directory / f"{tbl.athena_concept_relationship}.csv",
        [
            col.concept_id_1,
            col.concept_id_2,
            col.relationship_id,
            col.valid_start_date,
            col.valid_end_date,
            col.invalid_reason,
        ],
        [[str(a), str(b), "Maps to", "19700101", "20991231", ""] for a, b in maps_to],
    )

    vocabulary = [
        ["None", "OMOP Standardized Vocabularies", "OMOP generated", VOCABULARY_VERSION, "44819096"]
    ]  # fmt: skip
    vocabulary += [[v, v, "", "", "0"] for v in _VOCABULARIES]
    _write_tsv(
        directory / f"{tbl.athena_vocabulary}.csv",
        [
            col.vocabulary_id,
            "vocabulary_name",
            "vocabulary_reference",
            col.vocabulary_version,
            "vocabulary_concept_id",
        ],
        vocabulary,
    )

    # Needed by the era tables: every ingredient is its own ancestor.
    _write_tsv(
        directory / f"{tbl.athena_concept_ancestor}.csv",
        [
            col.ancestor_concept_id,
            col.descendant_concept_id,
            col.min_levels_of_separation,
            col.max_levels_of_separation,
        ],
        [[str(c[0]), str(c[0]), "0", "0"] for c in CONCEPTS if c[4] == "Ingredient"],
    )


def write_example(output_dir: Path) -> tuple[Path, Path]:
    """Write the example export and vocabulary into output_dir.

    Returns:
        The ZIP path and the Athena folder. Existing files of the same names are
        replaced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / ZIP_NAME
    athena_dir = output_dir / ATHENA_DIR
    write_export(zip_path)
    write_athena(athena_dir)
    return zip_path, athena_dir
