"""Shared pytest fixtures for transformer tests."""

import logging
from datetime import date
from pathlib import Path
from typing import Callable, Dict

import polars as pl
import pytest

from tnx_omop.util import columns as col
from tnx_omop.omop_vocab.vocabulary import (
    VocabularyLookup,
    build_lookup,
    collect_source_codes,
    collect_units,
)
from tnx_omop import __main__ as entry
from tnx_omop.util.concept_mappings import reset_mappings
from tests.athena_fixture import write_athena_fixture


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command-line option for output directory."""
    parser.addoption(
        "--output-dir",
        action="store",
        default=None,
        help="Path to OMOP output directory containing Parquet files",
    )


@pytest.fixture(autouse=True)
def restore_logging():
    """Undo the logging setup of `entry.main`, which writes to a captured stderr."""
    root = logging.getLogger()
    level = root.level
    yield
    if entry._handler is not None:
        root.removeHandler(entry._handler)
        entry._handler = None
    root.setLevel(level)


@pytest.fixture(autouse=True)
def restore_mappings():
    """Undo config overrides of the concept maps, which are module level state."""
    yield
    reset_mappings()


@pytest.fixture(scope="session")
def athena_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic Athena vocabulary folder (see tests/athena_fixture.py)."""
    return write_athena_fixture(tmp_path_factory.mktemp("athena"))


@pytest.fixture(scope="session")
def make_lookup(
    athena_dir: Path,
) -> Callable[[Dict[str, pl.DataFrame]], VocabularyLookup]:
    """Build a VocabularyLookup for TriNetX tables against the Athena fixture."""

    def make(tables: Dict[str, pl.DataFrame]) -> VocabularyLookup:
        return build_lookup(
            athena_dir, collect_source_codes(tables), collect_units(tables)
        )

    return make


@pytest.fixture
def sample_patient_df() -> pl.DataFrame:
    """Sample patient DataFrame for testing."""
    return pl.DataFrame(
        {
            col._source_id: ["default", "default", "default"],
            col.patient_id: ["P001", "P002", "P003"],
            col.person_id: [1, 2, 3],
            col.year_of_birth: [1980, 1990, 2000],
            col.sex: ["M", "F", "M"],
            col.race: ["White", "Black or African American", "Asian"],
            col.ethnicity: [
                "Not Hispanic or Latino",
                "Hispanic or Latino",
                "Not Hispanic or Latino",
            ],
            col.derived_by_TriNetX: [None, "T", None],
        }
    )


@pytest.fixture
def sample_encounter_df() -> pl.DataFrame:
    """Sample encounter DataFrame for testing."""
    return pl.DataFrame(
        {
            col._source_id: ["default", "default", "default"],
            col.patient_id: ["P001", "P001", "P002"],
            col.person_id: [1, 1, 2],
            col.encounter_id: ["E001", "E002", "E003"],
            col.visit_occurrence_id: [1, 2, 3],
            col.start_date: [date(2023, 1, 1), date(2023, 1, 15), date(2023, 2, 1)],
            col.end_date: [date(2023, 1, 2), date(2023, 1, 15), date(2023, 2, 5)],
            col.type: ["IMP", "AMB", "EMER"],  # HL7 v3 ActEncounterCode values
            "start_date_derived_by_TriNetX": ["F", "T", "T"],
            "end_date_derived_by_TriNetX": ["F", "T", "T"],
            col.derived_by_TriNetX: [None, None, "T"],
        }
    )


@pytest.fixture
def sample_diagnosis_df() -> pl.DataFrame:
    """Sample diagnosis DataFrame for testing."""
    return pl.DataFrame(
        {
            col._source_id: ["default", "default", "default"],
            col.patient_id: ["P001", "P001", "P002"],
            col.person_id: [1, 1, 2],
            col.encounter_id: ["E001", "E001", "E002"],
            col.visit_occurrence_id: [1, 1, 2],
            col.date: [date(2023, 1, 1), date(2023, 1, 1), date(2023, 2, 1)],
            col.code: ["I10", "E78.5", "K21.0"],
            col.code_system: ["ICD-10-CM", "ICD-10-CM", "ICD-10-CM"],
            col.principal_diagnosis_indicator: [
                "P",
                "S",
                "P",
            ],  # P=Primary, S=Secondary
            col.derived_by_TriNetX: [None, None, None],
        }
    )


@pytest.fixture
def sample_lab_result_df() -> pl.DataFrame:
    """Sample lab_result DataFrame for testing."""
    return pl.DataFrame(
        {
            col._source_id: ["default", "default"],
            col.patient_id: ["P001", "P001"],
            col.person_id: [1, 1],
            col.encounter_id: ["E001", "E001"],
            col.visit_occurrence_id: [1, 1],
            col.date: [date(2023, 1, 1), date(2023, 1, 1)],
            col.code_system: ["LOINC", "LOINC"],
            col.code: ["2160-0", "2345-7"],
            col.lab_result_num_val: [1.2, 100.0],
            col.lab_result_text_val: [None, "Normal"],
            col.units_of_measure: ["mg/dL", "mg/dL"],
            col.derived_by_TriNetX: [None, None],
        }
    )


@pytest.fixture
def sample_vitals_df() -> pl.DataFrame:
    """Sample vitals_signs DataFrame for testing."""
    return pl.DataFrame(
        {
            col._source_id: ["default", "default"],
            col.patient_id: ["P001", "P001"],
            col.person_id: [1, 1],
            col.encounter_id: ["E001", "E001"],
            col.visit_occurrence_id: [1, 1],
            col.date: [date(2023, 1, 1), date(2023, 1, 1)],
            col.code_system: ["LOINC", "LOINC"],
            col.code: ["8480-6", "8462-4"],
            col.value: [120.0, 80.0],
            col.text_value: [None, None],
            col.units_of_measure: ["mmHg", "mmHg"],
            col.derived_by_TriNetX: [None, None],
        }
    )


@pytest.fixture
def sample_medication_df() -> pl.DataFrame:
    """Sample medication_ingredient DataFrame for testing."""
    return pl.DataFrame(
        {
            col._source_id: ["default", "default"],
            col.patient_id: ["P001", "P001"],
            col.person_id: [1, 1],
            col.encounter_id: ["E001", "E001"],
            col.visit_occurrence_id: [1, 1],
            col.unique_id: ["M001", "M002"],
            col.start_date: [date(2023, 1, 1), date(2023, 1, 1)],
            col.end_date: [date(2023, 1, 7), date(2023, 1, 7)],
            col.code_system: ["RxNorm", "RxNorm"],
            col.code: ["197361", "312961"],
            col.route: ["Oral Product", "Injectable Product"],  # TriNetX route values
            col.derived_by_TriNetX: [None, None],
        }
    )


@pytest.fixture
def sample_procedure_df() -> pl.DataFrame:
    """Sample procedure DataFrame for testing."""
    return pl.DataFrame(
        {
            col._source_id: ["default", "default"],
            col.patient_id: ["P001", "P002"],
            col.person_id: [1, 2],
            col.encounter_id: ["E001", "E002"],
            col.visit_occurrence_id: [1, 2],
            col.date: [date(2023, 1, 1), date(2023, 2, 1)],
            col.code: ["99213", "99214"],
            col.code_system: ["CPT", "CPT"],
            col.modifier_1: ["25", None],
            col.derived_by_TriNetX: [None, None],
        }
    )
