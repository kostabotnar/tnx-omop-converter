"""Tests for tnx_omop/cdm_source.py."""

from datetime import date

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.cdm_source import build_cdm_source
from tnx_omop.transformers.cdm_source import CDM_VERSION_CONCEPT_ID

VOCAB_VERSION = "v5.0 30-AUG-26"


def _write_source(path, network, created, cohort):
    path.mkdir()
    (path / f"{tbl.tnx_dataset_details}.csv").write_text(
        '"total_number_unique_patients","total_number_HCOs","date_created",'
        '"network_name"\n'
        f'"10","1","{created}","{network}"\n'
    )
    (path / f"{tbl.tnx_cohort_details}.csv").write_text(
        f'"cohort_name","cohort_number","total_patient_records"\n"{cohort}","1","10"\n'
    )


def test_combines_metadata_of_all_sources(tmp_path):
    """One row with shared network name, latest export date, and all cohorts."""
    _write_source(tmp_path / "a", "Hospital A", "20260219", "Controls")
    _write_source(tmp_path / "b", "Hospital A", "20260220", "Cases")

    result = build_cdm_source(
        [tmp_path / "a", tmp_path / "b"], date(2026, 9, 28), VOCAB_VERSION
    )

    row = result.row(0, named=True)
    assert len(result) == 1
    assert row[col.cdm_source_name] == "Hospital A"
    assert row[col.cdm_holder] == "Hospital A"
    assert row[col.source_description] == (
        "TriNetX EHR export. Cohorts: Cases; Controls"
    )
    assert row[col.source_release_date] == date(2026, 2, 20)
    assert row[col.cdm_release_date] == date(2026, 9, 28)
    assert row[col.cdm_version] == "v5.4"
    assert row[col.cdm_version_concept_id] == CDM_VERSION_CONCEPT_ID
    assert row[col.vocabulary_version] == VOCAB_VERSION


def test_missing_metadata_uses_defaults(tmp_path):
    """Without metadata files the required fields still get values."""
    result = build_cdm_source([tmp_path], date(2026, 9, 28), VOCAB_VERSION)

    row = result.row(0, named=True)
    assert row[col.cdm_source_name] == "TriNetX export"
    assert row[col.source_release_date] == date(2026, 9, 28)
