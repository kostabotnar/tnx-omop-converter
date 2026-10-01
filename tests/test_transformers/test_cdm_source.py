"""Tests for transformers/cdm_source.py (plain values, no files)."""

from datetime import date

from tests.schema_asserts import assert_conforms
from tnx_omop.schema.omop import CDM_SOURCE_SCHEMA
from tnx_omop.transformers.cdm_source import (
    CDM_VERSION_CONCEPT_ID,
    transform_cdm_source,
)
from tnx_omop.util import columns as col

VOCAB_VERSION = "v5.0 30-AUG-26"
RELEASE = date(2026, 9, 28)


def test_combines_values_into_one_row():
    """Shared network name, latest export date, and all cohorts sorted."""
    result = transform_cdm_source(
        ["Hospital A", "Hospital A"],
        ["Controls", "Cases"],
        ["20260219", "20260220"],
        RELEASE,
        VOCAB_VERSION,
    )

    row = result.row(0, named=True)
    assert len(result) == 1
    assert row[col.cdm_source_name] == "Hospital A"
    assert row[col.cdm_holder] == "Hospital A"
    assert row[col.source_description] == (
        "TriNetX EHR export. Cohorts: Cases; Controls"
    )
    assert row[col.source_release_date] == date(2026, 2, 20)
    assert row[col.cdm_release_date] == RELEASE
    assert row[col.cdm_version] == "v5.4"
    assert row[col.cdm_version_concept_id] == CDM_VERSION_CONCEPT_ID
    assert row[col.vocabulary_version] == VOCAB_VERSION
    assert_conforms(result, CDM_SOURCE_SCHEMA)


def test_several_networks_are_joined():
    result = transform_cdm_source(["B", "A"], [], [], RELEASE, VOCAB_VERSION)

    assert result[col.cdm_source_name].to_list() == ["A; B"]


def test_no_values_use_defaults():
    result = transform_cdm_source([], [], [], RELEASE, VOCAB_VERSION)

    row = result.row(0, named=True)
    assert row[col.cdm_source_name] == "TriNetX export"
    assert row[col.source_description] == "TriNetX EHR export"
    assert row[col.source_release_date] == RELEASE
    assert_conforms(result, CDM_SOURCE_SCHEMA)


def test_unparseable_export_dates_are_ignored():
    result = transform_cdm_source([], [], ["x", "20260101"], RELEASE, VOCAB_VERSION)

    assert result[col.source_release_date].to_list() == [date(2026, 1, 1)]
