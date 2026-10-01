"""Tests for tnx_omop/transformers/person.py."""

import polars as pl

from tnx_omop.util import columns as col
from tests.id_helpers import with_ids
from tnx_omop.util.concept_mappings import (
    GENDER_CONCEPT_MAP,
    RACE_CONCEPT_MAP,
    ETHNICITY_CONCEPT_MAP,
    DEFAULT_CONCEPT_ID,
)
from tnx_omop.transformers.person import transform_person


class TestTransformPerson:
    """Tests for transform_person function."""

    def test_transforms_basic_patient(self, sample_patient_df):
        """Transforms basic patient data to PERSON format."""
        result = transform_person(sample_patient_df)

        # Derived records are kept
        assert len(result) == 3

        # Check schema has all required OMOP columns
        assert col.person_id in result.columns
        assert col.gender_concept_id in result.columns
        assert col.year_of_birth in result.columns
        assert col.person_source_value in result.columns

    def test_keeps_derived_records(self, sample_patient_df):
        """Keeps derived rows and drops the derived_by_TriNetX column."""
        result = transform_person(sample_patient_df)

        # P002 has derived_by_TriNetX = "T" and must still be present
        source_values = result[col.person_source_value].to_list()
        assert source_values == ["P001", "P002", "P003"]
        assert col.derived_by_TriNetX not in result.columns

    def test_keeps_assigned_person_ids(self, sample_patient_df):
        """Uses the person_id assigned before the transform."""
        result = transform_person(sample_patient_df)

        assert result[col.person_id].to_list() == [1, 2, 3]

    def test_preserves_year_of_birth(self, sample_patient_df):
        """Preserves year_of_birth values."""
        result = transform_person(sample_patient_df)

        # P001 (1980) and P003 (2000) should be present
        years = result[col.year_of_birth].to_list()
        assert 1980 in years
        assert 2000 in years

    def test_handles_missing_optional_columns(self):
        """Handles DataFrames missing optional columns."""
        minimal_df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default"],
                    col.patient_id: ["P001"],
                    col.year_of_birth: [1990],
                    col.derived_by_TriNetX: [None],
                }
            )
        )
        result = transform_person(minimal_df)
        assert len(result) == 1

    def test_all_derived_records_kept(self):
        """Keeps rows when every record is derived."""
        all_derived_df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default", "default"],
                    col.patient_id: ["P001", "P002"],
                    col.year_of_birth: [1990, 1991],
                    col.derived_by_TriNetX: ["T", "T"],
                }
            )
        )
        result = transform_person(all_derived_df)
        assert len(result) == 2

    def test_output_schema(self, sample_patient_df):
        """Output has correct OMOP PERSON schema."""
        result = transform_person(sample_patient_df)

        expected_columns = {
            col.person_id,
            col.gender_concept_id,
            col.year_of_birth,
            col.month_of_birth,
            col.day_of_birth,
            col.birth_datetime,
            col.race_concept_id,
            col.ethnicity_concept_id,
            col.location_id,
            col.provider_id,
            col.care_site_id,
            col.person_source_value,
            col.gender_source_value,
            col.gender_source_concept_id,
            col.race_source_value,
            col.race_source_concept_id,
            col.ethnicity_source_value,
            col.ethnicity_source_concept_id,
        }
        assert set(result.columns) == expected_columns

    def test_maps_gender_concept_ids(self):
        """Maps sex values to correct gender_concept_id."""
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default", "default", "default"],
                    col.patient_id: ["P001", "P002", "P003"],
                    col.year_of_birth: [1980, 1990, 2000],
                    col.sex: ["M", "F", "Unknown"],
                    col.derived_by_TriNetX: [None, None, None],
                }
            )
        )
        result = transform_person(df)

        gender_ids = result.sort(col.person_source_value)[
            col.gender_concept_id
        ].to_list()
        assert gender_ids[0] == GENDER_CONCEPT_MAP["M"]  # 8507
        assert gender_ids[1] == GENDER_CONCEPT_MAP["F"]  # 8532
        assert gender_ids[2] == DEFAULT_CONCEPT_ID  # 0 for Unknown

    def test_maps_race_concept_ids(self):
        """Maps race values to correct race_concept_id."""
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default"] * 6,
                    col.patient_id: ["P001", "P002", "P003", "P004", "P005", "P006"],
                    col.year_of_birth: [1980] * 6,
                    col.race: [
                        "White",
                        "Black or African American",
                        "Asian",
                        "American Indian or Alaska Native",
                        "Native Hawaiian or Other Pacific Islander",
                        "Unknown",
                    ],
                    col.derived_by_TriNetX: [None] * 6,
                }
            )
        )
        result = transform_person(df)

        race_ids = result.sort(col.person_source_value)[col.race_concept_id].to_list()
        assert race_ids[0] == RACE_CONCEPT_MAP["White"]  # 8527
        assert race_ids[1] == RACE_CONCEPT_MAP["Black or African American"]  # 8516
        assert race_ids[2] == RACE_CONCEPT_MAP["Asian"]  # 8515
        assert (
            race_ids[3] == RACE_CONCEPT_MAP["American Indian or Alaska Native"]
        )  # 8657
        assert (
            race_ids[4] == RACE_CONCEPT_MAP["Native Hawaiian or Other Pacific Islander"]
        )  # 8557
        assert race_ids[5] == DEFAULT_CONCEPT_ID  # 0 for Unknown

    def test_maps_ethnicity_concept_ids(self):
        """Maps ethnicity values to correct ethnicity_concept_id."""
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default", "default", "default"],
                    col.patient_id: ["P001", "P002", "P003"],
                    col.year_of_birth: [1980, 1990, 2000],
                    col.ethnicity: [
                        "Hispanic or Latino",
                        "Not Hispanic or Latino",
                        "Unknown",
                    ],
                    col.derived_by_TriNetX: [None, None, None],
                }
            )
        )
        result = transform_person(df)

        ethnicity_ids = result.sort(col.person_source_value)[
            col.ethnicity_concept_id
        ].to_list()
        assert (
            ethnicity_ids[0] == ETHNICITY_CONCEPT_MAP["Hispanic or Latino"]
        )  # 38003563
        assert (
            ethnicity_ids[1] == ETHNICITY_CONCEPT_MAP["Not Hispanic or Latino"]
        )  # 38003564
        assert ethnicity_ids[2] == DEFAULT_CONCEPT_ID  # 0 for Unknown

    def test_unmapped_values_default_to_zero(self):
        """Unmapped or unexpected values default to 0."""
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default"],
                    col.patient_id: ["P001"],
                    col.year_of_birth: [1980],
                    col.sex: ["X"],  # Invalid value
                    col.race: ["Other"],  # Unmapped
                    col.ethnicity: ["Declined"],  # Unmapped
                    col.derived_by_TriNetX: [None],
                }
            )
        )
        result = transform_person(df)

        assert result[col.gender_concept_id][0] == DEFAULT_CONCEPT_ID
        assert result[col.race_concept_id][0] == DEFAULT_CONCEPT_ID
        assert result[col.ethnicity_concept_id][0] == DEFAULT_CONCEPT_ID
