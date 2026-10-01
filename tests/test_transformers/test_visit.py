"""Tests for tnx_omop/transformers/visit.py."""

from datetime import date

import polars as pl

from tnx_omop.util import columns as col
from tests.id_helpers import with_ids
from tnx_omop.util.concept_mappings import VISIT_TYPE_CONCEPT_MAP, DEFAULT_CONCEPT_ID
from tnx_omop.transformers.visit import transform_visit


class TestTransformVisit:
    """Tests for transform_visit function."""

    def test_transforms_basic_encounter(self, sample_encounter_df):
        """Transforms basic encounter data to VISIT_OCCURRENCE format."""
        result = transform_visit(sample_encounter_df)

        # Derived records are kept
        assert len(result) == 3

        # Check schema has required OMOP columns
        assert col.visit_occurrence_id in result.columns
        assert col.person_id in result.columns
        assert col.visit_concept_id in result.columns
        assert col.visit_start_date in result.columns

    def test_keeps_derived_records(self, sample_encounter_df):
        """Keeps derived rows and drops every *derived_by_TriNetX column."""
        result = transform_visit(sample_encounter_df)

        # E003 has derived flags set to "T" and must still be present
        assert result[col.visit_source_value].to_list() == ["E001", "E002", "E003"]
        assert not [c for c in result.columns if "derived" in c.lower()]

    def test_maps_visit_concept_ids(self):
        """Maps encounter type to correct visit_concept_id."""
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default"] * 6,
                    col.patient_id: ["P001"] * 6,
                    col.encounter_id: ["E001", "E002", "E003", "E004", "E005", "E006"],
                    col.start_date: [date(2023, 1, 1)] * 6,
                    col.end_date: [date(2023, 1, 2)] * 6,
                    col.type: ["IMP", "EMER", "AMB", "HH", "VR", "FLD"],
                    col.derived_by_TriNetX: [None] * 6,
                }
            )
        )
        result = transform_visit(df)

        # Encounters E001 to E006 carry the types in order
        visit_map = dict(
            zip(
                ["IMP", "EMER", "AMB", "HH", "VR", "FLD"],
                result[col.visit_concept_id].to_list(),
            )
        )
        assert visit_map["IMP"] == VISIT_TYPE_CONCEPT_MAP["IMP"]  # 9201 - Inpatient
        assert visit_map["EMER"] == VISIT_TYPE_CONCEPT_MAP["EMER"]  # 9203 - Emergency
        assert visit_map["AMB"] == VISIT_TYPE_CONCEPT_MAP["AMB"]  # 9202 - Ambulatory
        assert visit_map["HH"] == VISIT_TYPE_CONCEPT_MAP["HH"]  # 581476 - Home Health
        assert visit_map["VR"] == VISIT_TYPE_CONCEPT_MAP["VR"]  # 722455 - Telehealth
        assert visit_map["FLD"] == VISIT_TYPE_CONCEPT_MAP["FLD"]  # 581478 - Field

    def test_maps_inpatient_variants(self):
        """Maps inpatient variants (ACUTE, NONAC) to same concept."""
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default"] * 3,
                    col.patient_id: ["P001"] * 3,
                    col.encounter_id: ["E001", "E002", "E003"],
                    col.start_date: [date(2023, 1, 1)] * 3,
                    col.end_date: [date(2023, 1, 2)] * 3,
                    col.type: ["IMP", "ACUTE", "NONAC"],
                    col.derived_by_TriNetX: [None] * 3,
                }
            )
        )
        result = transform_visit(df)

        visit_ids = result[col.visit_concept_id].to_list()
        # All should map to 9201 (Inpatient)
        assert all(vid == 9201 for vid in visit_ids)

    def test_maps_ambulatory_variants(self):
        """Maps ambulatory variants (SS, OBSENC, PRENC) to same concept."""
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default"] * 4,
                    col.patient_id: ["P001"] * 4,
                    col.encounter_id: ["E001", "E002", "E003", "E004"],
                    col.start_date: [date(2023, 1, 1)] * 4,
                    col.end_date: [date(2023, 1, 2)] * 4,
                    col.type: ["AMB", "SS", "OBSENC", "PRENC"],
                    col.derived_by_TriNetX: [None] * 4,
                }
            )
        )
        result = transform_visit(df)

        visit_ids = result[col.visit_concept_id].to_list()
        # All should map to 9202 (Ambulatory/Outpatient)
        assert all(vid == 9202 for vid in visit_ids)

    def test_unmapped_type_defaults_to_zero(self):
        """Unknown encounter types default to concept_id 0."""
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default", "default"],
                    col.patient_id: ["P001", "P001"],
                    col.encounter_id: ["E001", "E002"],
                    col.start_date: [date(2023, 1, 1), date(2023, 1, 1)],
                    col.end_date: [date(2023, 1, 2), date(2023, 1, 2)],
                    col.type: ["Unknown", "OTHER"],
                    col.derived_by_TriNetX: [None, None],
                }
            )
        )
        result = transform_visit(df)

        visit_ids = result[col.visit_concept_id].to_list()
        assert all(vid == DEFAULT_CONCEPT_ID for vid in visit_ids)

    def test_handles_missing_type_column(self):
        """Handles DataFrames without type column."""
        df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default"],
                    col.patient_id: ["P001"],
                    col.encounter_id: ["E001"],
                    col.start_date: [date(2023, 1, 1)],
                    col.end_date: [date(2023, 1, 2)],
                    col.derived_by_TriNetX: [None],
                }
            )
        )
        result = transform_visit(df)

        assert len(result) == 1
        assert result[col.visit_concept_id][0] == DEFAULT_CONCEPT_ID

    def test_keeps_source_ids_and_encounter_id_as_source_value(
        self, sample_encounter_df
    ):
        """Uses the IDs assigned before the transform and stores encounter_id."""
        result = transform_visit(sample_encounter_df)

        assert result[col.visit_occurrence_id].to_list() == [1, 2, 3]
        assert result[col.person_id].to_list() == [1, 1, 2]
        assert result[col.visit_source_value].to_list() == ["E001", "E002", "E003"]

    def test_all_derived_records_kept(self):
        """Keeps rows when every record is derived."""
        all_derived_df = with_ids(
            pl.DataFrame(
                {
                    col._source_id: ["default", "default"],
                    col.patient_id: ["P001", "P002"],
                    col.encounter_id: ["E001", "E002"],
                    col.start_date: [date(2023, 1, 1), date(2023, 1, 2)],
                    col.end_date: [date(2023, 1, 1), date(2023, 1, 2)],
                    col.type: ["IMP", "AMB"],
                    col.derived_by_TriNetX: ["T", "T"],
                }
            )
        )
        result = transform_visit(all_derived_df)
        assert len(result) == 2
