"""OMOP CDM 5.4 schema definitions for validation."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import polars as pl

from ..util import columns as col
from ..util import tables


@dataclass
class ColumnSpec:
    """Specification for an OMOP CDM column."""

    name: str
    dtype: pl.DataType
    required: bool  # True if column cannot be null
    primary_key: bool = False  # True if column is the primary key


@dataclass
class TableSchema:
    """Schema specification for an OMOP CDM table."""

    name: str
    columns: List[ColumnSpec]

    @property
    def column_names(self) -> List[str]:
        """Return list of column names."""
        return [c.name for c in self.columns]

    @property
    def primary_key(self) -> Optional[str]:
        """Return the primary key column name, or None if the table has none."""
        for c in self.columns:
            if c.primary_key:
                return c.name
        return None

    @property
    def required_columns(self) -> List[str]:
        """Return list of required (non-null) column names."""
        return [c.name for c in self.columns if c.required]

    @property
    def dtype_map(self) -> Dict[str, pl.DataType]:
        """Return mapping of column names to data types."""
        return {c.name: c.dtype for c in self.columns}


# PERSON table schema
PERSON_SCHEMA = TableSchema(
    name=tables.omop_person,
    columns=[
        ColumnSpec(col.person_id, pl.Int64, required=True, primary_key=True),
        ColumnSpec(col.gender_concept_id, pl.Int64, required=True),
        ColumnSpec(col.year_of_birth, pl.Int64, required=True),
        ColumnSpec(col.month_of_birth, pl.Int64, required=False),
        ColumnSpec(col.day_of_birth, pl.Int64, required=False),
        ColumnSpec(col.birth_datetime, pl.Datetime, required=False),
        ColumnSpec(col.race_concept_id, pl.Int64, required=True),
        ColumnSpec(col.ethnicity_concept_id, pl.Int64, required=True),
        ColumnSpec(col.location_id, pl.Int64, required=False),
        ColumnSpec(col.provider_id, pl.Int64, required=False),
        ColumnSpec(col.care_site_id, pl.Int64, required=False),
        ColumnSpec(col.person_source_value, pl.Utf8, required=False),
        ColumnSpec(col.gender_source_value, pl.Utf8, required=False),
        ColumnSpec(col.gender_source_concept_id, pl.Int64, required=True),
        ColumnSpec(col.race_source_value, pl.Utf8, required=False),
        ColumnSpec(col.race_source_concept_id, pl.Int64, required=True),
        ColumnSpec(col.ethnicity_source_value, pl.Utf8, required=False),
        ColumnSpec(col.ethnicity_source_concept_id, pl.Int64, required=True),
    ],
)

# VISIT_OCCURRENCE table schema
VISIT_OCCURRENCE_SCHEMA = TableSchema(
    name=tables.omop_visit_occurrence,
    columns=[
        ColumnSpec(col.visit_occurrence_id, pl.Int64, required=True, primary_key=True),
        ColumnSpec(col.person_id, pl.Int64, required=True),
        ColumnSpec(col.visit_concept_id, pl.Int64, required=True),
        ColumnSpec(col.visit_start_date, pl.Date, required=True),
        ColumnSpec(col.visit_start_datetime, pl.Datetime, required=False),
        ColumnSpec(col.visit_end_date, pl.Date, required=False),
        ColumnSpec(col.visit_end_datetime, pl.Datetime, required=False),
        ColumnSpec(col.visit_type_concept_id, pl.Int64, required=True),
        ColumnSpec(col.provider_id, pl.Int64, required=False),
        ColumnSpec(col.care_site_id, pl.Int64, required=False),
        ColumnSpec(col.visit_source_value, pl.Utf8, required=False),
        ColumnSpec(col.visit_source_concept_id, pl.Int64, required=True),
        ColumnSpec(col.admitted_from_concept_id, pl.Int64, required=True),
        ColumnSpec(col.admitted_from_source_value, pl.Utf8, required=False),
        ColumnSpec(col.discharged_to_concept_id, pl.Int64, required=True),
        ColumnSpec(col.discharged_to_source_value, pl.Utf8, required=False),
        ColumnSpec(col.preceding_visit_occurrence_id, pl.Int64, required=False),
    ],
)

# CONDITION_OCCURRENCE table schema
CONDITION_OCCURRENCE_SCHEMA = TableSchema(
    name=tables.omop_condition_occurrence,
    columns=[
        ColumnSpec(
            col.condition_occurrence_id, pl.Int64, required=True, primary_key=True
        ),
        ColumnSpec(col.person_id, pl.Int64, required=True),
        ColumnSpec(col.condition_concept_id, pl.Int64, required=True),
        ColumnSpec(col.condition_start_date, pl.Date, required=True),
        ColumnSpec(col.condition_start_datetime, pl.Datetime, required=False),
        ColumnSpec(col.condition_end_date, pl.Date, required=False),
        ColumnSpec(col.condition_end_datetime, pl.Datetime, required=False),
        ColumnSpec(col.condition_type_concept_id, pl.Int64, required=True),
        ColumnSpec(col.condition_status_concept_id, pl.Int64, required=True),
        ColumnSpec(col.stop_reason, pl.Utf8, required=False),
        ColumnSpec(col.provider_id, pl.Int64, required=False),
        ColumnSpec(col.visit_occurrence_id, pl.Int64, required=False),
        ColumnSpec(col.visit_detail_id, pl.Int64, required=False),
        ColumnSpec(col.condition_source_value, pl.Utf8, required=False),
        ColumnSpec(col.condition_source_concept_id, pl.Int64, required=True),
        ColumnSpec(col.condition_status_source_value, pl.Utf8, required=False),
    ],
)

# MEASUREMENT table schema
MEASUREMENT_SCHEMA = TableSchema(
    name=tables.omop_measurement,
    columns=[
        ColumnSpec(col.measurement_id, pl.Int64, required=True, primary_key=True),
        ColumnSpec(col.person_id, pl.Int64, required=True),
        ColumnSpec(col.measurement_concept_id, pl.Int64, required=True),
        ColumnSpec(col.measurement_date, pl.Date, required=True),
        ColumnSpec(col.measurement_datetime, pl.Datetime, required=False),
        ColumnSpec(col.measurement_time, pl.Utf8, required=False),
        ColumnSpec(col.measurement_type_concept_id, pl.Int64, required=True),
        ColumnSpec(col.operator_concept_id, pl.Int64, required=True),
        ColumnSpec(col.value_as_number, pl.Float64, required=False),
        ColumnSpec(col.value_as_concept_id, pl.Int64, required=True),
        ColumnSpec(col.unit_concept_id, pl.Int64, required=True),
        ColumnSpec(col.range_low, pl.Float64, required=False),
        ColumnSpec(col.range_high, pl.Float64, required=False),
        ColumnSpec(col.provider_id, pl.Int64, required=False),
        ColumnSpec(col.visit_occurrence_id, pl.Int64, required=False),
        ColumnSpec(col.visit_detail_id, pl.Int64, required=False),
        ColumnSpec(col.measurement_source_value, pl.Utf8, required=False),
        ColumnSpec(col.measurement_source_concept_id, pl.Int64, required=True),
        ColumnSpec(col.unit_source_value, pl.Utf8, required=False),
        ColumnSpec(col.unit_source_concept_id, pl.Int64, required=True),
        ColumnSpec(col.value_source_value, pl.Utf8, required=False),
        ColumnSpec(col.measurement_event_id, pl.Int64, required=False),
        ColumnSpec(col.meas_event_field_concept_id, pl.Int64, required=True),
    ],
)

# DRUG_EXPOSURE table schema
DRUG_EXPOSURE_SCHEMA = TableSchema(
    name=tables.omop_drug_exposure,
    columns=[
        ColumnSpec(col.drug_exposure_id, pl.Int64, required=True, primary_key=True),
        ColumnSpec(col.person_id, pl.Int64, required=True),
        ColumnSpec(col.drug_concept_id, pl.Int64, required=True),
        ColumnSpec(col.drug_exposure_start_date, pl.Date, required=True),
        ColumnSpec(col.drug_exposure_start_datetime, pl.Datetime, required=False),
        ColumnSpec(col.drug_exposure_end_date, pl.Date, required=True),
        ColumnSpec(col.drug_exposure_end_datetime, pl.Datetime, required=False),
        ColumnSpec(col.verbatim_end_date, pl.Date, required=False),
        ColumnSpec(col.drug_type_concept_id, pl.Int64, required=True),
        ColumnSpec(col.stop_reason, pl.Utf8, required=False),
        ColumnSpec(col.refills, pl.Int64, required=False),
        ColumnSpec(col.quantity, pl.Float64, required=False),
        ColumnSpec(col.days_supply, pl.Int64, required=False),
        ColumnSpec(col.sig, pl.Utf8, required=False),
        ColumnSpec(col.route_concept_id, pl.Int64, required=True),
        ColumnSpec(col.lot_number, pl.Utf8, required=False),
        ColumnSpec(col.provider_id, pl.Int64, required=False),
        ColumnSpec(col.visit_occurrence_id, pl.Int64, required=False),
        ColumnSpec(col.visit_detail_id, pl.Int64, required=False),
        ColumnSpec(col.drug_source_value, pl.Utf8, required=False),
        ColumnSpec(col.drug_source_concept_id, pl.Int64, required=True),
        ColumnSpec(col.route_source_value, pl.Utf8, required=False),
        ColumnSpec(col.dose_unit_source_value, pl.Utf8, required=False),
    ],
)

# PROCEDURE_OCCURRENCE table schema
PROCEDURE_OCCURRENCE_SCHEMA = TableSchema(
    name=tables.omop_procedure_occurrence,
    columns=[
        ColumnSpec(
            col.procedure_occurrence_id, pl.Int64, required=True, primary_key=True
        ),
        ColumnSpec(col.person_id, pl.Int64, required=True),
        ColumnSpec(col.procedure_concept_id, pl.Int64, required=True),
        ColumnSpec(col.procedure_date, pl.Date, required=True),
        ColumnSpec(col.procedure_datetime, pl.Datetime, required=False),
        ColumnSpec(col.procedure_end_date, pl.Date, required=False),
        ColumnSpec(col.procedure_end_datetime, pl.Datetime, required=False),
        ColumnSpec(col.procedure_type_concept_id, pl.Int64, required=True),
        ColumnSpec(col.modifier_concept_id, pl.Int64, required=True),
        ColumnSpec(col.quantity, pl.Int64, required=False),
        ColumnSpec(col.provider_id, pl.Int64, required=False),
        ColumnSpec(col.visit_occurrence_id, pl.Int64, required=False),
        ColumnSpec(col.visit_detail_id, pl.Int64, required=False),
        ColumnSpec(col.procedure_source_value, pl.Utf8, required=False),
        ColumnSpec(col.procedure_source_concept_id, pl.Int64, required=True),
        ColumnSpec(col.modifier_source_value, pl.Utf8, required=False),
    ],
)

# OBSERVATION table schema. Concept ID columns that are nullable in the CDM DDL
# are required here because the converter always fills them (0 when unknown).
OBSERVATION_SCHEMA = TableSchema(
    name=tables.omop_observation,
    columns=[
        ColumnSpec(col.observation_id, pl.Int64, required=True, primary_key=True),
        ColumnSpec(col.person_id, pl.Int64, required=True),
        ColumnSpec(col.observation_concept_id, pl.Int64, required=True),
        ColumnSpec(col.observation_date, pl.Date, required=True),
        ColumnSpec(col.observation_datetime, pl.Datetime, required=False),
        ColumnSpec(col.observation_type_concept_id, pl.Int64, required=True),
        ColumnSpec(col.value_as_number, pl.Float64, required=False),
        ColumnSpec(col.value_as_string, pl.Utf8, required=False),
        ColumnSpec(col.value_as_concept_id, pl.Int64, required=True),
        ColumnSpec(col.qualifier_concept_id, pl.Int64, required=True),
        ColumnSpec(col.unit_concept_id, pl.Int64, required=True),
        ColumnSpec(col.provider_id, pl.Int64, required=False),
        ColumnSpec(col.visit_occurrence_id, pl.Int64, required=False),
        ColumnSpec(col.visit_detail_id, pl.Int64, required=False),
        ColumnSpec(col.observation_source_value, pl.Utf8, required=False),
        ColumnSpec(col.observation_source_concept_id, pl.Int64, required=True),
        ColumnSpec(col.unit_source_value, pl.Utf8, required=False),
        ColumnSpec(col.qualifier_source_value, pl.Utf8, required=False),
        ColumnSpec(col.value_source_value, pl.Utf8, required=False),
        ColumnSpec(col.observation_event_id, pl.Int64, required=False),
        ColumnSpec(col.obs_event_field_concept_id, pl.Int64, required=True),
    ],
)

# DEVICE_EXPOSURE table schema (same concept ID convention as OBSERVATION)
DEVICE_EXPOSURE_SCHEMA = TableSchema(
    name=tables.omop_device_exposure,
    columns=[
        ColumnSpec(col.device_exposure_id, pl.Int64, required=True, primary_key=True),
        ColumnSpec(col.person_id, pl.Int64, required=True),
        ColumnSpec(col.device_concept_id, pl.Int64, required=True),
        ColumnSpec(col.device_exposure_start_date, pl.Date, required=True),
        ColumnSpec(col.device_exposure_start_datetime, pl.Datetime, required=False),
        ColumnSpec(col.device_exposure_end_date, pl.Date, required=False),
        ColumnSpec(col.device_exposure_end_datetime, pl.Datetime, required=False),
        ColumnSpec(col.device_type_concept_id, pl.Int64, required=True),
        ColumnSpec(col.unique_device_id, pl.Utf8, required=False),
        ColumnSpec(col.production_id, pl.Utf8, required=False),
        ColumnSpec(col.quantity, pl.Int64, required=False),
        ColumnSpec(col.provider_id, pl.Int64, required=False),
        ColumnSpec(col.visit_occurrence_id, pl.Int64, required=False),
        ColumnSpec(col.visit_detail_id, pl.Int64, required=False),
        ColumnSpec(col.device_source_value, pl.Utf8, required=False),
        ColumnSpec(col.device_source_concept_id, pl.Int64, required=True),
        ColumnSpec(col.unit_concept_id, pl.Int64, required=True),
        ColumnSpec(col.unit_source_value, pl.Utf8, required=False),
        ColumnSpec(col.unit_source_concept_id, pl.Int64, required=True),
    ],
)

# DEATH table schema (person_id is the primary key: one row per person)
DEATH_SCHEMA = TableSchema(
    name=tables.omop_death,
    columns=[
        ColumnSpec(col.person_id, pl.Int64, required=True, primary_key=True),
        ColumnSpec(col.death_date, pl.Date, required=True),
        ColumnSpec(col.death_datetime, pl.Datetime, required=False),
        ColumnSpec(col.death_type_concept_id, pl.Int64, required=False),
        ColumnSpec(col.cause_concept_id, pl.Int64, required=False),
        ColumnSpec(col.cause_source_value, pl.Utf8, required=False),
        ColumnSpec(col.cause_source_concept_id, pl.Int64, required=False),
    ],
)

# OBSERVATION_PERIOD table schema
OBSERVATION_PERIOD_SCHEMA = TableSchema(
    name=tables.omop_observation_period,
    columns=[
        ColumnSpec(
            col.observation_period_id, pl.Int64, required=True, primary_key=True
        ),
        ColumnSpec(col.person_id, pl.Int64, required=True),
        ColumnSpec(col.observation_period_start_date, pl.Date, required=True),
        ColumnSpec(col.observation_period_end_date, pl.Date, required=True),
        ColumnSpec(col.period_type_concept_id, pl.Int64, required=True),
    ],
)

# CDM_SOURCE table schema (single row, no primary key)
CDM_SOURCE_SCHEMA = TableSchema(
    name=tables.omop_cdm_source,
    columns=[
        ColumnSpec(col.cdm_source_name, pl.Utf8, required=True),
        ColumnSpec(col.cdm_source_abbreviation, pl.Utf8, required=True),
        ColumnSpec(col.cdm_holder, pl.Utf8, required=True),
        ColumnSpec(col.source_description, pl.Utf8, required=False),
        ColumnSpec(col.source_documentation_reference, pl.Utf8, required=False),
        ColumnSpec(col.cdm_etl_reference, pl.Utf8, required=False),
        ColumnSpec(col.source_release_date, pl.Date, required=True),
        ColumnSpec(col.cdm_release_date, pl.Date, required=True),
        ColumnSpec(col.cdm_version, pl.Utf8, required=False),
        ColumnSpec(col.cdm_version_concept_id, pl.Int64, required=True),
        ColumnSpec(col.vocabulary_version, pl.Utf8, required=True),
    ],
)

# CONCEPT table schema
CONCEPT_SCHEMA = TableSchema(
    name=tables.omop_concept,
    columns=[
        ColumnSpec(col.concept_id, pl.Int64, required=True, primary_key=True),
        ColumnSpec(col.concept_name, pl.Utf8, required=True),
        ColumnSpec(col.domain_id, pl.Utf8, required=True),
        ColumnSpec(col.vocabulary_id, pl.Utf8, required=True),
        ColumnSpec(col.concept_class_id, pl.Utf8, required=True),
        ColumnSpec(col.standard_concept, pl.Utf8, required=False),
        ColumnSpec(col.concept_code, pl.Utf8, required=True),
        ColumnSpec(col.valid_start_date, pl.Date, required=True),
        ColumnSpec(col.valid_end_date, pl.Date, required=True),
        ColumnSpec(col.invalid_reason, pl.Utf8, required=False),
    ],
)

# CONDITION_ERA table schema (optional, see eras.py)
CONDITION_ERA_SCHEMA = TableSchema(
    name=tables.omop_condition_era,
    columns=[
        ColumnSpec(col.condition_era_id, pl.Int64, required=True, primary_key=True),
        ColumnSpec(col.person_id, pl.Int64, required=True),
        ColumnSpec(col.condition_concept_id, pl.Int64, required=True),
        ColumnSpec(col.condition_era_start_date, pl.Date, required=True),
        ColumnSpec(col.condition_era_end_date, pl.Date, required=True),
        ColumnSpec(col.condition_occurrence_count, pl.Int64, required=False),
    ],
)

# DRUG_ERA table schema (optional, see eras.py)
DRUG_ERA_SCHEMA = TableSchema(
    name=tables.omop_drug_era,
    columns=[
        ColumnSpec(col.drug_era_id, pl.Int64, required=True, primary_key=True),
        ColumnSpec(col.person_id, pl.Int64, required=True),
        ColumnSpec(col.drug_concept_id, pl.Int64, required=True),
        ColumnSpec(col.drug_era_start_date, pl.Date, required=True),
        ColumnSpec(col.drug_era_end_date, pl.Date, required=True),
        ColumnSpec(col.drug_exposure_count, pl.Int64, required=False),
        ColumnSpec(col.gap_days, pl.Int64, required=False),
    ],
)

# All OMOP schemas indexed by table name
OMOP_SCHEMAS: Dict[str, TableSchema] = {
    tables.omop_person: PERSON_SCHEMA,
    tables.omop_visit_occurrence: VISIT_OCCURRENCE_SCHEMA,
    tables.omop_condition_occurrence: CONDITION_OCCURRENCE_SCHEMA,
    tables.omop_measurement: MEASUREMENT_SCHEMA,
    tables.omop_drug_exposure: DRUG_EXPOSURE_SCHEMA,
    tables.omop_procedure_occurrence: PROCEDURE_OCCURRENCE_SCHEMA,
    tables.omop_observation: OBSERVATION_SCHEMA,
    tables.omop_device_exposure: DEVICE_EXPOSURE_SCHEMA,
    tables.omop_death: DEATH_SCHEMA,
    tables.omop_observation_period: OBSERVATION_PERIOD_SCHEMA,
    tables.omop_cdm_source: CDM_SOURCE_SCHEMA,
    tables.omop_concept: CONCEPT_SCHEMA,
}

# Standardized vocabulary tables, exported from Athena on request (see
# omop_vocab/vocabulary_export.py). CONCEPT is in OMOP_SCHEMAS because the converter
# always writes it. Athena leaves some columns empty that CDM 5.4 marks NOT NULL
# (vocabulary_reference, vocabulary_version), so those are not required here.
CONCEPT_RELATIONSHIP_SCHEMA = TableSchema(
    name=tables.omop_concept_relationship,
    columns=[
        ColumnSpec(col.concept_id_1, pl.Int64, required=True),
        ColumnSpec(col.concept_id_2, pl.Int64, required=True),
        ColumnSpec(col.relationship_id, pl.Utf8, required=True),
        ColumnSpec(col.valid_start_date, pl.Date, required=True),
        ColumnSpec(col.valid_end_date, pl.Date, required=True),
        ColumnSpec(col.invalid_reason, pl.Utf8, required=False),
    ],
)

CONCEPT_ANCESTOR_SCHEMA = TableSchema(
    name=tables.omop_concept_ancestor,
    columns=[
        ColumnSpec(col.ancestor_concept_id, pl.Int64, required=True),
        ColumnSpec(col.descendant_concept_id, pl.Int64, required=True),
        ColumnSpec(col.min_levels_of_separation, pl.Int64, required=True),
        ColumnSpec(col.max_levels_of_separation, pl.Int64, required=True),
    ],
)

VOCABULARY_SCHEMA = TableSchema(
    name=tables.omop_vocabulary,
    columns=[
        ColumnSpec(col.vocabulary_id, pl.Utf8, required=True, primary_key=True),
        ColumnSpec(col.vocabulary_name, pl.Utf8, required=True),
        ColumnSpec(col.vocabulary_reference, pl.Utf8, required=False),
        ColumnSpec(col.vocabulary_version, pl.Utf8, required=False),
        ColumnSpec(col.vocabulary_concept_id, pl.Int64, required=True),
    ],
)

DOMAIN_SCHEMA = TableSchema(
    name=tables.omop_domain,
    columns=[
        ColumnSpec(col.domain_id, pl.Utf8, required=True, primary_key=True),
        ColumnSpec(col.domain_name, pl.Utf8, required=True),
        ColumnSpec(col.domain_concept_id, pl.Int64, required=True),
    ],
)

CONCEPT_CLASS_SCHEMA = TableSchema(
    name=tables.omop_concept_class,
    columns=[
        ColumnSpec(col.concept_class_id, pl.Utf8, required=True, primary_key=True),
        ColumnSpec(col.concept_class_name, pl.Utf8, required=True),
        ColumnSpec(col.concept_class_concept_id, pl.Int64, required=True),
    ],
)

RELATIONSHIP_SCHEMA = TableSchema(
    name=tables.omop_relationship,
    columns=[
        ColumnSpec(col.relationship_id, pl.Utf8, required=True, primary_key=True),
        ColumnSpec(col.relationship_name, pl.Utf8, required=True),
        ColumnSpec(col.is_hierarchical, pl.Utf8, required=True),
        ColumnSpec(col.defines_ancestry, pl.Utf8, required=True),
        ColumnSpec(col.reverse_relationship_id, pl.Utf8, required=True),
        ColumnSpec(col.relationship_concept_id, pl.Int64, required=True),
    ],
)

CONCEPT_SYNONYM_SCHEMA = TableSchema(
    name=tables.omop_concept_synonym,
    columns=[
        ColumnSpec(col.concept_id, pl.Int64, required=True),
        ColumnSpec(col.concept_synonym_name, pl.Utf8, required=True),
        ColumnSpec(col.language_concept_id, pl.Int64, required=True),
    ],
)

DRUG_STRENGTH_SCHEMA = TableSchema(
    name=tables.omop_drug_strength,
    columns=[
        ColumnSpec(col.drug_concept_id, pl.Int64, required=True),
        ColumnSpec(col.ingredient_concept_id, pl.Int64, required=True),
        ColumnSpec(col.amount_value, pl.Float64, required=False),
        ColumnSpec(col.amount_unit_concept_id, pl.Int64, required=False),
        ColumnSpec(col.numerator_value, pl.Float64, required=False),
        ColumnSpec(col.numerator_unit_concept_id, pl.Int64, required=False),
        ColumnSpec(col.denominator_value, pl.Float64, required=False),
        ColumnSpec(col.denominator_unit_concept_id, pl.Int64, required=False),
        ColumnSpec(col.box_size, pl.Int64, required=False),
        ColumnSpec(col.valid_start_date, pl.Date, required=True),
        ColumnSpec(col.valid_end_date, pl.Date, required=True),
        ColumnSpec(col.invalid_reason, pl.Utf8, required=False),
    ],
)

# Optional vocabulary tables indexed by table name; validated when present, never
# required. Kept apart from OMOP_SCHEMAS, which lists the tables every output has.
VOCABULARY_SCHEMAS: Dict[str, TableSchema] = {
    tables.omop_concept_relationship: CONCEPT_RELATIONSHIP_SCHEMA,
    tables.omop_concept_ancestor: CONCEPT_ANCESTOR_SCHEMA,
    tables.omop_vocabulary: VOCABULARY_SCHEMA,
    tables.omop_domain: DOMAIN_SCHEMA,
    tables.omop_concept_class: CONCEPT_CLASS_SCHEMA,
    tables.omop_relationship: RELATIONSHIP_SCHEMA,
    tables.omop_concept_synonym: CONCEPT_SYNONYM_SCHEMA,
    tables.omop_drug_strength: DRUG_STRENGTH_SCHEMA,
}

# Era tables, written by `convert --eras`; validated when present, never required.
# Every column except the IDs and dates is derived, so they carry a person_id and
# concept IDs like the clinical tables.
ERA_SCHEMAS: Dict[str, TableSchema] = {
    tables.omop_condition_era: CONDITION_ERA_SCHEMA,
    tables.omop_drug_era: DRUG_ERA_SCHEMA,
}
