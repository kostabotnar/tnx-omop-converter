"""OMOP domain routing constants.

Each mapped record goes to the OMOP table of its standard target concept's domain.
Targets in any domain not listed in DOMAIN_TABLES are excluded from the output.
"""

from typing import Dict

from .util import columns as col
from .util import tables as tbl

# Athena domain_id values
CONDITION = "Condition"
PROCEDURE = "Procedure"
DRUG = "Drug"
MEASUREMENT = "Measurement"
OBSERVATION = "Observation"
DEVICE = "Device"

DOMAIN_TABLES: Dict[str, str] = {
    CONDITION: tbl.omop_condition_occurrence,
    PROCEDURE: tbl.omop_procedure_occurrence,
    DRUG: tbl.omop_drug_exposure,
    MEASUREMENT: tbl.omop_measurement,
    OBSERVATION: tbl.omop_observation,
    DEVICE: tbl.omop_device_exposure,
}

EVENT_CONCEPT_COLUMNS: Dict[str, str] = {
    tbl.omop_condition_occurrence: col.condition_concept_id,
    tbl.omop_procedure_occurrence: col.procedure_concept_id,
    tbl.omop_drug_exposure: col.drug_concept_id,
    tbl.omop_measurement: col.measurement_concept_id,
    tbl.omop_observation: col.observation_concept_id,
    tbl.omop_device_exposure: col.device_concept_id,
}

EVENT_DATE_COLUMNS: Dict[str, str] = {
    tbl.omop_condition_occurrence: col.condition_start_date,
    tbl.omop_procedure_occurrence: col.procedure_date,
    tbl.omop_drug_exposure: col.drug_exposure_start_date,
    tbl.omop_measurement: col.measurement_date,
    tbl.omop_observation: col.observation_date,
    tbl.omop_device_exposure: col.device_exposure_start_date,
}

# exclusion_reason values
NO_SOURCE_CONCEPT = "no_source_concept"
NO_STANDARD_MAPPING = "no_standard_mapping"
UNSUPPORTED_DOMAIN = "unsupported_domain"
