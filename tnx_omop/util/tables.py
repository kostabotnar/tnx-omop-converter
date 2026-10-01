"""Table name constants for TriNetX and OMOP CDM tables."""

# TriNetX source tables (tnx_ prefix)
tnx_patient = "patient"
tnx_encounter = "encounter"
tnx_diagnosis = "diagnosis"
tnx_lab_result = "lab_result"
tnx_vitals_signs = "vitals_signs"
tnx_medication_ingredient = "medication_ingredient"
tnx_procedure = "procedure"
tnx_standardized_terminology = "standardized_terminology"
tnx_dataset_details = "dataset_details"
tnx_cohort_details = "cohort_details"

# OMOP CDM target tables (omop_ prefix)
omop_person = "PERSON"
omop_visit_occurrence = "VISIT_OCCURRENCE"
omop_condition_occurrence = "CONDITION_OCCURRENCE"
omop_measurement = "MEASUREMENT"
omop_drug_exposure = "DRUG_EXPOSURE"
omop_procedure_occurrence = "PROCEDURE_OCCURRENCE"
omop_death = "DEATH"
omop_observation_period = "OBSERVATION_PERIOD"
omop_cdm_source = "CDM_SOURCE"
omop_observation = "OBSERVATION"
omop_device_exposure = "DEVICE_EXPOSURE"

# Optional OMOP era tables, derived from the tables above (see eras.py)
omop_condition_era = "CONDITION_ERA"
omop_drug_era = "DRUG_ERA"

# OMOP vocabulary tables
omop_concept = "concept"
omop_concept_relationship = "concept_relationship"
omop_concept_ancestor = "concept_ancestor"
omop_vocabulary = "vocabulary"
omop_domain = "domain"
omop_concept_class = "concept_class"
omop_relationship = "relationship"
omop_concept_synonym = "concept_synonym"
omop_drug_strength = "drug_strength"

# Athena vocabulary download files (file stems, read as <stem>.csv)
athena_concept = "CONCEPT"
athena_concept_cpt4 = "CONCEPT_CPT4"
athena_concept_relationship = "CONCEPT_RELATIONSHIP"
athena_vocabulary = "VOCABULARY"
athena_concept_ancestor = "CONCEPT_ANCESTOR"
athena_domain = "DOMAIN"
athena_concept_class = "CONCEPT_CLASS"
athena_relationship = "RELATIONSHIP"
athena_concept_synonym = "CONCEPT_SYNONYM"
athena_drug_strength = "DRUG_STRENGTH"

# Output directory for records excluded from the OMOP tables
excluded = "excluded"
