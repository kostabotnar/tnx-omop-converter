"""Column name constants for TriNetX and OMOP CDM tables."""

# Internal columns (added during processing)
_source_id = "_source_id"
_tnx_table = "_tnx_table"
_domain_id = "_domain_id"
batch = "batch"
person_rows = "person_rows"

# Internal event columns (source adapter output, input to domain builders)
_start_date = "_start_date"
_end_date = "_end_date"
_concept_id = "_concept_id"
_source_concept_id = "_source_concept_id"
_source_value = "_source_value"
_value_as_number = "_value_as_number"
_value_source_value = "_value_source_value"
_value_as_concept_id = "_value_as_concept_id"
_unit_concept_id = "_unit_concept_id"
_unit_source_value = "_unit_source_value"
_status_concept_id = "_status_concept_id"
_status_source_value = "_status_source_value"
_route_concept_id = "_route_concept_id"
_route_source_value = "_route_source_value"
_modifier_source_value = "_modifier_source_value"

# TriNetX source columns
# patient
patient_id = "patient_id"
year_of_birth = "year_of_birth"
sex = "sex"
race = "race"
ethnicity = "ethnicity"
month_year_death = "month_year_death"
derived_by_TriNetX = "derived_by_TriNetX"

# encounter
encounter_id = "encounter_id"
start_date = "start_date"
end_date = "end_date"
type = "type"

# diagnosis
date = "date"
code = "code"
code_system = "code_system"
principal_diagnosis_indicator = "principal_diagnosis_indicator"

# standardized_terminology
code_description = "code_description"

# lab_result
lab_result_num_val = "lab_result_num_val"
lab_result_text_val = "lab_result_text_val"
units_of_measure = "units_of_measure"

# vitals_signs
value = "value"
text_value = "text_value"

# medication_ingredient
unique_id = "unique_id"
route = "route"

# procedure
modifier_1 = "modifier_1"

# OMOP CDM target columns
# PERSON
person_id = "person_id"
gender_concept_id = "gender_concept_id"
month_of_birth = "month_of_birth"
day_of_birth = "day_of_birth"
birth_datetime = "birth_datetime"
race_concept_id = "race_concept_id"
ethnicity_concept_id = "ethnicity_concept_id"
location_id = "location_id"
provider_id = "provider_id"
care_site_id = "care_site_id"
person_source_value = "person_source_value"
gender_source_value = "gender_source_value"
gender_source_concept_id = "gender_source_concept_id"
race_source_value = "race_source_value"
race_source_concept_id = "race_source_concept_id"
ethnicity_source_value = "ethnicity_source_value"
ethnicity_source_concept_id = "ethnicity_source_concept_id"

# VISIT_OCCURRENCE
visit_occurrence_id = "visit_occurrence_id"
visit_concept_id = "visit_concept_id"
visit_start_date = "visit_start_date"
visit_start_datetime = "visit_start_datetime"
visit_end_date = "visit_end_date"
visit_end_datetime = "visit_end_datetime"
visit_type_concept_id = "visit_type_concept_id"
visit_source_value = "visit_source_value"
visit_source_concept_id = "visit_source_concept_id"
admitted_from_concept_id = "admitted_from_concept_id"
admitted_from_source_value = "admitted_from_source_value"
discharged_to_concept_id = "discharged_to_concept_id"
discharged_to_source_value = "discharged_to_source_value"
preceding_visit_occurrence_id = "preceding_visit_occurrence_id"

# CONDITION_OCCURRENCE
condition_occurrence_id = "condition_occurrence_id"
condition_concept_id = "condition_concept_id"
condition_start_date = "condition_start_date"
condition_start_datetime = "condition_start_datetime"
condition_end_date = "condition_end_date"
condition_end_datetime = "condition_end_datetime"
condition_type_concept_id = "condition_type_concept_id"
condition_status_concept_id = "condition_status_concept_id"
stop_reason = "stop_reason"
visit_occurrence_id = "visit_occurrence_id"
visit_detail_id = "visit_detail_id"
condition_source_value = "condition_source_value"
condition_source_concept_id = "condition_source_concept_id"
condition_status_source_value = "condition_status_source_value"

# MEASUREMENT
measurement_id = "measurement_id"
measurement_concept_id = "measurement_concept_id"
measurement_date = "measurement_date"
measurement_datetime = "measurement_datetime"
measurement_time = "measurement_time"
measurement_type_concept_id = "measurement_type_concept_id"
operator_concept_id = "operator_concept_id"
value_as_number = "value_as_number"
value_as_concept_id = "value_as_concept_id"
unit_concept_id = "unit_concept_id"
range_low = "range_low"
range_high = "range_high"
measurement_source_value = "measurement_source_value"
measurement_source_concept_id = "measurement_source_concept_id"
unit_source_value = "unit_source_value"
unit_source_concept_id = "unit_source_concept_id"
value_source_value = "value_source_value"
measurement_event_id = "measurement_event_id"
meas_event_field_concept_id = "meas_event_field_concept_id"

# DRUG_EXPOSURE
drug_exposure_id = "drug_exposure_id"
drug_concept_id = "drug_concept_id"
drug_exposure_start_date = "drug_exposure_start_date"
drug_exposure_start_datetime = "drug_exposure_start_datetime"
drug_exposure_end_date = "drug_exposure_end_date"
drug_exposure_end_datetime = "drug_exposure_end_datetime"
verbatim_end_date = "verbatim_end_date"
drug_type_concept_id = "drug_type_concept_id"
refills = "refills"
quantity = "quantity"
days_supply = "days_supply"
sig = "sig"
route_concept_id = "route_concept_id"
lot_number = "lot_number"
drug_source_value = "drug_source_value"
drug_source_concept_id = "drug_source_concept_id"
route_source_value = "route_source_value"
dose_unit_source_value = "dose_unit_source_value"

# PROCEDURE_OCCURRENCE
procedure_occurrence_id = "procedure_occurrence_id"
procedure_concept_id = "procedure_concept_id"
procedure_date = "procedure_date"
procedure_datetime = "procedure_datetime"
procedure_end_date = "procedure_end_date"
procedure_end_datetime = "procedure_end_datetime"
procedure_type_concept_id = "procedure_type_concept_id"
modifier_concept_id = "modifier_concept_id"
procedure_source_value = "procedure_source_value"
procedure_source_concept_id = "procedure_source_concept_id"
modifier_source_value = "modifier_source_value"

# OBSERVATION
observation_id = "observation_id"
observation_concept_id = "observation_concept_id"
observation_date = "observation_date"
observation_datetime = "observation_datetime"
observation_type_concept_id = "observation_type_concept_id"
value_as_string = "value_as_string"
qualifier_concept_id = "qualifier_concept_id"
observation_source_value = "observation_source_value"
observation_source_concept_id = "observation_source_concept_id"
qualifier_source_value = "qualifier_source_value"
observation_event_id = "observation_event_id"
obs_event_field_concept_id = "obs_event_field_concept_id"

# DEVICE_EXPOSURE
device_exposure_id = "device_exposure_id"
device_concept_id = "device_concept_id"
device_exposure_start_date = "device_exposure_start_date"
device_exposure_start_datetime = "device_exposure_start_datetime"
device_exposure_end_date = "device_exposure_end_date"
device_exposure_end_datetime = "device_exposure_end_datetime"
device_type_concept_id = "device_type_concept_id"
unique_device_id = "unique_device_id"
production_id = "production_id"
device_source_value = "device_source_value"
device_source_concept_id = "device_source_concept_id"

# DEATH
death_date = "death_date"
death_datetime = "death_datetime"
death_type_concept_id = "death_type_concept_id"
cause_concept_id = "cause_concept_id"
cause_source_value = "cause_source_value"
cause_source_concept_id = "cause_source_concept_id"

# OBSERVATION_PERIOD
observation_period_id = "observation_period_id"
observation_period_start_date = "observation_period_start_date"
observation_period_end_date = "observation_period_end_date"
period_type_concept_id = "period_type_concept_id"

# CDM_SOURCE
cdm_source_name = "cdm_source_name"
cdm_source_abbreviation = "cdm_source_abbreviation"
cdm_holder = "cdm_holder"
source_description = "source_description"
source_documentation_reference = "source_documentation_reference"
cdm_etl_reference = "cdm_etl_reference"
source_release_date = "source_release_date"
cdm_release_date = "cdm_release_date"
cdm_version = "cdm_version"
cdm_version_concept_id = "cdm_version_concept_id"
vocabulary_version = "vocabulary_version"

# TriNetX dataset_details and cohort_details
network_name = "network_name"
date_created = "date_created"
cohort_name = "cohort_name"

# CONCEPT
concept_id = "concept_id"
concept_name = "concept_name"
domain_id = "domain_id"
vocabulary_id = "vocabulary_id"
concept_class_id = "concept_class_id"
standard_concept = "standard_concept"
concept_code = "concept_code"
valid_start_date = "valid_start_date"
valid_end_date = "valid_end_date"
invalid_reason = "invalid_reason"

# Athena CONCEPT_RELATIONSHIP
concept_id_1 = "concept_id_1"
concept_id_2 = "concept_id_2"
relationship_id = "relationship_id"

# CONDITION_ERA (condition_concept_id is defined with CONDITION_OCCURRENCE)
condition_era_id = "condition_era_id"
condition_era_start_date = "condition_era_start_date"
condition_era_end_date = "condition_era_end_date"
condition_occurrence_count = "condition_occurrence_count"

# DRUG_ERA (drug_concept_id is defined with DRUG_EXPOSURE)
drug_era_id = "drug_era_id"
drug_era_start_date = "drug_era_start_date"
drug_era_end_date = "drug_era_end_date"
drug_exposure_count = "drug_exposure_count"
gap_days = "gap_days"

# Athena CONCEPT_ANCESTOR
ancestor_concept_id = "ancestor_concept_id"
descendant_concept_id = "descendant_concept_id"
min_levels_of_separation = "min_levels_of_separation"
max_levels_of_separation = "max_levels_of_separation"

# Athena VOCABULARY, DOMAIN, CONCEPT_CLASS and RELATIONSHIP
vocabulary_name = "vocabulary_name"
vocabulary_reference = "vocabulary_reference"
vocabulary_concept_id = "vocabulary_concept_id"
domain_name = "domain_name"
domain_concept_id = "domain_concept_id"
concept_class_name = "concept_class_name"
concept_class_concept_id = "concept_class_concept_id"
relationship_name = "relationship_name"
is_hierarchical = "is_hierarchical"
defines_ancestry = "defines_ancestry"
reverse_relationship_id = "reverse_relationship_id"
relationship_concept_id = "relationship_concept_id"

# Athena CONCEPT_SYNONYM
concept_synonym_name = "concept_synonym_name"
language_concept_id = "language_concept_id"

# Athena DRUG_STRENGTH (drug_concept_id is defined with DRUG_EXPOSURE)
ingredient_concept_id = "ingredient_concept_id"
amount_value = "amount_value"
amount_unit_concept_id = "amount_unit_concept_id"
numerator_value = "numerator_value"
numerator_unit_concept_id = "numerator_unit_concept_id"
denominator_value = "denominator_value"
denominator_unit_concept_id = "denominator_unit_concept_id"
box_size = "box_size"

# Vocabulary lookup and excluded-record output
source_concept_id = "source_concept_id"
exclusion_reason = "exclusion_reason"
target_domain_id = "target_domain_id"

# Coverage summary (coverage.csv)
tnx_table = "tnx_table"
outcome = "outcome"
rows = "rows"
rows_pct = "rows_pct"
codes = "codes"
codes_pct = "codes_pct"
