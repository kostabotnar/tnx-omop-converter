# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First release, planned as 0.1.0.

### Added

- `tnx-omop convert`: converts TriNetX electronic health record (EHR) exports (ZIP files of CSV) to OMOP (Observational Medical Outcomes Partnership) Common Data Model (CDM) 5.4 tables in Parquet format, using an Athena vocabulary download.
  - Batched processing: `--batch-rows` bounds the source rows per batch, so memory use depends on the batch size and not on the number of patients. The output does not depend on it.
  - Resumable runs: a rerun of the same command continues from the working folder (`--work-dir`, `--keep-work-dir`) when the inputs and options match.
  - Several ZIP files in one run.
  - Mapping of TriNetX codes to standard OMOP concepts through Athena "Maps to" relationships, with one output row per standard concept.
  - Routing of records to the OMOP table of the target concept's domain (condition, procedure, drug, measurement, observation, device).
  - Excluded records (no source concept, no standard mapping, or an unsupported domain) written to `excluded/` with an `exclusion_reason`, and a `coverage.csv` summary with rows, distinct codes and percentages per TriNetX table, code system and outcome.
  - `--config` JSON file to override the bundled concept maps.
  - Sequential IDs that do not depend on the batch size.
- Full Athena vocabulary tables (CONCEPT, CONCEPT_RELATIONSHIP, CONCEPT_ANCESTOR, VOCABULARY, DOMAIN, CONCEPT_CLASS, RELATIONSHIP, CONCEPT_SYNONYM, DRUG_STRENGTH) and CONDITION_ERA and DRUG_ERA written by default. `--no-export-vocabulary` and `--no-eras` turn them off.
- `tnx-omop validate`: checks an output folder for tables, columns, types, keys, person and concept references, standard concepts and domains, and agreement with `coverage.csv`. `--coverage` prints rows per table and excluded rows per reason.
- `tnx-omop coverage`: Markdown report of how many TriNetX codes and rows map to standard concepts in Athena.
- `tnx-omop dqd`: runs the OHDSI (Observational Health Data Sciences and Informatics) Data Quality Dashboard (DQD) on an output folder through R and DuckDB, with a list of accepted failures in `tnx_omop/config/dqd_accepted.json`. `tnx-omop dqd --install-r-packages` installs the required R packages from the installed package.
- `run_report.json`: written by a successful `convert` with versions, times, inputs, vocabulary version, row counts, excluded rows, and seconds and peak memory per stage.
- `tnx-omop example <output_dir>`: writes a small synthetic TriNetX export (`example_export.zip`) and a tiny vocabulary folder (`athena`) so that `convert` and `validate` can be tried without TriNetX data or an Athena download.
