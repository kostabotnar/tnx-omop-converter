# tnx-omop-converter

`tnx-omop` is a command line tool that converts TriNetX electronic health record (EHR) exports to OMOP Common Data Model (CDM) 5.4 tables in Parquet format. TriNetX codes are mapped to standard OMOP concepts with an Athena vocabulary download.

## Background

TriNetX is a federated health research network. Its data exports are ZIP files of CSV tables (patients, encounters, diagnoses, procedures, medications, lab results, vital signs) plus a data dictionary workbook, `datadictionary.xlsx`.

The OMOP CDM is a common data model of the Observational Medical Outcomes Partnership, maintained by the OHDSI community (Observational Health Data Sciences and Informatics). It stores clinical data in a fixed set of tables where every clinical fact refers to a standard concept from a shared vocabulary. Tools such as ATLAS and HADES, and the Data Quality Dashboard (DQD), work on data in this form.

This tool converts and validates. It reads the export, maps codes to standard concepts, writes the OMOP tables, and checks the result. It contains no cohort definition, no study logic and no analysis.

## Requirements

- Python 3.11 or later.
- An Athena vocabulary download. Athena is the vocabulary service of OHDSI at [athena.ohdsi.org](https://athena.ohdsi.org); a free account is needed. The CPT4 vocabulary is licensed by the American Medical Association (AMA); follow its terms when you download and use it.
- R and a Java JDK, only for the `dqd` command.

## Installation

Without cloning the repository, install the command from GitHub:

```bash
pip install git+https://github.com/kostabotnar/tnx-omop-converter
```

or, with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/kostabotnar/tnx-omop-converter
```

For development, clone the repository and run `uv sync`; see [CONTRIBUTING.md](CONTRIBUTING.md). In a checkout, run the commands below as `uv run tnx-omop ...`.

## Quick start

`tnx-omop example` writes a small synthetic TriNetX export and a small vocabulary excerpt, so you can try the tool without TriNetX data or an Athena account:

```bash
tnx-omop example demo
tnx-omop convert --input demo/example_export.zip --output demo/omop --vocab-dir demo/athena
tnx-omop validate demo/omop
```

`demo/example_export.zip` holds three invented patients with encounters, diagnoses, a procedure, medications, lab results and vital signs. `demo/athena` is a small excerpt of real Athena rows: only the concepts the example needs and their relationships, so it is for trying the commands and not for real data. The `convert` run logs that some optional vocabulary files are missing (DOMAIN, CONCEPT_CLASS, RELATIONSHIP, CONCEPT_SYNONYM, DRUG_STRENGTH) and skips those tables. `validate` should report that every check passed.

For your own data, give your TriNetX ZIP and your Athena folder:

```bash
tnx-omop convert --input <trinetx.zip> --output <output_dir> --vocab-dir <athena_dir>
tnx-omop validate <output_dir>
```

The optional `tnx-omop dqd <output_dir>` runs the OHDSI Data Quality Dashboard, see "Data Quality Dashboard" below. The example vocabulary is a small excerpt, so DQD results on the example are not meaningful for data quality.

## Usage

```bash
tnx-omop convert --input <trinetx.zip> [<more.zip> ...] --output <output_dir> --vocab-dir <athena_dir>
```

The column types of the TriNetX CSV files come from the `datadictionary.xlsx` workbook in each input ZIP. Every ZIP must contain one, and with several ZIP files the dictionaries must agree on table, column and data type; otherwise `convert` stops before any work.

`tnx-omop` has five subcommands: `convert`, `validate`, `coverage`, `dqd` and `example`. Options of `convert`:

- `--batch-rows N`: upper bound of source rows per batch of patients (default 5,000,000). Memory use grows with the batch size, not with the number of patients; the output does not depend on it.
- `--work-dir <dir>`: folder for intermediate files (default `<output_dir>/_work`). It needs disk space for a Parquet copy of the input and the batch parts.
- `--keep-work-dir`: keep the working folder after a successful run. It is deleted after success by default and left in place after a failure. A rerun of the same command resumes from the working folder when the inputs and options match.
- `--config <file.json>`: override the bundled concept maps, see "Config file" below. `coverage` accepts it too.
- `--no-export-vocabulary`: write only the referenced concepts instead of the full Athena vocabulary tables, see "Vocabulary export" below.
- `--no-eras`: skip CONDITION_ERA and DRUG_ERA, see "Eras" below.

The vocabulary export and the eras are on by default, so the output is complete for OHDSI tools such as the Data Quality Dashboard, ATLAS and HADES. Turn them off only when the output is not meant for those tools; `--export-vocabulary` and `--eras` are accepted and change nothing.

Progress messages go to stderr through `logging`. Give `-v`/`--verbose` (debug messages) or `-q`/`--quiet` (warnings and errors only) before the subcommand, for example `tnx-omop -q convert ...`. The results of `validate`, `coverage` and `dqd` are written to stdout, so they can be piped or redirected.

## Athena vocabulary

Download from [athena.ohdsi.org](https://athena.ohdsi.org) and unzip into `<athena_dir>`. Select SNOMED, ICD10CM, ICD9CM, ICD9Proc, ICD10PCS, CPT4, HCPCS, VA Class, CVX, RxNorm, RxNorm Extension, LOINC, UCUM and OMOP Extension. The converter needs `CONCEPT.csv`, `CONCEPT_RELATIONSHIP.csv`, `VOCABULARY.csv` and `CONCEPT_ANCESTOR.csv` unless `--no-eras` is given, and reads `CONCEPT_CPT4.csv` when present.

You do not need a UMLS key or the `cpt.bat`/`cpt.sh` step: CPT4 concept names are taken from the TriNetX `standardized_terminology.csv` descriptions.

## Conventions and limitations

- A coded value that is not in the concept maps (an unknown sex, race, ethnicity, visit type, route or lab value, or a unit with no UCUM concept) gets concept ID 0, and the original text stays in the `*_source_value` column.
- A diagnosis, procedure, medication, lab or vital record whose code has no source concept, no standard mapping, or a target concept in an unsupported domain is not written to the OMOP tables. It is kept in `excluded/` with the reason.
- Each diagnosis, procedure, medication, lab and vital record gets one output row per standard concept it maps to, so a code with several "Maps to" targets gives several rows.
- TriNetX medications have no end date, so the converter sets `drug_exposure_end_date` equal to the start date.
- CPT4 concept names are empty in Athena. They are filled from the TriNetX `standardized_terminology.csv` descriptions, so the names come from the export.
- IDs are sequential integers. The same input and vocabulary give the same IDs, but IDs are not stable across different exports or vocabulary versions.
- `person_source_value` holds the TriNetX `patient_id` and `visit_source_value` the `encounter_id`. TriNetX numbers patients and encounters separately in each archive, so equal IDs in two ZIP files are different patients; the converter keeps the files apart by their file names, which must therefore be unique.
- Records dated after a patient's recorded death month are dropped, and duplicates are removed during cleaning.

## Config file

`--config` takes a JSON file with any subset of these keys: `gender`, `race`, `ethnicity`, `visit_type`, `condition_status`, `route`, `lab_value` (TriNetX value to integer OMOP concept ID), `unit_ucum` (TriNetX unit to UCUM code) and `source_vocabularies` (TriNetX table, then code system, to an Athena vocabulary ID, or `null` for a code system with no OMOP vocabulary). The bundled maps are in `tnx_omop/config/`. Each map in the file is merged into the bundled one: entries are added or replaced and the others stay. `source_vocabularies` is merged per table, then per code system.

```json
{
  "gender": {"O": 8521},
  "visit_type": {"IMP": 9201},
  "unit_ucum": {"mg/dl": "mg/dL"},
  "source_vocabularies": {"procedure": {"CPT": "CPT4", "VA": null}}
}
```

An unknown key, a value of the wrong type or unreadable JSON stops the command with an error that names the key, before any work starts. The config changes the output, so a changed file (path or content) makes a rerun discard the old working folder. `run_report.json` records its path and SHA-256 hash (`config`, null without one).

## Output

- One folder per OMOP table: PERSON, VISIT_OCCURRENCE, CONDITION_OCCURRENCE, PROCEDURE_OCCURRENCE, DRUG_EXPOSURE, MEASUREMENT, OBSERVATION, DEVICE_EXPOSURE, DEATH, OBSERVATION_PERIOD, CDM_SOURCE and CONCEPT.
- Each record goes to the table of its standard concept's domain. A procedure code can land in DRUG_EXPOSURE or DEVICE_EXPOSURE, a diagnosis code in OBSERVATION.
- `excluded/<tnx_table>.parquet`: records without a source concept, without a standard mapping, or mapped to an unsupported domain, with an `exclusion_reason` column. The folder is replaced on each run.
- `coverage.csv`: one row per TriNetX table, code system and outcome, where the outcome is the OMOP table the rows went to or the exclusion reason. Columns `rows` (output rows), `rows_pct`, `codes` (distinct codes) and `codes_pct`; the percentages are of all output rows and distinct codes of the TriNetX table. A code that maps to concepts in several tables counts in each, so `codes_pct` of a TriNetX table can add up to more than 100.
- CONCEPT contains the full Athena vocabulary, or only the concepts referenced by the output with `--no-export-vocabulary`. `CDM_SOURCE.vocabulary_version` records the Athena version.
- `run_report.json`, see "Run report" below.

All IDs are sequential integers. `person_id` follows the sorted TriNetX `patient_id` (per ZIP file), `visit_occurrence_id` follows the sorted `person_id` and `encounter_id`, and record IDs follow the sort order of each output table (by `person_id`, date and concept). None of them depends on `--batch-rows`.

## Vocabulary export

By default `convert` writes the OMOP CDM 5.4 vocabulary tables from the Athena folder, one Parquet file each in `<output>/<table>/<table>.parquet`: `concept`, `concept_relationship`, `concept_ancestor`, `vocabulary`, `domain`, `concept_class`, `relationship`, `concept_synonym` and `drug_strength`. CONCEPT is then the full table, not sorted (sorting it would need gigabytes of memory); with `--no-export-vocabulary` it holds only the referenced concepts, sorted by `concept_id`. It is built the same way (CONCEPT.csv plus CONCEPT_CPT4.csv, empty CPT4 names filled from the TriNetX descriptions), so every row of the reduced table appears unchanged in the full one. A missing `CONCEPT_ANCESTOR.csv`, `DOMAIN.csv`, `CONCEPT_CLASS.csv`, `RELATIONSHIP.csv`, `CONCEPT_SYNONYM.csv` or `DRUG_STRENGTH.csv` is logged and that table is skipped. The export adds disk space (several hundred MB of Parquet for a full Athena download) and some run time, and it is not part of the resume check. Running again with `--no-export-vocabulary` removes the vocabulary tables of an earlier export, so the output folder always matches the options of the last run.

## Eras

By default `convert` also writes `condition_era` and `drug_era` (OMOP CDM 5.4), built from the merged CONDITION_OCCURRENCE and DRUG_EXPOSURE tables with the standard OHDSI era logic and a 30 day persistence window. `CONCEPT_ANCESTOR.csv` must be in the Athena folder; without it `convert` stops with an error before any work unless `--no-eras` is given.

- CONDITION_ERA: per person and condition concept (concept 0 is skipped), occurrences are merged when the next one starts at most 30 days after the latest end so far. An occurrence without an end date ends one day after its start. `condition_occurrence_count` is the number of occurrences in the era.
- DRUG_ERA: each drug concept is mapped to its ingredients through CONCEPT_ANCESTOR (standard concepts of class Ingredient in RxNorm or RxNorm Extension; an ingredient maps to itself). A drug with no ingredient is not in DRUG_ERA, and a drug with several ingredients gives one era per ingredient. The exposure end is `drug_exposure_end_date`, else the start plus `days_supply`, else the start plus one day. Overlapping or touching exposures are merged into sub-exposures, which are merged into eras with the 30 day window. `drug_exposure_count` is the number of exposures and `gap_days` is the era length in days minus the days its sub-exposures cover. TriNetX medications have no end date, so the converter sets `drug_exposure_end_date` to the start date and such an exposure covers 0 days.
- The eras are built per range of `person_id` of about `--batch-rows` source rows, so memory does not grow with the table size. IDs run from 1 in order of `person_id`, concept and start date, as in the other tables.
- They do not depend on the batches and are not part of the resume check: they are rebuilt from the merged tables on every run. A run with `--no-eras` removes the era tables of an earlier run in the same output folder. `validate` checks the era tables when they exist, including their person and concept references. The stage is named `eras` in the run report.

## Run report

A successful `convert` writes `<output_dir>/run_report.json` and logs the total time and the peak memory of the process in its last line. The report holds the tool, Python and Polars versions, start and end time (UTC), the inputs (path and size), the vocabulary folder and version, `batch_rows`, the batch and person counts, whether the run resumed from an earlier working folder, row counts per output table, excluded rows per TriNetX table, and the seconds and peak memory (bytes) of each stage. The stages are `raw`, `persons`, `batches`, `lookup`, `transform`, `merge`, `observation_period`, `cdm_source`, `eras` (not with `--no-eras`), `concept`, `vocabulary_export` (not with `--no-export-vocabulary`) and `coverage`; a stage taken over from an earlier run is missing.

Stage peaks are the highest of samples taken about every 0.2 s, so a short spike can be missed, and they are null where the platform cannot report current memory. `memory_counter` names the counters: on Windows private bytes (`PrivateUsage`) for the stages and `PeakPagefileUsage` for the process; on Linux resident memory (`VmRSS`) and `ru_maxrss`; on macOS only `ru_maxrss` for the process.

## Validation

```bash
tnx-omop validate <output_dir>              # check an output folder; exit code 1 when a check fails
tnx-omop validate <output_dir> --coverage   # also print rows per table and excluded rows per reason
```

`validate` checks that every table exists with the OMOP CDM 5.4 columns and types, that primary keys are unique and required columns have no nulls, that every `person_id` is in PERSON, that every concept ID is in CONCEPT, and that concepts are standard and in the domain of their table. It also compares `excluded/` and the table row counts with `coverage.csv`. Exported vocabulary tables and the era tables are checked for columns, types and primary keys when their files exist and are not required; era concepts must be conditions (CONDITION_ERA) or ingredients (DRUG_ERA).

The same checks are available as pytest tests in a development checkout: `uv run pytest tests/test_omop_validation.py -v --output-dir=<output_dir>`. `uv run pytest -q` runs the unit and end to end tests.

## Data Quality Dashboard

```bash
tnx-omop dqd --install-r-packages             # once: install the R packages at pinned versions
tnx-omop dqd <output_dir>                     # load into DuckDB and run the DQD checks
tnx-omop dqd <output_dir> --no-run            # summarize the results of the last run again
```

`dqd` runs the OHDSI Data Quality Dashboard (DQD) on an output folder. It needs R (`Rscript` on PATH, or `--rscript <path>`), a Java JDK for rJava, and the packages duckdb, DatabaseConnector, CommonDataModel and DataQualityDashboard, which `tnx-omop dqd --install-r-packages` installs (it runs the bundled `install_packages.R` with Rscript and takes `--rscript` like the normal run). The script `tnx_omop/r/run_dqd.R` creates every OMOP CDM 5.4 table in a DuckDB file from the official DDL, loads the output Parquet files into them by column name, and runs `DataQualityDashboard::executeDqChecks()`. A column that is not in the CDM, a value of the wrong type or a null in a required column stops the load with an error. DuckDB downloads its ICU extension (needed for the date checks) on each run, so the run needs internet access. The default `convert` output has everything DQD needs. For an output made with `--no-export-vocabulary` or `--no-eras` the command warns: the other vocabulary tables stay empty and the era checks fail.

The results go to `<output_dir>/dqd/dqd_results.json` (`--results-dir` to change); view them with `DataQualityDashboard::viewDqDashboard("<path>")` in R. The DuckDB file is removed at the end unless `--keep-database` is given. stdout gets one line per failed check and per check whose SQL failed, then the counts. The exit code is 1 when a failure or error is not accepted.

Accepted failures are listed in `tnx_omop/config/dqd_accepted.json` (or the file given with `--accepted`) as `{"accepted": [{"check_id": "...", "reason": "..."}]}`. `check_id` is a DQD check ID, such as `table_measurepersoncompleteness_provider`, or a pattern with `*`; every entry needs a reason. Patterns that match no failure are logged as warnings so the list stays current.

## Vocabulary coverage

```bash
tnx-omop coverage --input <trinetx.zip> [<more.zip> ...] --vocab-dir <athena_dir> [--output <report.md>]
```

Writes a Markdown report of how many TriNetX codes and rows map to standard concepts, to stdout and to `--output` (default `build/vocab_coverage_report.md`).

## Patient data

TriNetX exports and the OMOP output made from them contain patient-level data covered by data use agreements. Never post such files, or rows from them, in issues, discussions or pull requests. Use synthetic data, aggregate counts, or logs with data rows removed. See [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, tests and the patient data rule.

## Citation

If you use this software, please cite it with the metadata in [CITATION.cff](CITATION.cff).

## License

Apache License 2.0, see [LICENSE](LICENSE).

## Disclaimer

This project is not affiliated with or endorsed by TriNetX or OHDSI. It is provided as is and has not been validated for clinical decision making.
