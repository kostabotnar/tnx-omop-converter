"""End-to-end test of convert() on a tiny TriNetX zip and the Athena fixture."""

import json
import logging
import shutil
import sys
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import polars as pl
import pytest

from tnx_omop import __main__ as main
from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.omop_vocab.concept_table import table_path
from tnx_omop import ingest
from tnx_omop.batch_transform import IdCounters, order_table, transform_batch
from tnx_omop.converter import build_vocab_lookup, convert
from tnx_omop.domains import (
    EVENT_CONCEPT_COLUMNS,
    NO_SOURCE_CONCEPT,
    NO_STANDARD_MAPPING,
    UNSUPPORTED_DOMAIN,
)
from tnx_omop.merge import COVERAGE_FILE
from tnx_omop.omop_schema import ERA_SCHEMAS, OMOP_SCHEMAS, VOCABULARY_SCHEMAS
from tnx_omop.run_report import REPORT_FILE, REPORT_TEMP_FILE
from tnx_omop.transformers.clinical import COVERAGE_SCHEMA
from tnx_omop.transformers.sources import SOURCE_ADAPTERS
from tnx_omop.validation import validate
from tests import athena_fixture as fx
from tests.schema_asserts import assert_conforms
from tnx_omop.tnx_schema import DictionaryRow
from tests.tnx_dictionary import DICTIONARY, ROWS, add_dictionary

Rows = List[List[Optional[str]]]

# (file stem, header, rows) in the layout of a TriNetX export
_TNX_FILES: Dict[str, Dict[str, object]] = {
    tbl.tnx_patient: {
        "header": [
            "patient_id",
            "sex",
            "race",
            "ethnicity",
            "marital_status",
            "year_of_birth",
            "reason_yob_missing",
            "month_year_death",
            "death_date_source_id",
            "patient_regional_location",
            "source_id",
        ],
        "rows": [
            ["P1", "M", "White", "Not Hispanic or Latino", "Single", "1950", "Present", "", "", "Unknown", "EHR"],
            ["P2", "F", "Asian", "Hispanic or Latino", "Married", "1960", "Present", "202401", "EHR", "Unknown", "EHR"],
        ],
    },
    tbl.tnx_encounter: {
        "header": [
            "encounter_id",
            "patient_id",
            "start_date",
            "end_date",
            "type",
            "start_date_derived_by_TriNetX",
            "end_date_derived_by_TriNetX",
            "derived_by_TriNetX",
            "source_id",
        ],
        "rows": [
            ["E1", "P1", "20230101", "20230102", "IMP", "F", "F", "F", "EHR"],
            ["E2", "P1", "20230301", "20230301", "AMB", "F", "F", "F", "EHR"],
            ["E3", "P2", "20230201", "20230201", "EMER", "F", "F", "F", "EHR"],
            ["E4", "P2", "20230401", "20230401", "VR", "F", "F", "F", "EHR"],
        ],
    },
    tbl.tnx_diagnosis: {
        "header": [
            "patient_id",
            "encounter_id",
            "code_system",
            "code",
            "principal_diagnosis_indicator",
            "admitting_diagnosis",
            "reason_for_visit",
            "date",
            "derived_by_TriNetX",
            "source_id",
        ],
        "rows": [
            ["P1", "E1", "ICD-10-CM", "E11.65", "Primary", "F", "F", "20230101", "F", "EHR"],
            ["P1", "E1", "ICD-10-CM", "Z87.891", "Unknown", "F", "F", "20230101", "F", "EHR"],
            ["P1", "E2", "ICD-10-CM", "C79.51", "Unknown", "F", "F", "20230301", "F", "EHR"],
            ["P1", "", "SNOMED", "44054006", "Unknown", "F", "F", "20230301", "T", "TriNetX"],
            ["P2", "E3", "ICD-10-CM", "C91.00", "Unknown", "F", "F", "20230201", "F", "EHR"],
            ["P2", "E3", "ICD-9-CM", "38.93", "Unknown", "F", "F", "20230201", "F", "EHR"],
            # After the death month of P2: dropped by cleaning
            ["P2", "", "ICD-10-CM", "I10", "Unknown", "F", "F", "20240315", "F", "EHR"],
        ],
    },
    tbl.tnx_procedure: {
        "header": [
            "patient_id",
            "encounter_id",
            "code_system",
            "code",
            "principal_procedure_indicator",
            "date",
            "derived_by_TriNetX",
            "source_id",
        ],
        "rows": [
            ["P1", "E1", "CPT", "80053", "Unknown", "20230101", "F", "EHR"],
            ["P1", "E1", "CPT", "99213", "Unknown", "20230101", "F", "EHR"],
            ["P2", "E3", "HCPCS", "J1100", "Unknown", "20230201", "F", "EHR"],
            ["P2", "E3", "HCPCS", "E0601", "Unknown", "20230201", "F", "EHR"],
            ["P1", "E2", "CVX", "207", "Unknown", "20230301", "F", "EHR"],
        ],
    },
    tbl.tnx_medication_ingredient: {
        "header": [
            "patient_id",
            "encounter_id",
            "unique_id",
            "code_system",
            "code",
            "start_date",
            "route",
            "brand",
            "strength",
            "medication_source",
            "derived_by_TriNetX",
            "source_id",
        ],
        "rows": [
            ["P1", "E1", "M1", "RxNorm", "1191", "20230101", "Oral Product", "Unknown", "Unknown", "", "F", "EHR"],
            ["P1", "E2", "M2", "RxNorm", "OMOP123", "20230301", "Unknown", "Unknown", "Unknown", "", "F", "EHR"],
            ["P2", "E3", "M3", "RxNorm", "999999", "20230201", "Unknown", "Unknown", "Unknown", "", "F", "EHR"],
        ],
    },
    tbl.tnx_lab_result: {
        "header": [
            "patient_id",
            "encounter_id",
            "code_system",
            "code",
            "date",
            "lab_result_num_val",
            "lab_result_text_val",
            "units_of_measure",
            "derived_by_TriNetX",
            "source_id",
        ],
        "rows": [
            ["P1", "E1", "LOINC", "2160-0", "20230101", "1.2", "", "mg/dL", "F", "EHR"],
            ["P1", "E1", "LOINC", "72166-2", "20230101", "", "Positive", "", "F", "EHR"],
            ["P2", "E3", "TNX", "9037-0", "20230201", "7.0", "", "%", "F", "EHR"],
        ],
    },
    tbl.tnx_vitals_signs: {
        "header": [
            "patient_id",
            "encounter_id",
            "code_system",
            "code",
            "date",
            "value",
            "text_value",
            "units_of_measure",
            "derived_by_TriNetX",
            "source_id",
        ],
        "rows": [
            ["P1", "E1", "LOINC", "8480-6", "20230101", "120", "", "mm[Hg]", "F", "EHR"],
        ],
    },
    tbl.tnx_standardized_terminology: {
        "header": ["code_system", "code", "code_description", "path", "unit"],
        "rows": [
            ["CPT", "80053", "Comprehensive metabolic panel", "0/80053", ""],
            ["CPT", "99213", "Office visit, established patient, low", "0/99213", ""],
        ],
    },
    tbl.tnx_dataset_details: {
        "header": [
            "total_number_unique_patients",
            "total_number_HCOs",
            "date_created",
            "network_name",
        ],
        "rows": [["2", "1", "20260219", "Test Network"]],
    },
    tbl.tnx_cohort_details: {
        "header": ["cohort_name", "cohort_number", "total_patient_records"],
        "rows": [["Test cohort", "1", "2"]],
    },
}  # fmt: skip


def _csv(header: List[str], rows: Rows) -> str:
    def quote(value: Optional[str]) -> str:
        return f'"{value}"' if value else ""

    lines = [",".join(quote(h) for h in header)]
    lines += [",".join(quote(v) for v in row) for row in rows]
    return "\n".join(lines) + "\n"


def _write_zip(path: Path, dictionary: Optional[List[DictionaryRow]] = ROWS) -> Path:
    """Write the test export; dictionary None leaves out datadictionary.xlsx."""
    with zipfile.ZipFile(path, "w") as zf:
        for stem, spec in _TNX_FILES.items():
            zf.writestr(f"{stem}.csv", _csv(spec["header"], spec["rows"]))
        if dictionary is not None:
            add_dictionary(zf, rows=dictionary)
    return path


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory: pytest.TempPathFactory, athena_dir: Path) -> Path:
    root = tmp_path_factory.mktemp("convert")
    zip_path = _write_zip(root / "tiny_export.zip")
    out = root / "omop"
    # The tests of the exported vocabulary and of the eras have their own outputs.
    convert([zip_path], out, athena_dir, export_vocabulary=False, eras=False)
    return out


def read(output_dir: Path, table_name: str) -> pl.DataFrame:
    return pl.read_parquet(table_path(output_dir, table_name))


def concept_ids_in_tables(output_dir: Path) -> Set[int]:
    ids: Set[int] = set()
    for table_name in OMOP_SCHEMAS:
        if table_name == tbl.omop_concept:
            continue
        df = read(output_dir, table_name)
        for name in df.columns:
            if name.endswith("_concept_id"):
                ids.update(v for v in df.get_column(name).to_list() if v)
    return ids


class TestOutputTables:
    @pytest.mark.parametrize("table_name", list(OMOP_SCHEMAS))
    def test_table_matches_schema(self, output_dir: Path, table_name: str):
        assert_conforms(read(output_dir, table_name), OMOP_SCHEMAS[table_name])

    @pytest.mark.parametrize(
        "table_name", [t for t, s in OMOP_SCHEMAS.items() if s.primary_key]
    )
    def test_primary_keys_unique(self, output_dir: Path, table_name: str):
        key = read(output_dir, table_name).get_column(
            OMOP_SCHEMAS[table_name].primary_key
        )
        assert key.null_count() == 0
        assert key.n_unique() == key.len()

    def test_row_counts(self, output_dir: Path):
        heights = {t: read(output_dir, t).height for t in OMOP_SCHEMAS}
        assert heights[tbl.omop_person] == 2
        assert heights[tbl.omop_visit_occurrence] == 4
        assert heights[tbl.omop_death] == 1
        # E11.65 (two targets), C91.00 (Condition target), ICD-9-CM 38.93
        assert heights[tbl.omop_condition_occurrence] == 4
        # Z87.891 from diagnosis, 72166-2 from lab_result
        assert heights[tbl.omop_observation] == 2
        assert heights[tbl.omop_device_exposure] == 1
        assert heights[tbl.omop_observation_period] == 2
        assert heights[tbl.omop_cdm_source] == 1

    def test_no_intermediate_files_in_output(self, output_dir: Path):
        expected = {t.lower() for t in OMOP_SCHEMAS} | {
            tbl.excluded,
            REPORT_FILE,
            COVERAGE_FILE,
        }
        assert {p.name for p in output_dir.iterdir()} == expected
        for table_name in OMOP_SCHEMAS:
            files = list(table_path(output_dir, table_name).parent.iterdir())
            assert [f.name for f in files] == [f"{table_name.lower()}.parquet"]


@pytest.fixture(scope="module")
def report(output_dir: Path) -> dict:
    return json.loads((output_dir / REPORT_FILE).read_text(encoding="utf-8"))


class TestRunReport:
    def test_keys(self, report: dict):
        assert set(report) == {
            "tool",
            "python_version",
            "polars_version",
            "started_at",
            "finished_at",
            "total_seconds",
            "inputs",
            "vocabulary",
            "config",
            "batch_rows",
            "batch_count",
            "person_count",
            "resumed",
            "ingest_stages_skipped",
            "batches_already_done",
            "row_counts",
            "excluded_rows",
            "stages",
            "process_peak_memory_bytes",
            "memory_counter",
        }

    def test_no_config(self, report: dict):
        assert report["config"] is None

    def test_run_description(self, report: dict, output_dir: Path):
        zip_path = output_dir.parent / "tiny_export.zip"
        assert report["inputs"] == [
            {"path": str(zip_path), "size_bytes": zip_path.stat().st_size}
        ]
        assert report["vocabulary"]["version"] == fx.VOCABULARY_VERSION
        assert report["polars_version"] == pl.__version__
        assert report["batch_count"] == 1
        assert report["person_count"] == 2
        assert report["resumed"] is False
        started = datetime.fromisoformat(report["started_at"])
        finished = datetime.fromisoformat(report["finished_at"])
        assert started.utcoffset() == timedelta(0)
        assert started <= finished
        assert report["total_seconds"] >= 0

    def test_row_counts_match_the_output(self, report: dict, output_dir: Path):
        assert report["row_counts"] == {
            t: read(output_dir, t).height for t in OMOP_SCHEMAS
        }

    def test_excluded_rows_match_the_excluded_files(
        self, report: dict, output_dir: Path
    ):
        by_table = {
            t: pl.read_parquet(output_dir / tbl.excluded / f"{t}.parquet").height
            for t in SOURCE_ADAPTERS
        }
        assert report["excluded_rows"] == {t: n for t, n in by_table.items() if n}

    def test_stages_and_memory(self, report: dict):
        assert list(report["stages"]) == [
            *ingest.STAGES,
            "lookup",
            "transform",
            "merge",
            "observation_period",
            "cdm_source",
            "concept",
            "coverage",
        ]
        for stage in report["stages"].values():
            assert stage["seconds"] >= 0
            assert stage["peak_memory_bytes"] is None or stage["peak_memory_bytes"] > 0
        assert report["process_peak_memory_bytes"] > 0
        assert report["memory_counter"]

    def test_no_temporary_file_left(self, output_dir: Path):
        assert not (output_dir / REPORT_TEMP_FILE).exists()


class TestSequentialIds:
    EVENT_TABLES = list(EVENT_CONCEPT_COLUMNS)

    def test_person_ids_follow_sorted_patient_ids(self, output_dir: Path):
        person = read(output_dir, tbl.omop_person)
        assert person.select(col.person_id, col.person_source_value).rows() == [
            (1, "P1"),
            (2, "P2"),
        ]

    def test_visit_ids_follow_sorted_encounter_ids(self, output_dir: Path):
        visit = read(output_dir, tbl.omop_visit_occurrence)
        assert visit.select(col.visit_occurrence_id, col.visit_source_value).rows() == [
            (1, "E1"),
            (2, "E2"),
            (3, "E3"),
            (4, "E4"),
        ]
        assert visit.get_column(col.person_id).to_list() == [1, 1, 2, 2]

    @pytest.mark.parametrize("table_name", EVENT_TABLES)
    def test_record_ids_are_one_to_n(self, output_dir: Path, table_name: str):
        key = read(output_dir, table_name).get_column(
            OMOP_SCHEMAS[table_name].primary_key
        )
        assert key.to_list() == list(range(1, key.len() + 1))

    @pytest.mark.parametrize("table_name", EVENT_TABLES)
    def test_foreign_keys_match_person_and_visit(
        self, output_dir: Path, table_name: str
    ):
        df = read(output_dir, table_name)
        persons = set(read(output_dir, tbl.omop_person)[col.person_id])
        visits = set(
            read(output_dir, tbl.omop_visit_occurrence)[col.visit_occurrence_id]
        )
        assert set(df[col.person_id]) <= persons
        assert set(df[col.visit_occurrence_id].drop_nulls()) <= visits

    @pytest.mark.parametrize("table_name", EVENT_TABLES + [tbl.omop_visit_occurrence])
    def test_sorted_by_person_id(self, output_dir: Path, table_name: str):
        person_ids = read(output_dir, table_name)[col.person_id].to_list()
        assert person_ids == sorted(person_ids)


class TestOrderTable:
    SCHEMA = OMOP_SCHEMAS[tbl.omop_condition_occurrence]
    ROWS = [(2, 5, 5), (1, 5, 7), (1, 5, 6), (1, 1, 9)]  # person, day, concept

    def frame(self, rows) -> pl.LazyFrame:
        df = pl.DataFrame(
            {
                col.person_id: [r[0] for r in rows],
                col.condition_start_date: [date(2023, 1, r[1]) for r in rows],
                col.condition_concept_id: [r[2] for r in rows],
            }
        ).with_columns(
            pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in self.SCHEMA.dtype_map.items()
            if name
            not in (col.person_id, col.condition_start_date, col.condition_concept_id)
        )
        return df.select(self.SCHEMA.column_names).cast(self.SCHEMA.dtype_map).lazy()

    def ordered(self, rows, offset: int = 0) -> pl.DataFrame:
        return order_table(
            self.frame(rows), tbl.omop_condition_occurrence, self.SCHEMA, offset
        ).collect()

    def test_ids_sequential_in_sort_order(self):
        result = self.ordered(self.ROWS)

        assert result.columns == self.SCHEMA.column_names
        assert dict(result.schema) == self.SCHEMA.dtype_map
        assert result.select(
            col.condition_occurrence_id,
            col.person_id,
            col.condition_start_date,
            col.condition_concept_id,
        ).rows() == [
            (1, 1, date(2023, 1, 1), 9),
            (2, 1, date(2023, 1, 5), 6),
            (3, 1, date(2023, 1, 5), 7),
            (4, 2, date(2023, 1, 5), 5),
        ]

    def test_ids_continue_after_offset(self):
        result = self.ordered(self.ROWS, offset=10)
        assert result.get_column(col.condition_occurrence_id).to_list() == [
            11,
            12,
            13,
            14,
        ]

    def test_ids_do_not_depend_on_row_order(self):
        assert self.ordered(self.ROWS).equals(self.ordered(self.ROWS[::-1]))


class TestRouting:
    def source_values(self, output_dir: Path, table_name: str, column: str) -> Set:
        return set(read(output_dir, table_name).get_column(column).to_list())

    def test_procedure_codes_routed_by_domain(self, output_dir: Path):
        assert self.source_values(
            output_dir, tbl.omop_procedure_occurrence, col.procedure_source_value
        ) == {"CPT:99213"}
        assert self.source_values(
            output_dir, tbl.omop_drug_exposure, col.drug_source_value
        ) == {"1191", "OMOP123", "HCPCS:J1100"}
        assert self.source_values(
            output_dir, tbl.omop_device_exposure, col.device_source_value
        ) == {"HCPCS:E0601"}
        assert self.source_values(
            output_dir, tbl.omop_measurement, col.measurement_source_value
        ) == {"CPT:80053", "2160-0", "8480-6"}

    def test_diagnosis_and_lab_to_observation(self, output_dir: Path):
        assert self.source_values(
            output_dir, tbl.omop_observation, col.observation_concept_id
        ) == {fx.OBS_EX_SMOKER, fx.LOINC_72166_2}

    def test_condition_targets(self, output_dir: Path):
        assert self.source_values(
            output_dir, tbl.omop_condition_occurrence, col.condition_concept_id
        ) == {fx.COND_T2DM, fx.COND_HYPERGLYCEMIA, fx.COND_ALL, fx.COND_ICD9_TARGET}

    def test_measurement_units(self, output_dir: Path):
        units = dict(
            read(output_dir, tbl.omop_measurement)
            .select(col.measurement_source_value, col.unit_concept_id)
            .iter_rows()
        )
        assert units == {
            "2160-0": fx.UCUM_MG_DL,
            "8480-6": fx.UCUM_MM_HG,
            "CPT:80053": 0,
        }


class TestExcluded:
    def test_one_file_per_clinical_table(self, output_dir: Path):
        names = {p.name for p in (output_dir / tbl.excluded).iterdir()}
        assert names == {f"{t}.parquet" for t in SOURCE_ADAPTERS}

    def test_diagnosis_reasons(self, output_dir: Path):
        df = pl.read_parquet(output_dir / tbl.excluded / f"{tbl.tnx_diagnosis}.parquet")
        reasons = dict(df.select(col.code, col.exclusion_reason).iter_rows())
        assert reasons == {
            "C79.51": NO_STANDARD_MAPPING,
            "44054006": NO_SOURCE_CONCEPT,
            "C91.00": UNSUPPORTED_DOMAIN,
        }
        assert {col.target_domain_id, col.source_concept_id} <= set(df.columns)

    def test_coverage_csv(self, output_dir: Path):
        coverage = pl.read_csv(output_dir / COVERAGE_FILE, schema=COVERAGE_SCHEMA)
        by_outcome = dict(
            coverage.group_by(col.outcome).agg(pl.col(col.rows).sum()).iter_rows()
        )
        for table_name in [
            tbl.omop_condition_occurrence,
            tbl.omop_procedure_occurrence,
            tbl.omop_drug_exposure,
            tbl.omop_measurement,
            tbl.omop_observation,
            tbl.omop_device_exposure,
        ]:
            assert by_outcome[table_name] == read(output_dir, table_name).height
        excluded_rows = sum(
            pl.read_parquet(output_dir / tbl.excluded / f"{t}.parquet").height
            for t in SOURCE_ADAPTERS
        )
        reasons = [NO_SOURCE_CONCEPT, NO_STANDARD_MAPPING, UNSUPPORTED_DOMAIN]
        assert sum(by_outcome.get(r, 0) for r in reasons) == excluded_rows

    def test_coverage_percentages_and_codes(self, output_dir: Path):
        coverage = pl.read_csv(output_dir / COVERAGE_FILE, schema=COVERAGE_SCHEMA)
        pct_sums = coverage.group_by(col.tnx_table).agg(pl.col(col.rows_pct).sum())
        for _, pct in pct_sums.iter_rows():
            assert pct == pytest.approx(100, abs=0.05)

        excluded = pl.read_parquet(
            output_dir / tbl.excluded / f"{tbl.tnx_diagnosis}.parquet"
        )
        expected = dict(
            excluded.group_by(col.exclusion_reason)
            .agg(pl.col(col.code).n_unique())
            .iter_rows()
        )
        diagnosis = coverage.filter(pl.col(col.tnx_table) == tbl.tnx_diagnosis)
        codes = dict(
            diagnosis.filter(pl.col(col.outcome).is_in(list(expected)))
            .group_by(col.outcome)
            .agg(pl.col(col.codes).sum())
            .iter_rows()
        )
        assert codes == expected
        assert (diagnosis[col.codes_pct] > 0).all()


class TestVocabularyTables:
    def test_cdm_source_vocabulary_version(self, output_dir: Path):
        cdm_source = read(output_dir, tbl.omop_cdm_source)
        assert cdm_source.item(0, col.vocabulary_version) == fx.VOCABULARY_VERSION

    def test_concept_has_exactly_the_referenced_ids(self, output_dir: Path):
        concept_ids = set(read(output_dir, tbl.omop_concept)[col.concept_id])
        assert concept_ids == concept_ids_in_tables(output_dir)

    def test_cpt4_concept_named_from_terminology(self, output_dir: Path):
        names = dict(
            read(output_dir, tbl.omop_concept)
            .select(col.concept_id, col.concept_name)
            .iter_rows()
        )
        assert names[fx.CPT4_80053] == "Comprehensive metabolic panel"
        assert names[fx.CPT4_99213] == "Office visit, established patient, low"


class TestValidation:
    def test_converted_output_passes_every_check(self, output_dir: Path):
        report = validate(output_dir)

        assert report.ok, "\n".join(str(i) for r in report.failed for i in r.issues)
        assert not report.skipped

    def test_damaged_output_fails(self, output_dir: Path, tmp_path: Path):
        damaged = tmp_path / "damaged"
        shutil.copytree(output_dir, damaged)
        table_path(damaged, tbl.omop_visit_occurrence).unlink()
        person = table_path(damaged, tbl.omop_person)
        pl.read_parquet(person).with_columns(
            pl.lit(1).alias(col.person_id)
        ).write_parquet(person)

        failed = {r.name for r in validate(damaged).failed}

        assert "all tables present" in failed
        assert f"{tbl.omop_visit_occurrence} schema" in failed
        assert f"{tbl.omop_person} schema" in failed


def test_stale_excluded_files_removed(tmp_path: Path, athena_dir: Path):
    out = tmp_path / "omop"
    stale = out / tbl.excluded / "old_table.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    convert([_write_zip(tmp_path / "tiny_export.zip")], out, athena_dir)

    assert not stale.exists()
    assert (out / COVERAGE_FILE).exists()


def test_no_missing_concepts_warning(
    tmp_path: Path, athena_dir: Path, caplog: pytest.LogCaptureFixture
):
    zip_path = _write_zip(tmp_path / "tiny_export.zip")
    with caplog.at_level(logging.WARNING):
        convert([zip_path], tmp_path / "omop", athena_dir)
    assert "not in the Athena vocabulary" not in caplog.text


def _two_source_zips(root: Path) -> List[Path]:
    """Two sources with the same content, so four persons in all."""
    return [_write_zip(root / "src_a.zip"), _write_zip(root / "src_b.zip")]


def _read_output(out: Path) -> Dict[str, pl.DataFrame]:
    frames = {t: read(out, t) for t in [*OMOP_SCHEMAS, *ERA_SCHEMAS]}
    # The exported vocabulary tables are written unsorted (see vocabulary_export.py)
    for t in [tbl.omop_concept, *VOCABULARY_SCHEMAS]:
        frames[t] = read(out, t).sort(pl.all())
    for path in sorted((out / tbl.excluded).iterdir()):
        frames[f"{tbl.excluded}/{path.name}"] = (
            pl.read_csv(path) if path.suffix == ".csv" else pl.read_parquet(path)
        )
    return frames


@pytest.fixture(scope="module")
def batch_outputs(
    tmp_path_factory: pytest.TempPathFactory, athena_dir: Path
) -> Dict[int, Path]:
    root = tmp_path_factory.mktemp("batches")
    zips = _two_source_zips(root)
    outputs = {}
    for batch_rows in (1, 1_000_000):
        outputs[batch_rows] = root / f"omop_{batch_rows}"
        convert(zips, outputs[batch_rows], athena_dir, batch_rows=batch_rows)
    return outputs


class TestBatches:
    def test_output_identical_for_any_batch_rows(self, batch_outputs: Dict[int, Path]):
        one = _read_output(batch_outputs[1])
        large = _read_output(batch_outputs[1_000_000])

        assert one.keys() == large.keys()
        for name, frame in one.items():
            assert frame.equals(large[name]), name

    def test_person_visit_and_record_ids_are_one_to_n_across_batches(
        self, batch_outputs: Dict[int, Path]
    ):
        out = batch_outputs[1]
        assert read(out, tbl.omop_person)[col.person_id].to_list() == [1, 2, 3, 4]
        visits = read(out, tbl.omop_visit_occurrence)[col.visit_occurrence_id]
        assert visits.to_list() == list(range(1, 9))
        for table_name in EVENT_CONCEPT_COLUMNS:
            key = read(out, table_name).get_column(OMOP_SCHEMAS[table_name].primary_key)
            assert key.to_list() == list(range(1, key.len() + 1))
        assert read(out, tbl.omop_condition_occurrence).height == 8

    def test_visit_ids_of_records_point_to_own_persons_visits(
        self, batch_outputs: Dict[int, Path]
    ):
        out = batch_outputs[1]
        visit_person = dict(
            read(out, tbl.omop_visit_occurrence)
            .select(col.visit_occurrence_id, col.person_id)
            .iter_rows()
        )
        measurement = read(out, tbl.omop_measurement).drop_nulls(
            col.visit_occurrence_id
        )
        assert measurement.height > 0
        for visit_id, person_id in measurement.select(
            col.visit_occurrence_id, col.person_id
        ).iter_rows():
            assert visit_person[visit_id] == person_id

    def test_coverage_counts_all_batches(self, batch_outputs: Dict[int, Path]):
        coverage = pl.read_csv(batch_outputs[1] / COVERAGE_FILE, schema=COVERAGE_SCHEMA)
        by_outcome = dict(
            coverage.group_by(col.outcome).agg(pl.col(col.rows).sum()).iter_rows()
        )
        assert by_outcome[tbl.omop_condition_occurrence] == 8


class TestWorkDir:
    def test_default_work_dir_deleted_after_success(
        self, tmp_path: Path, athena_dir: Path
    ):
        out = tmp_path / "omop"
        convert([_write_zip(tmp_path / "tiny_export.zip")], out, athena_dir)

        assert not (out / "_work").exists()

    def test_custom_work_dir_deleted_but_foreign_files_kept(
        self, tmp_path: Path, athena_dir: Path
    ):
        work = tmp_path / "work"
        work.mkdir()
        (work / "notes.txt").write_text("keep")

        convert(
            [_write_zip(tmp_path / "tiny_export.zip")],
            tmp_path / "omop",
            athena_dir,
            work_dir=work,
        )

        assert [p.name for p in work.iterdir()] == ["notes.txt"]

    def test_keep_work_dir(self, tmp_path: Path, athena_dir: Path):
        work = tmp_path / "work"
        convert(
            [_write_zip(tmp_path / "tiny_export.zip")],
            tmp_path / "omop",
            athena_dir,
            work_dir=work,
            batch_rows=1,
            keep_work_dir=True,
        )

        assert (work / "persons.parquet").exists()
        assert (work / "parts" / "coverage" / "00001.parquet").exists()
        assert not (work / "scratch").exists()

    def test_work_dir_kept_after_failure(
        self, tmp_path: Path, athena_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def fail(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("tnx_omop.converter.run_batches", fail)
        out = tmp_path / "omop"

        with pytest.raises(RuntimeError, match="boom"):
            convert([_write_zip(tmp_path / "tiny_export.zip")], out, athena_dir)

        assert (out / "_work" / "batches" / tbl.tnx_patient).exists()


class TestBatchAtomicity:
    @pytest.fixture()
    def stage_inputs(self, tmp_path: Path, athena_dir: Path):
        result = ingest.ingest(
            _two_source_zips(tmp_path), tmp_path / "work", DICTIONARY, batch_rows=1
        )
        return result, build_vocab_lookup(result, athena_dir)

    def test_crash_before_commit_leaves_no_finished_batch(
        self, stage_inputs, monkeypatch: pytest.MonkeyPatch
    ):
        result, lookup = stage_inputs

        def crash(*args, **kwargs):
            raise RuntimeError("crash")

        monkeypatch.setattr("tnx_omop.batch_transform._commit_batch", crash)
        with pytest.raises(RuntimeError, match="crash"):
            transform_batch(result, 0, lookup, IdCounters())

        assert not list((result.work_dir / "parts").rglob("*.parquet"))

    def test_rerun_after_crash_replaces_leftovers(self, stage_inputs):
        result, lookup = stage_inputs
        counters, _ = transform_batch(result, 0, lookup, IdCounters())
        leftover = result.work_dir / "parts" / tbl.omop_person / "00001.parquet"
        leftover.parent.mkdir(parents=True, exist_ok=True)
        leftover.write_bytes(b"partial")
        (result.work_dir / "scratch" / "00001").mkdir(parents=True)

        transform_batch(result, 1, lookup, counters)

        assert pl.read_parquet(leftover).height == 1
        assert not (result.work_dir / "scratch" / "00001").exists()

    def test_counters_continue_across_batches(self, stage_inputs):
        result, lookup = stage_inputs
        first, first_counts = transform_batch(result, 0, lookup, IdCounters())
        second, second_counts = transform_batch(result, 1, lookup, first)

        assert first.last_visit_id == 2
        assert (
            first.record_counts[tbl.omop_condition_occurrence]
            == first_counts[tbl.omop_condition_occurrence]
        )
        assert second.last_visit_id == 4
        assert second.record_counts[tbl.omop_condition_occurrence] == (
            first_counts[tbl.omop_condition_occurrence]
            + second_counts[tbl.omop_condition_occurrence]
        )


class TestMainValidation:
    def run_main(self, monkeypatch, tmp_path: Path, vocab_dir: Path) -> int:
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        argv = [
            "tnx-omop",
            "convert",
            "-i",
            str(zip_path),
            "-o",
            str(tmp_path / "omop"),
        ]
        monkeypatch.setattr(sys, "argv", argv + ["--vocab-dir", str(vocab_dir)])
        return main.main()

    def test_missing_vocab_dir(self, monkeypatch, tmp_path: Path, caplog):
        assert self.run_main(monkeypatch, tmp_path, tmp_path / "nowhere") == 1
        assert "not found" in caplog.text
        assert not (tmp_path / "omop").exists()

    def test_incomplete_vocab_dir(self, monkeypatch, tmp_path: Path, caplog):
        vocab_dir = tmp_path / "vocab"
        vocab_dir.mkdir()
        (vocab_dir / f"{tbl.athena_concept}.csv").write_text("x\n")

        assert self.run_main(monkeypatch, tmp_path, vocab_dir) == 1
        assert f"{tbl.athena_concept_relationship}.csv" in caplog.text
        assert f"{tbl.athena_vocabulary}.csv" in caplog.text

    def test_batch_rows_must_be_positive(self, monkeypatch, tmp_path: Path):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        argv = [
            "tnx-omop",
            "convert",
            "-i",
            str(zip_path),
            "-o",
            str(tmp_path / "omop"),
        ]
        argv += ["--vocab-dir", str(tmp_path), "--batch-rows", "0"]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            main.main()

    def test_work_dir_options_reach_convert(
        self, monkeypatch, tmp_path: Path, athena_dir: Path
    ):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        work = tmp_path / "work"
        argv = [
            "tnx-omop",
            "convert",
            "-i",
            str(zip_path),
            "-o",
            str(tmp_path / "omop"),
        ]
        argv += ["--vocab-dir", str(athena_dir), "--work-dir", str(work)]
        argv += ["--batch-rows", "1", "--keep-work-dir"]
        monkeypatch.setattr(sys, "argv", argv)

        assert main.main() == 0
        assert (work / "parts" / "coverage" / "00001.parquet").exists()
