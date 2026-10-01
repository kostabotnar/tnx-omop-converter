"""Tests for tnx_omop/omop_vocab/vocabulary_export.py and `convert --export-vocabulary`."""

import json
import logging
import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tnx_omop import __main__ as entry
from tnx_omop.converter import STAGE_VOCABULARY_EXPORT, convert
from tnx_omop.schema.omop import CONCEPT_SCHEMA, OMOP_SCHEMAS, VOCABULARY_SCHEMAS
from tnx_omop.omop_vocab import vocabulary_export
from tnx_omop.omop_vocab.athena import scan_concepts, unique_concepts
from tnx_omop.omop_vocab.concept_table import table_path
from tnx_omop.run_report import REPORT_FILE
from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.quality.validation import validate
from tests import athena_fixture as fx
from tests.schema_asserts import assert_conforms
from tests.test_converter import _write_zip

OPTIONAL_TABLES = [
    tbl.omop_concept_ancestor,
    tbl.omop_domain,
    tbl.omop_concept_class,
    tbl.omop_relationship,
    tbl.omop_concept_synonym,
    tbl.omop_drug_strength,
]


def read(output_dir: Path, table_name: str) -> pl.DataFrame:
    return pl.read_parquet(table_path(output_dir, table_name))


@pytest.fixture(scope="module")
def outputs(
    tmp_path_factory: pytest.TempPathFactory, athena_dir: Path
) -> tuple[Path, Path, dict[str, int]]:
    """The same conversion without and with the export: (reduced, full, row counts)."""
    root = tmp_path_factory.mktemp("vocabulary_export")
    zip_path = _write_zip(root / "tiny_export.zip")
    convert(
        [zip_path], root / "reduced", athena_dir, export_vocabulary=False, eras=False
    )
    counts = convert(
        [zip_path], root / "full", athena_dir, export_vocabulary=True, eras=False
    )
    return root / "reduced", root / "full", counts


class TestExportedTables:
    @pytest.mark.parametrize("table_name", list(VOCABULARY_SCHEMAS))
    def test_table_has_the_schema(self, outputs, table_name: str):
        assert_conforms(read(outputs[1], table_name), VOCABULARY_SCHEMAS[table_name])

    def test_concept_has_the_schema(self, outputs):
        assert_conforms(read(outputs[1], tbl.omop_concept), CONCEPT_SCHEMA)

    def test_values_are_typed_from_the_athena_strings(self, outputs):
        strength = read(outputs[1], tbl.omop_drug_strength).sort(col.drug_concept_id)
        first, second = strength.to_dicts()

        assert first[col.drug_concept_id] == fx.RXNORM_197361
        assert first[col.amount_value] == 5.0
        assert first[col.numerator_value] is None
        assert first[col.valid_start_date] == date(1970, 1, 1)
        assert second[col.numerator_value] == 0.5
        assert second[col.box_size] == 10
        relationships = read(outputs[1], tbl.omop_concept_relationship)
        assert relationships[col.valid_end_date].dtype == pl.Date

    def test_row_counts_are_returned_and_match_the_files(self, outputs):
        counts = outputs[2]

        assert set(VOCABULARY_SCHEMAS) <= set(counts)
        for table_name in [*VOCABULARY_SCHEMAS, tbl.omop_concept]:
            assert counts[table_name] == read(outputs[1], table_name).height
        assert counts[tbl.omop_concept] > read(outputs[0], tbl.omop_concept).height

    def test_all_athena_rows_are_present(self, outputs, athena_dir: Path):
        relationships = read(outputs[1], tbl.omop_concept_relationship)
        source = pl.read_csv(
            athena_dir / f"{tbl.athena_concept_relationship}.csv",
            separator="\t",
            infer_schema=False,
        )

        assert relationships.height == source.height

    def test_no_temporary_files_are_left(self, outputs):
        assert not list(outputs[1].rglob("*.tmp"))


class TestFullConcept:
    def test_reduced_rows_are_unchanged_in_the_full_table(self, outputs):
        reduced = read(outputs[0], tbl.omop_concept)
        full = read(outputs[1], tbl.omop_concept)

        assert reduced.height > 0
        # null standard_concept and invalid_reason must match each other
        missing = reduced.join(full, on=reduced.columns, how="anti", nulls_equal=True)
        assert missing.height == 0
        assert (
            full.filter(pl.col(col.concept_id).is_in(reduced[col.concept_id].implode()))
            .sort(col.concept_id)
            .equals(reduced.sort(col.concept_id))
        )

    def test_full_table_holds_every_athena_concept_once(
        self, outputs, athena_dir: Path
    ):
        full = read(outputs[1], tbl.omop_concept)
        reduced_ids = set(read(outputs[0], tbl.omop_concept)[col.concept_id])
        expected = (
            unique_concepts(scan_concepts(athena_dir))
            .select(pl.col(col.concept_id).cast(pl.Int64))
            .collect()
        )

        assert full[col.concept_id].is_unique().all()
        assert set(full[col.concept_id]) == set(expected[col.concept_id])
        # a concept that no output table references is only in the full table
        assert fx.ICD10_C79_51 not in reduced_ids
        assert fx.ICD10_C79_51 in set(full[col.concept_id])

    def test_empty_cpt4_names_are_filled_from_the_terminology(self, outputs):
        names = dict(
            read(outputs[1], tbl.omop_concept)
            .select(col.concept_id, col.concept_name)
            .iter_rows()
        )

        assert names[fx.CPT4_80053] == "Comprehensive metabolic panel"
        assert names[fx.CPT4_99213] == "Office visit, established patient, low"
        # the name in CONCEPT.csv wins over the empty one in CONCEPT_CPT4.csv
        assert names[fx.CPT4_99214] == "Office visit, established patient"
        assert "" not in names.values()


class TestOutputValidity:
    def test_validate_passes_with_the_export(self, outputs):
        report = validate(outputs[1])

        assert report.ok, "\n".join(str(i) for r in report.failed for i in r.issues)
        assert not report.skipped
        names = {r.name for r in report.results}
        assert {f"{t} schema" for t in VOCABULARY_SCHEMAS} <= names

    def test_validate_passes_without_the_export(self, outputs):
        report = validate(outputs[0])

        assert report.ok
        assert not {f"{t} schema" for t in VOCABULARY_SCHEMAS} & {
            r.name for r in report.results
        }

    def test_without_the_export_only_the_clinical_tables_and_concept_are_written(
        self, outputs
    ):
        written = {p.name for p in outputs[0].iterdir() if p.is_dir()}

        assert written == {t.lower() for t in OMOP_SCHEMAS} | {tbl.excluded}

    def test_a_run_without_the_export_removes_an_earlier_export(
        self, tmp_path: Path, athena_dir: Path
    ):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        output_dir = tmp_path / "out"
        convert([zip_path], output_dir, athena_dir, export_vocabulary=True)
        convert([zip_path], output_dir, athena_dir, export_vocabulary=False, eras=False)

        written = {p.name for p in output_dir.iterdir() if p.is_dir()}
        assert written == {t.lower() for t in OMOP_SCHEMAS} | {tbl.excluded}
        assert validate(output_dir).ok

    def test_validate_reports_a_broken_exported_table(self, outputs, tmp_path: Path):
        damaged = tmp_path / "damaged"
        shutil.copytree(outputs[1], damaged)
        path = table_path(damaged, tbl.omop_domain)
        broken = pl.read_parquet(path).with_columns(
            pl.lit("Condition").alias(col.domain_id)
        )
        broken.write_parquet(path)

        report = validate(damaged)

        assert [r.name for r in report.failed] == [f"{tbl.omop_domain} schema"]


class TestRunReport:
    def test_stage_and_row_counts_are_in_the_report(self, outputs):
        report = json.loads((outputs[1] / REPORT_FILE).read_text(encoding="utf-8"))

        assert STAGE_VOCABULARY_EXPORT in report["stages"]
        assert report["row_counts"][tbl.omop_concept_ancestor] == 5
        assert report["row_counts"][tbl.omop_concept] == (
            read(outputs[1], tbl.omop_concept).height
        )

    def test_no_stage_without_the_export(self, outputs):
        report = json.loads((outputs[0] / REPORT_FILE).read_text(encoding="utf-8"))

        assert STAGE_VOCABULARY_EXPORT not in report["stages"]
        assert not set(VOCABULARY_SCHEMAS) & set(report["row_counts"])


class TestMissingFiles:
    def test_missing_optional_files_are_skipped(self, tmp_path: Path, caplog):
        vocab = fx.write_athena_fixture(tmp_path / "athena", optional_tables=False)
        out = tmp_path / "out"

        with caplog.at_level(logging.INFO):
            counts = vocabulary_export.export_vocabulary(out, vocab)

        assert set(counts) == {
            tbl.omop_concept,
            tbl.omop_concept_relationship,
            tbl.omop_vocabulary,
        }
        for table_name in OPTIONAL_TABLES:
            assert not table_path(out, table_name).exists()
        assert "DRUG_STRENGTH.csv not found" in caplog.text
        assert all(r.levelno == logging.INFO for r in caplog.records)

    def test_some_optional_files_present(self, tmp_path: Path):
        vocab = fx.write_athena_fixture(tmp_path / "athena")
        (vocab / f"{tbl.athena_domain}.csv").unlink()

        counts = vocabulary_export.export_vocabulary(tmp_path / "out", vocab)

        assert tbl.omop_domain not in counts
        assert counts[tbl.omop_concept_class] == 1

    @pytest.mark.parametrize(
        "stem",
        [tbl.athena_concept, tbl.athena_concept_relationship, tbl.athena_vocabulary],
    )
    def test_missing_required_file_raises(self, tmp_path: Path, stem: str):
        vocab = fx.write_athena_fixture(tmp_path / "athena")
        (vocab / f"{stem}.csv").unlink()

        with pytest.raises(FileNotFoundError, match=stem):
            vocabulary_export.export_vocabulary(tmp_path / "out", vocab)

    def test_failed_export_keeps_the_existing_file(self, tmp_path: Path):
        vocab = fx.write_athena_fixture(tmp_path / "athena")
        out = tmp_path / "out"
        vocabulary_export.export_vocabulary(out, vocab)
        before = read(out, tbl.omop_domain)
        (vocab / f"{tbl.athena_domain}.csv").write_text(
            "domain_id\tdomain_name\tdomain_concept_id\nX\tX\tnot a number\n"
        )

        with pytest.raises(pl.exceptions.PolarsError):
            vocabulary_export.export_vocabulary(out, vocab)

        assert read(out, tbl.omop_domain).equals(before)
        assert not list(out.rglob("*.tmp"))


class TestCommandLine:
    def test_flag_exports_the_vocabulary(self, tmp_path: Path, athena_dir: Path):
        out = tmp_path / "omop"

        code = entry.main(
            ["convert", "-i", str(_write_zip(tmp_path / "tiny_export.zip"))]
            + ["-o", str(out), "--vocab-dir", str(athena_dir), "--export-vocabulary"]
        )

        assert code == 0
        assert table_path(out, tbl.omop_concept_ancestor).exists()
