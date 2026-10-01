"""Tests for the Athena vocabulary reader."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tnx_omop.omop_vocab import athena
from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tests import athena_fixture as fx


class TestMissingFiles:
    def test_complete_fixture(self, athena_dir: Path):
        assert athena.missing_files(athena_dir) == []

    def test_reports_missing(self, tmp_path: Path):
        (tmp_path / f"{tbl.athena_concept}.csv").write_text("x\n")
        assert athena.missing_files(tmp_path) == [
            f"{tbl.athena_concept_relationship}.csv",
            f"{tbl.athena_vocabulary}.csv",
        ]


class TestScanConcepts:
    def test_all_columns_are_strings(self, athena_dir: Path):
        schema = athena.scan_concepts(athena_dir).collect_schema()
        assert schema.names() == athena.CONCEPT_COLUMNS
        assert all(dtype == pl.String for dtype in schema.dtypes())

    def test_includes_cpt4_file(self, athena_dir: Path):
        ids = (
            athena.scan_concepts(athena_dir)
            .filter(pl.col(col.vocabulary_id) == "CPT4")
            .collect()
            .get_column(col.concept_id)
            .to_list()
        )
        assert str(fx.CPT4_80053) in ids

    def test_works_without_cpt4_file(self, athena_dir: Path, tmp_path: Path):
        for name in athena.REQUIRED_FILES:
            (tmp_path / name).write_bytes((athena_dir / name).read_bytes())
        ids = athena.scan_concepts(tmp_path).collect().get_column(col.concept_id)
        assert str(fx.CPT4_80053) not in ids.to_list()
        assert str(fx.CPT4_99214) in ids.to_list()

    def test_quotes_in_names_are_literal(self, tmp_path: Path):
        path = tmp_path / "CONCEPT.csv"
        header = "\t".join(athena.CONCEPT_COLUMNS)
        path.write_text(
            header
            + '\n1\t5" needle\tDevice\tSNOMED\tPhysical Object\tS\tX1\t19700101\t20991231\t\n'
        )
        name = athena.scan_athena(path).collect().item(0, col.concept_name)
        assert name == '5" needle'


class TestUniqueConcepts:
    def test_prefers_non_empty_name(self, athena_dir: Path):
        rows = (
            athena.unique_concepts(
                athena.scan_concepts(athena_dir).filter(
                    pl.col(col.concept_id) == str(fx.CPT4_99214)
                )
            )
            .collect()
            .get_column(col.concept_name)
            .to_list()
        )
        assert rows == ["Office visit, established patient"]

    def test_one_row_per_concept(self, athena_dir: Path):
        df = athena.unique_concepts(athena.scan_concepts(athena_dir)).collect()
        assert df.get_column(col.concept_id).is_unique().all()


class TestTypedConcepts:
    def test_types(self, athena_dir: Path):
        df = athena.typed_concepts(athena.scan_concepts(athena_dir)).collect()
        assert df.schema[col.concept_id] == pl.Int64
        assert df.schema[col.valid_start_date] == pl.Date
        assert df.schema[col.valid_end_date] == pl.Date
        row = df.filter(pl.col(col.concept_id) == fx.CPT4_80053).row(0, named=True)
        assert row[col.concept_name] == ""
        assert row[col.valid_start_date] == date(1970, 1, 1)
        assert row[col.valid_end_date] == date(2099, 12, 31)


class TestScanMapsTo:
    @pytest.fixture
    def maps_to(self, athena_dir: Path) -> pl.DataFrame:
        return athena.scan_maps_to(athena_dir).collect()

    def test_schema(self, maps_to: pl.DataFrame):
        assert maps_to.schema == pl.Schema(
            {col.concept_id_1: pl.Int64, col.concept_id_2: pl.Int64}
        )

    def test_invalid_maps_to_dropped(self, maps_to: pl.DataFrame):
        targets = maps_to.filter(pl.col(col.concept_id_1) == fx.ICD10_I10)
        assert targets.get_column(col.concept_id_2).to_list() == [fx.COND_HYPERTENSION]

    def test_other_relationships_dropped(self, maps_to: pl.DataFrame):
        assert fx.ICD10_C79_51 not in maps_to.get_column(col.concept_id_1).to_list()
        # "Mapped from" rows start at the target concept.
        assert fx.COND_T2DM not in maps_to.get_column(col.concept_id_1).to_list()


class TestVocabularyVersion:
    def test_reads_none_row(self, athena_dir: Path):
        assert athena.read_vocabulary_version(athena_dir) == fx.VOCABULARY_VERSION

    def test_missing_none_row(self, tmp_path: Path):
        (tmp_path / f"{tbl.athena_vocabulary}.csv").write_text(
            "vocabulary_id\tvocabulary_name\tvocabulary_reference\t"
            "vocabulary_version\tvocabulary_concept_id\n"
            "ICD10CM\tICD10CM\t\t2026\t1\n"
        )
        with pytest.raises(ValueError, match="None"):
            athena.read_vocabulary_version(tmp_path)
