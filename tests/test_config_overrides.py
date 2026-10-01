"""User config file that overrides the bundled concept maps."""

import json
import logging
from pathlib import Path

import polars as pl
import pytest

from tnx_omop import __main__ as entry
from tnx_omop import manifest as mf
from tnx_omop.converter import convert
from tnx_omop.domains import NO_SOURCE_CONCEPT
from tnx_omop.omop_vocab.concept_table import table_path
from tnx_omop.util import columns as col
from tnx_omop.util import concept_mappings as cm
from tnx_omop.util import tables as tbl
from tests.test_converter import _write_zip
from tests.test_resume import _vocab


def _write_config(path: Path, content: object) -> Path:
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def _bundled(key: str) -> dict:
    path = Path(cm.__file__).parents[1] / "config" / "concept_mappings.json"
    return json.loads(path.read_text(encoding="utf-8"))[key]


class TestLoadOverrides:
    def test_valid_file_is_returned(self, tmp_path: Path):
        content = {
            "gender": {"X": 8521},
            "unit_ucum": {"widgets": "[arb'U]"},
            "source_vocabularies": {"diagnosis": {"SNOMED": "SNOMED", "TNX": None}},
        }
        assert cm.load_overrides(_write_config(tmp_path / "c.json", content)) == content

    def test_empty_object_is_valid(self, tmp_path: Path):
        assert cm.load_overrides(_write_config(tmp_path / "c.json", {})) == {}

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ({"genders": {}}, "Unknown config key 'genders'"),
            ({"default_concept_id": 0}, "Unknown config key 'default_concept_id'"),
            ({"gender": [1]}, "'gender' must be a JSON object"),
            ({"gender": {"M": "8507"}}, "'gender', entry 'M': expected integer"),
            ({"race": {"White": True}}, "'race', entry 'White': expected integer"),
            ({"route": {"Oral": 1.5}}, "'route', entry 'Oral': expected integer"),
            ({"unit_ucum": {"mg": 5}}, "'unit_ucum', entry 'mg': expected string"),
            (
                {"source_vocabularies": {"diagnosis": "ICD10CM"}},
                "'source_vocabularies', table 'diagnosis'",
            ),
            (
                {"source_vocabularies": {"diagnosis": {"ICD-10-CM": 3}}},
                "table 'diagnosis', code system 'ICD-10-CM': expected string or null",
            ),
            ([], "must contain a JSON object"),
        ],
    )
    def test_invalid_content_names_the_key(
        self, tmp_path: Path, content: object, message: str
    ):
        with pytest.raises(cm.ConfigError, match=message):
            cm.load_overrides(_write_config(tmp_path / "c.json", content))

    def test_invalid_json(self, tmp_path: Path):
        path = tmp_path / "c.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(cm.ConfigError, match="not valid JSON"):
            cm.load_overrides(path)

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(cm.ConfigError, match="Cannot read config file"):
            cm.load_overrides(tmp_path / "missing.json")


class TestApplyOverrides:
    def test_concept_map_entries_added_and_replaced(self):
        same_object = cm.GENDER_CONCEPT_MAP
        cm.apply_overrides({"gender": {"M": 1, "X": 2}})

        assert cm.GENDER_CONCEPT_MAP is same_object
        assert same_object["M"] == 1
        assert same_object["X"] == 2
        assert same_object["F"] == 8532

    def test_unit_map_merged(self):
        cm.apply_overrides({"unit_ucum": {"widgets": "[arb'U]"}})

        assert cm.UNIT_UCUM_MAP["widgets"] == "[arb'U]"
        assert len(cm.UNIT_UCUM_MAP) > 1

    def test_source_vocabularies_merge_per_table_then_code_system(self):
        cm.apply_overrides(
            {
                "source_vocabularies": {
                    "diagnosis": {"ICD-10-CM": None, "SNOMED": "SNOMED"},
                    "new_table": {"X": "Y"},
                }
            }
        )

        diagnosis = cm.SOURCE_VOCABULARY_MAP["diagnosis"]
        assert diagnosis["ICD-10-CM"] is None
        assert diagnosis["SNOMED"] == "SNOMED"
        assert diagnosis["ICD-9-CM"] == "ICD9CM"
        assert cm.SOURCE_VOCABULARY_MAP["procedure"]["CPT"] == "CPT4"
        assert cm.SOURCE_VOCABULARY_MAP["new_table"] == {"X": "Y"}

    def test_null_keeps_no_vocabulary_meaning(self):
        cm.apply_overrides({"source_vocabularies": {"lab_result": {"TNX": None}}})
        assert cm.SOURCE_VOCABULARY_MAP["lab_result"]["TNX"] is None

    def test_reset_restores_the_bundled_maps(self):
        cm.apply_overrides(
            {
                "gender": {"M": 1, "X": 2},
                "unit_ucum": {"widgets": "x"},
                "source_vocabularies": {"diagnosis": {"ICD-10-CM": None}, "t": {}},
            }
        )

        cm.reset_mappings()

        assert cm.GENDER_CONCEPT_MAP == _bundled("gender")
        assert "widgets" not in cm.UNIT_UCUM_MAP
        assert cm.SOURCE_VOCABULARY_MAP["diagnosis"]["ICD-10-CM"] == "ICD10CM"
        assert "t" not in cm.SOURCE_VOCABULARY_MAP


def _genders(out: Path) -> dict[str, int]:
    person = pl.read_parquet(table_path(out, tbl.omop_person))
    return dict(
        person.select(col.person_source_value, col.gender_concept_id).iter_rows()
    )


class TestConvertWithConfig:
    def test_gender_override_changes_the_output(self, tmp_path: Path, athena_dir: Path):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        config = _write_config(tmp_path / "c.json", {"gender": {"M": 8532}})

        convert([zip_path], tmp_path / "plain", athena_dir)
        convert([zip_path], tmp_path / "changed", athena_dir, config_path=config)

        plain = _genders(tmp_path / "plain")
        changed = _genders(tmp_path / "changed")
        assert plain["P1"] == 8507
        assert changed["P1"] == 8532
        assert changed["P2"] == plain["P2"]

    def test_null_code_system_sends_records_to_excluded(
        self, tmp_path: Path, athena_dir: Path
    ):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        config = _write_config(
            tmp_path / "c.json",
            {"source_vocabularies": {"diagnosis": {"ICD-10-CM": None}}},
        )

        convert([zip_path], tmp_path / "plain", athena_dir)
        convert([zip_path], tmp_path / "out", athena_dir, config_path=config)

        excluded = pl.read_parquet(
            tmp_path / "out" / tbl.excluded / f"{tbl.tnx_diagnosis}.parquet"
        )
        reasons = excluded.filter(pl.col(col.code) == "C79.51")[col.exclusion_reason]
        assert reasons.to_list() == [NO_SOURCE_CONCEPT]
        conditions = {
            name: pl.read_parquet(
                table_path(tmp_path / name, tbl.omop_condition_occurrence)
            )
            for name in ("plain", "out")
        }
        assert len(conditions["out"]) < len(conditions["plain"])

    def test_report_records_the_config(self, tmp_path: Path, athena_dir: Path):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        config = _write_config(tmp_path / "c.json", {"gender": {"M": 8532}})

        convert([zip_path], tmp_path / "omop", athena_dir, config_path=config)

        report = json.loads((tmp_path / "omop" / "run_report.json").read_text("utf-8"))
        assert report["config"] == cm.describe_config(config)
        assert set(report["config"]) == {"path", "sha256"}
        assert len(report["config"]["sha256"]) == 64

    def test_invalid_config_raises_before_any_work(
        self, tmp_path: Path, athena_dir: Path
    ):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        config = _write_config(tmp_path / "c.json", {"bogus": {}})

        with pytest.raises(cm.ConfigError, match="bogus"):
            convert([zip_path], tmp_path / "omop", athena_dir, config_path=config)

        assert not (tmp_path / "omop").exists()

    def test_cli_passes_the_config(self, tmp_path: Path, athena_dir: Path):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        config = _write_config(tmp_path / "c.json", {"gender": {"M": 8532}})
        out = tmp_path / "omop"

        code = entry.main(
            ["convert", "-i", str(zip_path), "-o", str(out)]
            + ["--vocab-dir", str(athena_dir), "--config", str(config)]
        )

        assert code == 0
        assert _genders(out)["P1"] == 8532


class TestFingerprint:
    @pytest.fixture()
    def inputs(self, tmp_path: Path) -> tuple[list[Path], Path]:
        return [_write_zip(tmp_path / "a.zip")], _vocab(tmp_path)

    def test_without_config_is_null(self, inputs):
        zips, vocab = inputs
        assert mf.build_fingerprint(zips, vocab, 10)["config"] is None

    def test_config_adds_path_and_hash(self, inputs, tmp_path: Path):
        zips, vocab = inputs
        config = _write_config(tmp_path / "c.json", {"gender": {"M": 1}})
        fingerprint = mf.build_fingerprint(zips, vocab, 10, config)
        assert fingerprint["config"] == cm.describe_config(config)
        assert fingerprint != mf.build_fingerprint(zips, vocab, 10)

    def test_content_change_changes_the_fingerprint(self, inputs, tmp_path: Path):
        zips, vocab = inputs
        config = _write_config(tmp_path / "c.json", {"gender": {"M": 1}})
        before = mf.build_fingerprint(zips, vocab, 10, config)
        _write_config(config, {"gender": {"M": 2}})
        assert mf.build_fingerprint(zips, vocab, 10, config) != before

    def test_path_change_changes_the_fingerprint(self, inputs, tmp_path: Path):
        zips, vocab = inputs
        first = _write_config(tmp_path / "c1.json", {})
        second = _write_config(tmp_path / "c2.json", {})
        assert mf.build_fingerprint(zips, vocab, 10, first) != mf.build_fingerprint(
            zips, vocab, 10, second
        )

    def test_changed_config_discards_the_work_dir(
        self, tmp_path: Path, athena_dir: Path, caplog: pytest.LogCaptureFixture
    ):
        zip_path = _write_zip(tmp_path / "tiny_export.zip")
        config = _write_config(tmp_path / "c.json", {"gender": {"M": 8507}})
        work = tmp_path / "work"
        caplog.set_level(logging.INFO)

        def run(name: str) -> str:
            caplog.clear()
            convert(
                [zip_path],
                tmp_path / name,
                athena_dir,
                work_dir=work,
                keep_work_dir=True,
                config_path=config,
            )
            return caplog.text

        assert "discarding" not in run("o1")
        assert "discarding" not in run("o2")
        _write_config(config, {"gender": {"M": 8532}})
        assert "discarding" in run("o3")
