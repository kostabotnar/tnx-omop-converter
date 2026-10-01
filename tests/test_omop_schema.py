"""Tests for tnx_omop/schema/omop.py and the conform helper."""

import polars as pl
import pytest

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.schema.omop import (
    DEVICE_EXPOSURE_SCHEMA,
    CONDITION_ERA_SCHEMA,
    DRUG_ERA_SCHEMA,
    DRUG_EXPOSURE_SCHEMA,
    ERA_SCHEMAS,
    OBSERVATION_SCHEMA,
    OMOP_SCHEMAS,
    VOCABULARY_SCHEMAS,
)
from tnx_omop.transformers.base import conform


class TestSchemas:
    def test_new_tables_registered(self):
        assert OMOP_SCHEMAS[tbl.omop_observation] is OBSERVATION_SCHEMA
        assert OMOP_SCHEMAS[tbl.omop_device_exposure] is DEVICE_EXPOSURE_SCHEMA

    def test_observation_columns(self):
        assert len(OBSERVATION_SCHEMA.columns) == 21
        assert OBSERVATION_SCHEMA.primary_key == col.observation_id
        assert set(OBSERVATION_SCHEMA.required_columns) >= {
            col.person_id,
            col.observation_concept_id,
            col.observation_date,
            col.observation_type_concept_id,
        }

    def test_device_exposure_columns(self):
        assert len(DEVICE_EXPOSURE_SCHEMA.columns) == 19
        assert DEVICE_EXPOSURE_SCHEMA.primary_key == col.device_exposure_id
        assert DEVICE_EXPOSURE_SCHEMA.dtype_map[col.quantity] == pl.Int64
        assert col.device_exposure_end_date not in (
            DEVICE_EXPOSURE_SCHEMA.required_columns
        )

    def test_drug_exposure_end_date_required(self):
        assert col.drug_exposure_end_date in DRUG_EXPOSURE_SCHEMA.required_columns

    @pytest.mark.parametrize("schema", list(OMOP_SCHEMAS.values()))
    def test_column_names_unique(self, schema):
        assert len(schema.column_names) == len(set(schema.column_names))


class TestVocabularySchemas:
    def test_not_required_tables(self):
        assert not set(VOCABULARY_SCHEMAS) & set(OMOP_SCHEMAS)
        assert all(name == s.name for name, s in VOCABULARY_SCHEMAS.items())

    @pytest.mark.parametrize(
        ("table", "key"),
        [
            (tbl.omop_vocabulary, col.vocabulary_id),
            (tbl.omop_domain, col.domain_id),
            (tbl.omop_concept_class, col.concept_class_id),
            (tbl.omop_relationship, col.relationship_id),
            (tbl.omop_concept_relationship, None),
            (tbl.omop_concept_ancestor, None),
            (tbl.omop_concept_synonym, None),
            (tbl.omop_drug_strength, None),
        ],
    )
    def test_primary_keys(self, table: str, key: str | None):
        assert VOCABULARY_SCHEMAS[table].primary_key == key

    def test_drug_strength_numeric_types(self):
        types = VOCABULARY_SCHEMAS[tbl.omop_drug_strength].dtype_map
        assert types[col.amount_value] == pl.Float64
        assert types[col.box_size] == pl.Int64
        assert types[col.valid_start_date] == pl.Date

    @pytest.mark.parametrize("schema", list(VOCABULARY_SCHEMAS.values()))
    def test_column_names_unique(self, schema):
        assert len(schema.column_names) == len(set(schema.column_names))


class TestConform:
    def test_selects_orders_and_casts(self):
        values = {
            name: [None] for name in reversed(DEVICE_EXPOSURE_SCHEMA.column_names)
        }
        values[col.device_exposure_id] = [1]
        values["extra"] = ["dropped"]
        result = conform(pl.DataFrame(values), DEVICE_EXPOSURE_SCHEMA)

        assert result.columns == DEVICE_EXPOSURE_SCHEMA.column_names
        assert result.schema[col.device_exposure_start_date] == pl.Date
        assert result.schema[col.device_source_value] == pl.Utf8
        assert result[col.device_exposure_id].to_list() == [1]

    def test_empty_frame(self):
        frame = pl.DataFrame(
            schema={name: pl.Utf8 for name in OBSERVATION_SCHEMA.column_names}
        )
        result = conform(frame, OBSERVATION_SCHEMA)
        assert result.is_empty()
        assert result.schema[col.observation_id] == pl.Int64

    def test_missing_column_raises(self):
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            conform(pl.DataFrame({col.observation_id: [1]}), OBSERVATION_SCHEMA)


class TestEraSchemas:
    def test_not_required_tables(self):
        assert not set(ERA_SCHEMAS) & set(OMOP_SCHEMAS)
        assert all(name == s.name for name, s in ERA_SCHEMAS.items())

    def test_condition_era_columns(self):
        assert CONDITION_ERA_SCHEMA.column_names == [
            "condition_era_id",
            "person_id",
            "condition_concept_id",
            "condition_era_start_date",
            "condition_era_end_date",
            "condition_occurrence_count",
        ]
        assert CONDITION_ERA_SCHEMA.primary_key == col.condition_era_id
        assert CONDITION_ERA_SCHEMA.dtype_map[col.condition_era_end_date] == pl.Date
        assert col.condition_occurrence_count not in (
            CONDITION_ERA_SCHEMA.required_columns
        )

    def test_drug_era_columns(self):
        assert DRUG_ERA_SCHEMA.column_names == [
            "drug_era_id",
            "person_id",
            "drug_concept_id",
            "drug_era_start_date",
            "drug_era_end_date",
            "drug_exposure_count",
            "gap_days",
        ]
        assert DRUG_ERA_SCHEMA.primary_key == col.drug_era_id
        assert DRUG_ERA_SCHEMA.dtype_map[col.gap_days] == pl.Int64
