"""Tests for tnx_omop/util/id_generator.py."""

import polars as pl

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.util.id_generator import add_visit_ids, build_visit_map, with_record_ids


def encounters(rows):
    return pl.DataFrame(
        rows, schema=[col.person_id, col.encounter_id], orient="row"
    ).with_columns(pl.col(col.person_id).cast(pl.Int64))


class TestBuildVisitMap:
    def test_numbers_sorted_distinct_pairs_from_offset(self):
        df = encounters([(2, "E1"), (1, "E9"), (1, "E2"), (1, "E2")])
        result = build_visit_map(df, offset=10)

        assert result.rows() == [(1, "E2", 11), (1, "E9", 12), (2, "E1", 13)]
        assert result.schema[col.visit_occurrence_id] == pl.Int64

    def test_same_encounter_id_of_two_persons_gets_two_ids(self):
        df = encounters([(1, "E1"), (2, "E1")])
        assert build_visit_map(df)[col.visit_occurrence_id].to_list() == [1, 2]

    def test_independent_of_row_order(self):
        df = encounters([(1, "E3"), (1, "E1"), (2, "E2")])
        assert build_visit_map(df).equals(build_visit_map(df.reverse()))

    def test_ignores_null_encounter_ids(self):
        assert build_visit_map(encounters([(1, None), (1, "E1")])).height == 1

    def test_no_encounter_table(self):
        result = build_visit_map(None)
        assert result.height == 0
        assert result.columns == [
            col.person_id,
            col.encounter_id,
            col.visit_occurrence_id,
        ]


def tables():
    return {
        tbl.tnx_encounter: pl.DataFrame(
            {
                col.person_id: [2, 1, 3],
                col.encounter_id: ["E2", "E9", "E2"],
            }
        ),
        tbl.tnx_diagnosis: pl.DataFrame(
            {
                col.person_id: [3, 2, 1],
                col.encounter_id: ["E2", None, "E9"],
            }
        ),
    }


class TestAddVisitIds:
    def test_visit_ids_follow_sorted_person_and_encounter(self):
        result, last_id = add_visit_ids(tables(), offset=5)
        encounter = result[tbl.tnx_encounter]

        assert encounter.select(
            col.person_id, col.encounter_id, col.visit_occurrence_id
        ).sort(col.visit_occurrence_id).rows() == [
            (1, "E9", 6),
            (2, "E2", 7),
            (3, "E2", 8),
        ]
        assert last_id == 8

    def test_ids_consistent_across_tables(self):
        diagnosis = add_visit_ids(tables())[0][tbl.tnx_diagnosis]

        assert diagnosis.select(col.person_id, col.visit_occurrence_id).sort(
            col.person_id
        ).rows() == [(1, 1), (2, None), (3, 3)]

    def test_input_unchanged_and_row_order_kept(self):
        original = tables()
        result, _ = add_visit_ids(original)

        assert col.visit_occurrence_id not in original[tbl.tnx_encounter].columns
        assert (
            result[tbl.tnx_diagnosis][col.person_id].to_list()
            == original[tbl.tnx_diagnosis][col.person_id].to_list()
        )

    def test_without_encounter_table_visit_ids_are_null(self):
        data = tables()
        del data[tbl.tnx_encounter]
        result, last_id = add_visit_ids(data, offset=4)

        assert result[tbl.tnx_diagnosis][col.visit_occurrence_id].null_count() == 3
        assert last_id == 4

    def test_ids_continue_across_batches(self):
        first, last = add_visit_ids(tables())
        second, _ = add_visit_ids(tables(), offset=last)

        assert first[tbl.tnx_encounter][col.visit_occurrence_id].max() == 3
        assert second[tbl.tnx_encounter][col.visit_occurrence_id].min() == 4


class TestWithRecordIds:
    def frame(self):
        return pl.LazyFrame(
            {
                "value": ["b", "a", "c", "a"],
                "rid": [None, None, None, None],
                "person_id": [2, 1, 1, 1],
            },
            schema_overrides={"rid": pl.Int64},
        )

    def test_numbers_sorted_rows_and_keeps_columns(self):
        result = with_record_ids(self.frame(), "rid", ["person_id", "value"]).collect()

        assert result.columns == ["value", "rid", "person_id"]
        assert result.schema["rid"] == pl.Int64
        assert result["rid"].to_list() == [1, 2, 3, 4]
        assert result["value"].to_list() == ["a", "a", "c", "b"]

    def test_ids_continue_after_offset(self):
        result = with_record_ids(
            self.frame(), "rid", ["person_id", "value"], offset=7
        ).collect()
        assert result["rid"].to_list() == [8, 9, 10, 11]

    def test_independent_of_input_order(self):
        keys = ["person_id", "value"]
        a = with_record_ids(self.frame(), "rid", keys).collect()
        b = with_record_ids(self.frame().reverse(), "rid", keys).collect()
        assert a.equals(b)
