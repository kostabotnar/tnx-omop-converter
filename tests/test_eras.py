"""Tests for tnx_omop/eras.py (file access, ingredient map, person ranges).

The era logic itself is tested in tests/test_transformers/test_eras.py.
"""

from pathlib import Path

import polars as pl
import pytest

from tests import athena_fixture as fx
from tests.era_frames import CONDITION, OTHER_CONDITION, conditions, day, exposures
from tests.schema_asserts import assert_conforms
from tnx_omop import eras
from tnx_omop.schema.omop import CONDITION_ERA_SCHEMA, DRUG_ERA_SCHEMA
from tnx_omop.omop_vocab.athena import CONCEPT_COLUMNS
from tnx_omop.omop_vocab.concept_table import table_path
from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl

# Ingredient and drug concepts of the test vocabulary
ASPIRIN = 1191  # RxNorm ingredient
EXT_INGREDIENT = 2001  # RxNorm Extension ingredient
OLD_INGREDIENT = 2002  # RxNorm ingredient that is not standard
ATC_INGREDIENT = 2003  # standard, class Ingredient, but not an RxNorm vocabulary
COMBO_TABLET = 3001  # two ingredients: ASPIRIN and EXT_INGREDIENT
ASPIRIN_TABLET = 3002  # one ingredient: ASPIRIN
UNMAPPED_DRUG = 3003  # no ingredient
ODD_DRUG = 3004  # ingredients that do not count: non-standard and ATC
NO_SELF_ROW = 2004  # RxNorm ingredient without a self row in CONCEPT_ANCESTOR


def write_vocabulary(directory: Path, self_rows: bool = True) -> Path:
    """CONCEPT and CONCEPT_ANCESTOR with the ingredient cases above."""
    directory.mkdir(parents=True, exist_ok=True)
    concepts = [
        (ASPIRIN, "aspirin", "Drug", "RxNorm", "Ingredient", "S", "1191", None),
        (
            EXT_INGREDIENT,
            "ext",
            "Drug",
            "RxNorm Extension",
            "Ingredient",
            "S",
            "X",
            None,
        ),
        (OLD_INGREDIENT, "old", "Drug", "RxNorm", "Ingredient", None, "Y", None),
        (ATC_INGREDIENT, "atc", "Drug", "ATC", "Ingredient", "S", "Z", None),
        (NO_SELF_ROW, "noself", "Drug", "RxNorm", "Ingredient", "S", "W", None),
        (COMBO_TABLET, "combo", "Drug", "RxNorm", "Clinical Drug", "S", "C1", None),
        (ASPIRIN_TABLET, "tablet", "Drug", "RxNorm", "Clinical Drug", "S", "C2", None),
        (UNMAPPED_DRUG, "lone", "Drug", "RxNorm", "Clinical Drug", "S", "C3", None),
        (ODD_DRUG, "odd", "Drug", "RxNorm", "Clinical Drug", "S", "C4", None),
    ]
    fx._write_tsv(
        directory / f"{tbl.athena_concept}.csv",
        CONCEPT_COLUMNS,
        [fx._concept_line(row) for row in concepts],
    )
    pairs = [
        (COMBO_TABLET, ASPIRIN),
        (COMBO_TABLET, EXT_INGREDIENT),
        (ASPIRIN_TABLET, ASPIRIN),
        (ODD_DRUG, OLD_INGREDIENT),
        (ODD_DRUG, ATC_INGREDIENT),
        (ASPIRIN_TABLET, COMBO_TABLET),  # an ancestor that is not an ingredient
    ]
    if self_rows:
        pairs += [(ASPIRIN, ASPIRIN), (EXT_INGREDIENT, EXT_INGREDIENT)]
    fx._write_tsv(
        directory / f"{tbl.athena_concept_ancestor}.csv",
        [
            col.ancestor_concept_id,
            col.descendant_concept_id,
            col.min_levels_of_separation,
            col.max_levels_of_separation,
        ],
        [
            [str(a), str(d), "0" if a == d else "1", "0" if a == d else "1"]
            for d, a in pairs
        ],
    )
    return directory


class TestIngredientMap:
    @pytest.fixture
    def vocab(self, tmp_path: Path) -> Path:
        return write_vocabulary(tmp_path / "vocab")

    def ids(self, *values: int) -> pl.Series:
        return pl.Series(col.drug_concept_id, values, dtype=pl.Int64)

    def test_only_standard_rxnorm_ingredients_count(self, vocab: Path):
        result = eras.ingredient_map(
            vocab, self.ids(COMBO_TABLET, ASPIRIN_TABLET, UNMAPPED_DRUG, ODD_DRUG)
        )

        assert result.rows() == [
            (COMBO_TABLET, ASPIRIN),
            (COMBO_TABLET, EXT_INGREDIENT),
            (ASPIRIN_TABLET, ASPIRIN),
        ]

    def test_an_ingredient_maps_to_itself(self, vocab: Path):
        result = eras.ingredient_map(vocab, self.ids(ASPIRIN, EXT_INGREDIENT))

        assert result.rows() == [(ASPIRIN, ASPIRIN), (EXT_INGREDIENT, EXT_INGREDIENT)]

    def test_self_mapping_does_not_need_the_self_row(self, tmp_path: Path):
        vocab = write_vocabulary(tmp_path / "vocab", self_rows=False)

        result = eras.ingredient_map(
            vocab, self.ids(ASPIRIN, NO_SELF_ROW, OLD_INGREDIENT)
        )

        # The non-standard ingredient is not an ingredient for the eras
        assert result.rows() == [(ASPIRIN, ASPIRIN), (NO_SELF_ROW, NO_SELF_ROW)]

    def test_only_the_given_drugs_are_mapped(self, vocab: Path):
        assert eras.ingredient_map(vocab, self.ids(ASPIRIN_TABLET)).rows() == [
            (ASPIRIN_TABLET, ASPIRIN)
        ]

    def test_no_drugs(self, vocab: Path):
        result = eras.ingredient_map(vocab, self.ids())

        assert result.height == 0
        assert result.columns == [col.drug_concept_id, col.ingredient_concept_id]


class TestPersonRanges:
    def persons(self, *ids: int) -> pl.LazyFrame:
        return pl.LazyFrame({col.person_id: list(ids)})

    def test_one_range_when_everything_fits(self):
        assert eras.person_ranges(self.persons(1, 1, 2, 5), 100) == [(1, 5)]

    def test_ranges_follow_the_row_bound(self):
        # Rows per person: 1 -> 2, 2 -> 1, 3 -> 2, 4 -> 1
        ranges = eras.person_ranges(self.persons(1, 1, 2, 3, 3, 4), 3)

        assert ranges == [(1, 2), (3, 4)]

    def test_a_person_with_more_rows_than_the_bound_has_a_range(self):
        ranges = eras.person_ranges(self.persons(1, 2, 2, 2, 2, 3), 2)

        assert ranges == [(1, 2), (3, 3)]

    def test_no_rows(self):
        assert eras.person_ranges(self.persons(), 10) == []


def write_output(
    output_dir: Path, condition_rows: list[tuple], exposure_rows: list[tuple]
) -> None:
    """Write CONDITION_OCCURRENCE and DRUG_EXPOSURE with the columns eras reads."""
    for name, frame in (
        (tbl.omop_condition_occurrence, conditions(condition_rows)),
        (tbl.omop_drug_exposure, exposures(exposure_rows)),
    ):
        path = table_path(output_dir, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # small row groups, so that a range filter has statistics to skip with
        frame.sort(col.person_id).write_parquet(path, row_group_size=2)


def read(output_dir: Path, name: str) -> pl.DataFrame:
    return pl.read_parquet(table_path(output_dir, name))


CONDITION_ROWS = [
    (1, CONDITION, day(0), day(10)),
    (1, CONDITION, day(40), day(41)),
    (1, CONDITION, day(200), None),
    (1, 0, day(0), day(1)),
    (2, OTHER_CONDITION, day(5), day(6)),
    (2, CONDITION, day(5), day(6)),
    (3, CONDITION, day(0), day(3)),
    (4, CONDITION, day(0), day(3)),
    (4, CONDITION, day(100), day(103)),
]
EXPOSURE_ROWS = [
    (1, COMBO_TABLET, day(0), day(10), None),
    (1, ASPIRIN_TABLET, day(20), day(30), None),
    (2, ASPIRIN, day(0), None, 14),
    (2, UNMAPPED_DRUG, day(0), day(3), None),
    (3, COMBO_TABLET, day(0), day(1), None),
    (3, 0, day(0), day(1), None),
    (4, ODD_DRUG, day(0), day(9), None),
    (5, EXT_INGREDIENT, day(50), day(60), None),
]


class TestBuildEras:
    @pytest.fixture
    def vocab(self, tmp_path: Path) -> Path:
        return write_vocabulary(tmp_path / "vocab")

    @pytest.fixture
    def output(self, tmp_path: Path) -> Path:
        out = tmp_path / "out"
        write_output(out, CONDITION_ROWS, EXPOSURE_ROWS)
        return out

    def test_condition_eras(self, vocab: Path, output: Path, tmp_path: Path):
        counts = eras.build_eras(output, vocab, tmp_path / "work")

        result = read(output, tbl.omop_condition_era)
        assert_conforms(result, CONDITION_ERA_SCHEMA)
        assert result.rows() == [
            (1, 1, CONDITION, day(0), day(41), 2),
            (2, 1, CONDITION, day(200), day(201), 1),
            (3, 2, OTHER_CONDITION, day(5), day(6), 1),
            (4, 2, CONDITION, day(5), day(6), 1),
            (5, 3, CONDITION, day(0), day(3), 1),
            (6, 4, CONDITION, day(0), day(3), 1),
            (7, 4, CONDITION, day(100), day(103), 1),
        ]
        assert counts[tbl.omop_condition_era] == 7

    def test_drug_eras(self, vocab: Path, output: Path, tmp_path: Path):
        counts = eras.build_eras(output, vocab, tmp_path / "work")

        result = read(output, tbl.omop_drug_era)
        assert_conforms(result, DRUG_ERA_SCHEMA)
        # Person 1: aspirin from the combination tablet (days 0 to 10) and from the
        # aspirin tablet (days 20 to 30) is one era with a gap of 10 days; the
        # extension ingredient of the tablet has its own era. Person 2: the
        # ingredient itself for 14 days. Person 3: the combination tablet. Person 4:
        # only ingredients that do not count. Person 5: an ingredient exposure
        # without a person in other tables is still an era.
        assert result.rows() == [
            (1, 1, ASPIRIN, day(0), day(30), 2, 10),
            (2, 1, EXT_INGREDIENT, day(0), day(10), 1, 0),
            (3, 2, ASPIRIN, day(0), day(14), 1, 0),
            (4, 3, ASPIRIN, day(0), day(1), 1, 0),
            (5, 3, EXT_INGREDIENT, day(0), day(1), 1, 0),
            (6, 5, EXT_INGREDIENT, day(50), day(60), 1, 0),
        ]
        assert counts[tbl.omop_drug_era] == 6

    @pytest.mark.parametrize("batch_rows", [1, 2, 3, 5, 1000])
    def test_person_ranges_give_the_same_eras_as_one_range(
        self, vocab: Path, output: Path, tmp_path: Path, batch_rows: int
    ):
        eras.build_eras(output, vocab, tmp_path / "work", batch_rows=10_000)
        expected = {
            name: read(output, name)
            for name in (tbl.omop_condition_era, tbl.omop_drug_era)
        }

        counts = eras.build_eras(
            output, vocab, tmp_path / "work", batch_rows=batch_rows
        )

        for name, frame in expected.items():
            assert read(output, name).equals(frame)
            assert counts[name] == frame.height

    def test_ids_are_sequential_and_rows_are_sorted(
        self, vocab: Path, output: Path, tmp_path: Path
    ):
        eras.build_eras(output, vocab, tmp_path / "work", batch_rows=2)

        for name, id_column, concept in (
            (tbl.omop_condition_era, col.condition_era_id, col.condition_concept_id),
            (tbl.omop_drug_era, col.drug_era_id, col.drug_concept_id),
        ):
            result = read(output, name)
            assert result[id_column].to_list() == list(range(1, result.height + 1))
            assert result.equals(
                result.sort(col.person_id, concept, maintain_order=True)
            )

    def test_parts_are_removed(self, vocab: Path, output: Path, tmp_path: Path):
        work = tmp_path / "work"

        eras.build_eras(output, vocab, work, batch_rows=2)

        assert not (work / eras.ERAS_DIR).exists()

    def test_missing_concept_ancestor_raises(
        self, vocab: Path, output: Path, tmp_path: Path
    ):
        (vocab / f"{tbl.athena_concept_ancestor}.csv").unlink()

        with pytest.raises(FileNotFoundError, match="CONCEPT_ANCESTOR"):
            eras.build_eras(output, vocab, tmp_path / "work")

        assert not table_path(output, tbl.omop_drug_era).exists()

    def test_empty_source_tables_give_empty_era_tables(
        self, vocab: Path, tmp_path: Path
    ):
        out = tmp_path / "out"
        write_output(out, [], [])

        counts = eras.build_eras(out, vocab, tmp_path / "work")

        assert counts == {tbl.omop_condition_era: 0, tbl.omop_drug_era: 0}
        assert_conforms(read(out, tbl.omop_condition_era), CONDITION_ERA_SCHEMA)
        assert_conforms(read(out, tbl.omop_drug_era), DRUG_ERA_SCHEMA)

    def test_missing_source_tables_give_empty_era_tables(
        self, vocab: Path, tmp_path: Path
    ):
        counts = eras.build_eras(tmp_path / "out", vocab, tmp_path / "work")

        assert counts == {tbl.omop_condition_era: 0, tbl.omop_drug_era: 0}

    def test_an_old_era_file_is_replaced(
        self, vocab: Path, output: Path, tmp_path: Path
    ):
        eras.build_eras(output, vocab, tmp_path / "work")
        write_output(output, CONDITION_ROWS[:1], [])

        counts = eras.build_eras(output, vocab, tmp_path / "work")

        assert counts == {tbl.omop_condition_era: 1, tbl.omop_drug_era: 0}
        assert read(output, tbl.omop_condition_era).height == 1
