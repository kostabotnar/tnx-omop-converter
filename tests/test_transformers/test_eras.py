"""Tests for transformers/eras.py with small hand-made frames (no files).

The expected eras are worked out in each test from the dates it passes.
"""

import pytest

from tests.era_frames import (
    CONDITION,
    OTHER_CONDITION,
    conditions,
    day,
    exposures,
    ingredient_pairs,
)
from tnx_omop.transformers import eras
from tnx_omop.util import columns as col


class TestConditionEras:
    def test_gap_of_30_days_merges_and_31_days_does_not(self):
        # Occurrence ends on day 10. Day 40 is 30 days later, day 71 is 31 days after
        # the end (day 40) of the merged era.
        result = eras.condition_eras(
            conditions(
                [
                    (1, CONDITION, day(0), day(10)),
                    (1, CONDITION, day(40), day(45)),
                    (1, CONDITION, day(76), day(80)),
                ]
            )
        )

        assert result.rows() == [
            (1, CONDITION, day(0), day(45), 2),
            (1, CONDITION, day(76), day(80), 1),
        ]

    def test_gap_of_31_days_does_not_merge(self):
        result = eras.condition_eras(
            conditions(
                [(1, CONDITION, day(0), day(10)), (1, CONDITION, day(41), day(42))]
            )
        )

        assert result.rows() == [
            (1, CONDITION, day(0), day(10), 1),
            (1, CONDITION, day(41), day(42), 1),
        ]

    def test_gap_is_measured_from_the_latest_end_so_far(self):
        # The third occurrence is 24 days after the end of the first (day 60) but 78
        # days after the end of the second.
        result = eras.condition_eras(
            conditions(
                [
                    (1, CONDITION, day(0), day(60)),
                    (1, CONDITION, day(10), day(12)),
                    (1, CONDITION, day(84), day(85)),
                ]
            )
        )

        assert result.rows() == [(1, CONDITION, day(0), day(85), 3)]

    def test_null_end_date_is_start_plus_one_day(self):
        result = eras.condition_eras(
            conditions(
                [
                    (1, CONDITION, day(0), None),
                    (1, OTHER_CONDITION, day(5), None),
                ]
            )
        )

        assert result.rows() == [
            (1, OTHER_CONDITION, day(5), day(6), 1),
            (1, CONDITION, day(0), day(1), 1),
        ]

    def test_a_null_end_uses_start_plus_one_when_merging(self):
        # Start day 0 without end ends on day 1; day 31 is 30 days later.
        result = eras.condition_eras(
            conditions([(1, CONDITION, day(0), None), (1, CONDITION, day(31), None)])
        )

        assert result.rows() == [(1, CONDITION, day(0), day(32), 2)]

    def test_concept_zero_is_skipped(self):
        result = eras.condition_eras(
            conditions([(1, 0, day(0), day(1)), (1, CONDITION, day(0), day(1))])
        )

        assert result[col.condition_concept_id].to_list() == [CONDITION]

    def test_persons_and_concepts_are_kept_apart(self):
        result = eras.condition_eras(
            conditions(
                [
                    (2, CONDITION, day(0), day(1)),
                    (1, OTHER_CONDITION, day(0), day(1)),
                    (1, CONDITION, day(2), day(3)),
                ]
            )
        )

        assert result.rows() == [
            (1, OTHER_CONDITION, day(0), day(1), 1),
            (1, CONDITION, day(2), day(3), 1),
            (2, CONDITION, day(0), day(1), 1),
        ]

    def test_end_before_start_is_moved_to_the_start(self):
        result = eras.condition_eras(conditions([(1, CONDITION, day(5), day(2))]))

        assert result.rows() == [(1, CONDITION, day(5), day(5), 1)]

    def test_no_rows(self):
        assert eras.condition_eras(conditions([])).height == 0


class TestDrugEras:
    ONE = ingredient_pairs((10, 1))

    def test_overlapping_exposures_form_one_sub_exposure(self):
        result = eras.drug_eras(
            exposures([(1, 10, day(0), day(10), None), (1, 10, day(5), day(20), None)]),
            self.ONE,
        )

        # Days 0 to 20 are covered without a gap: length 20, covered 20
        assert result.rows() == [(1, 1, day(0), day(20), 2, 0)]

    def test_touching_exposures_are_one_sub_exposure(self):
        result = eras.drug_eras(
            exposures(
                [(1, 10, day(0), day(10), None), (1, 10, day(10), day(15), None)]
            ),
            self.ONE,
        )

        assert result.rows() == [(1, 1, day(0), day(15), 2, 0)]

    def test_gap_days_counts_the_uncovered_days_of_the_era(self):
        # Covered: days 0 to 10 and 20 to 30, so 10 + 10 of the 30 days of the era
        result = eras.drug_eras(
            exposures(
                [(1, 10, day(0), day(10), None), (1, 10, day(20), day(30), None)]
            ),
            self.ONE,
        )

        assert result.rows() == [(1, 1, day(0), day(30), 2, 10)]

    def test_gap_days_does_not_count_overlaps_twice(self):
        # Sub-exposure days 0 to 12 (overlap 8 to 10), gap 8 days, then days 20 to 25
        result = eras.drug_eras(
            exposures(
                [
                    (1, 10, day(0), day(10), None),
                    (1, 10, day(8), day(12), None),
                    (1, 10, day(20), day(25), None),
                ]
            ),
            self.ONE,
        )

        assert result.rows() == [(1, 1, day(0), day(25), 3, 25 - 12 - 5)]

    def test_gap_of_30_days_merges_and_31_days_does_not(self):
        merged = eras.drug_eras(
            exposures(
                [(1, 10, day(0), day(10), None), (1, 10, day(40), day(50), None)]
            ),
            self.ONE,
        )
        split = eras.drug_eras(
            exposures(
                [(1, 10, day(0), day(10), None), (1, 10, day(41), day(50), None)]
            ),
            self.ONE,
        )

        assert merged.rows() == [(1, 1, day(0), day(50), 2, 30)]
        assert split.rows() == [
            (1, 1, day(0), day(10), 1, 0),
            (1, 1, day(41), day(50), 1, 0),
        ]

    def test_gap_is_measured_from_the_latest_sub_exposure_end(self):
        # A long first exposure keeps the era open for the third one
        result = eras.drug_eras(
            exposures(
                [
                    (1, 10, day(0), day(100), None),
                    (1, 10, day(120), day(125), None),
                    (1, 10, day(150), day(151), None),
                ]
            ),
            self.ONE,
        )

        assert result.rows() == [(1, 1, day(0), day(151), 3, 151 - 100 - 5 - 1)]

    def test_end_date_days_supply_and_one_day_fallbacks(self):
        # End date wins over days_supply; days_supply is used without an end date;
        # one day when both are missing. The three are 35 days apart, so they stay
        # separate eras of three persons.
        result = eras.drug_eras(
            exposures(
                [
                    (1, 10, day(0), day(3), 30),
                    (2, 10, day(0), None, 7),
                    (3, 10, day(0), None, None),
                ]
            ),
            self.ONE,
        )

        assert result.rows() == [
            (1, 1, day(0), day(3), 1, 0),
            (2, 1, day(0), day(7), 1, 0),
            (3, 1, day(0), day(1), 1, 0),
        ]

    def test_days_supply_end_takes_part_in_merging(self):
        # First exposure ends on day 7 (0 + 7), the second starts 30 days later
        result = eras.drug_eras(
            exposures([(1, 10, day(0), None, 7), (1, 10, day(37), day(40), None)]),
            self.ONE,
        )

        assert result.rows() == [(1, 1, day(0), day(40), 2, 30)]

    def test_clinical_drug_with_two_ingredients_makes_two_eras(self):
        result = eras.drug_eras(
            exposures([(1, 10, day(0), day(5), None)]),
            ingredient_pairs((10, 1), (10, 2)),
        )

        assert result.rows() == [
            (1, 1, day(0), day(5), 1, 0),
            (1, 2, day(0), day(5), 1, 0),
        ]

    def test_different_drugs_of_one_ingredient_share_an_era(self):
        result = eras.drug_eras(
            exposures([(1, 10, day(0), day(5), None), (1, 11, day(6), day(9), None)]),
            ingredient_pairs((10, 1), (11, 1)),
        )

        # Days 0 to 5 and 6 to 9 are covered: 8 of the 9 days of the era
        assert result.rows() == [(1, 1, day(0), day(9), 2, 1)]

    def test_drug_without_ingredient_and_concept_zero_are_skipped(self):
        result = eras.drug_eras(
            exposures(
                [
                    (1, 10, day(0), day(5), None),
                    (1, 99, day(0), day(5), None),
                    (1, 0, day(0), day(5), None),
                ]
            ),
            self.ONE,
        )

        assert result.rows() == [(1, 1, day(0), day(5), 1, 0)]

    def test_same_day_exposures_merge(self):
        result = eras.drug_eras(
            exposures([(1, 10, day(3), day(3), None), (1, 10, day(3), day(3), None)]),
            self.ONE,
        )

        assert result.rows() == [(1, 1, day(3), day(3), 2, 0)]

    def test_end_before_start_is_moved_to_the_start(self):
        result = eras.drug_eras(exposures([(1, 10, day(5), day(2), None)]), self.ONE)

        assert result.rows() == [(1, 1, day(5), day(5), 1, 0)]

    def test_no_rows(self):
        assert eras.drug_eras(exposures([]), self.ONE).height == 0


def reference_eras(intervals: list[tuple[int, int]], gap: int) -> list[tuple]:
    """Plain loop over (start, end) day pairs: (start, end, count, covered days)."""
    result: list[list[int]] = []
    latest_end = 0
    for start, end in sorted(intervals):
        if result and start - latest_end <= gap:
            result[-1][1] = max(result[-1][1], end)
            result[-1][2] += 1
            result[-1][3] += end - start
            latest_end = max(latest_end, end)
        else:
            result.append([start, end, 1, end - start])
            latest_end = end
    return [tuple(r) for r in result]


class TestAgainstPlainLoop:
    """Random intervals give the same eras as a plain loop over each person."""

    @pytest.fixture
    def rows(self) -> list[tuple[int, int, int, int]]:
        import random

        rng = random.Random(7)
        rows = []
        for _ in range(600):
            start = rng.randint(0, 400)
            rows.append(
                (
                    rng.randint(1, 6),
                    rng.randint(1, 3),
                    start,
                    start + rng.randint(0, 25),
                )
            )
        return rows

    def test_condition_eras(self, rows):
        frame = conditions([(p, c, day(s), day(e)) for p, c, s, e in rows])
        expected = []
        for person in sorted({r[0] for r in rows}):
            for concept in sorted({r[1] for r in rows if r[0] == person}):
                pairs = [(s, e) for p, c, s, e in rows if (p, c) == (person, concept)]
                expected += [
                    (person, concept, day(s), day(e), n)
                    for s, e, n, _ in reference_eras(pairs, 30)
                ]

        assert eras.condition_eras(frame).rows() == expected

    def test_drug_eras(self, rows):
        frame = exposures([(p, c, day(s), day(e), None) for p, c, s, e in rows])
        ingredients = ingredient_pairs((1, 1), (2, 1), (3, 2))
        expected = []
        for person in sorted({r[0] for r in rows}):
            for ingredient, drugs in ((1, {1, 2}), (2, {3})):
                pairs = [(s, e) for p, c, s, e in rows if p == person and c in drugs]
                subs = reference_eras(pairs, 0)
                for s, e, n, covered in reference_eras(
                    [(s, e) for s, e, _, _ in subs], 30
                ):
                    count = sum(x[2] for x in subs if s <= x[0] and x[1] <= e)
                    days = sum(x[1] - x[0] for x in subs if s <= x[0] and x[1] <= e)
                    expected.append(
                        (person, ingredient, day(s), day(e), count, e - s - days)
                    )

        assert eras.drug_eras(frame, ingredients).rows() == expected
