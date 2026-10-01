"""Tests for tnx_omop/cleaning.py."""

import io
from datetime import date

import polars as pl

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.cleaning import _fix_encounter_dates, clean_batch
from tnx_omop.tnx_schema import apply_data_types, drop_derived_columns
from tests.tnx_dictionary import DICTIONARY

PATIENT = "patient_id,sex,year_of_birth,month_year_death\n"
ENCOUNTER = "encounter_id,patient_id,start_date,end_date,type\n"
DIAGNOSIS = "patient_id,encounter_id,code_system,code,date\n"
QUOTED_DIAGNOSIS = '"patient_id","encounter_id","code_system","code","date"\n'


def batch(*sources: dict[str, str]) -> dict[str, pl.DataFrame]:
    """Tables of one batch as ingest writes them, from CSV text per table.

    Source i is named s<i>; persons are numbered by (source, patient_id).
    """
    frames: dict[str, list[pl.DataFrame]] = {}
    for i, files in enumerate(sources, start=1):
        for table, text in files.items():
            lf = apply_data_types(
                drop_derived_columns(
                    pl.scan_csv(io.BytesIO(text.encode()), infer_schema=False)
                ),
                table,
                DICTIONARY,
            )
            frames.setdefault(table, []).append(
                lf.with_columns(pl.lit(f"s{i}").alias(col._source_id)).collect()
            )
    tables = {t: pl.concat(f, how="diagonal") for t, f in frames.items()}
    persons = (
        tables[tbl.tnx_patient]
        .select(col._source_id, col.patient_id)
        .unique()
        .sort(col._source_id, col.patient_id)
        .with_row_index(col.person_id, offset=1)
        .with_columns(pl.col(col.person_id).cast(pl.Int64))
    )
    return {
        t: df.join(persons, on=[col._source_id, col.patient_id], how="left")
        for t, df in tables.items()
    }


def test_deduplicates_rows_that_differ_only_in_derived_flags():
    """Derived flag columns are gone before deduplication and derived rows stay."""
    # The two E2 rows differ only in their flags, so they become duplicates
    tables = clean_batch(
        batch(
            {
                tbl.tnx_patient: PATIENT + "P1,F,1950,\n",
                tbl.tnx_encounter: "encounter_id,patient_id,start_date,end_date,type,"
                "start_date_derived_by_TriNetX,end_date_derived_by_TriNetX,"
                "derived_by_TriNetX\n"
                "E1,P1,20200101,20200101,AMB,F,F,F\n"
                "E2,P1,20200201,20200201,AMB,T,T,T\n"
                "E2,P1,20200201,20200201,AMB,F,T,T\n",
                tbl.tnx_diagnosis: "patient_id,encounter_id,code_system,code,date,"
                "derived_by_TriNetX\n"
                "P1,E1,ICD-10-CM,E11.9,20200101,F\n"
                "P1,,ICD-10-CM,I10,20200201,T\n",
            }
        )
    )

    assert tables[tbl.tnx_encounter]["encounter_id"].sort().to_list() == ["E1", "E2"]
    assert len(tables[tbl.tnx_diagnosis]) == 2


def test_drops_duplicate_patient_rows_and_rows_without_birth_year():
    tables = clean_batch(
        batch(
            {
                tbl.tnx_patient: PATIENT + "P1,F,1950,\nP1,F,1950,\nP1,F,,\n",
                tbl.tnx_diagnosis: DIAGNOSIS + "P1,,ICD-10-CM,I10,20200101\n",
            }
        )
    )

    assert tables[tbl.tnx_patient].select(col.patient_id, col.year_of_birth).rows() == [
        ("P1", 1950)
    ]


class TestFixEncounterDates:
    """Tests for encounter date repair during cleaning."""

    @staticmethod
    def _encounters(rows):
        return pl.DataFrame(
            rows,
            schema=[col.encounter_id, col.type, col.start_date, col.end_date],
            orient="row",
        ).with_columns(
            pl.lit(1, dtype=pl.Int64).alias(col.person_id),
            pl.col(col.start_date, col.end_date)
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False),
        )

    @staticmethod
    def _linked(rows):
        return (
            pl.DataFrame(rows, schema=[col.encounter_id, col.date], orient="row")
            .with_columns(
                pl.lit(1, dtype=pl.Int64).alias(col.person_id),
                pl.col(col.date).str.to_date("%Y%m%d", strict=False),
            )
            .select(col.person_id, col.encounter_id, col.date)
            .lazy()
        )

    @staticmethod
    def _dates(df):
        return {
            r[col.encounter_id]: (
                r[col.start_date].strftime("%Y%m%d"),
                r[col.end_date].strftime("%Y%m%d"),
            )
            for r in df.iter_rows(named=True)
        }

    def test_non_inpatient_missing_end_uses_start(self):
        enc = self._encounters(
            [("E1", "AMB", "20200105", None), ("E2", None, "20200106", "")]
        )
        linked = self._linked([("E1", "20200101"), ("E1", "20200110")])

        result = _fix_encounter_dates(enc, [linked])

        assert self._dates(result) == {
            "E1": ("20200105", "20200105"),
            "E2": ("20200106", "20200106"),
        }

    def test_end_before_start_uses_linked_range(self):
        enc = self._encounters(
            [
                ("E1", "IMP", "20200110", "20200105"),
                ("E2", "AMB", "20200110", "20200101"),
            ]
        )
        diagnosis = self._linked([("E1", "20200103"), ("E2", "20200108")])
        procedure = self._linked([("E1", "20200112"), ("E1", "20200107")])

        result = _fix_encounter_dates(enc, [diagnosis, procedure])

        assert self._dates(result) == {
            "E1": ("20200103", "20200112"),
            "E2": ("20200108", "20200108"),
        }

    def test_inpatient_missing_end_uses_linked_range(self):
        enc = self._encounters([("E1", "IMP", "20200110", None)])
        linked = self._linked([("E1", "20200111"), ("E1", "20200115")])

        result = _fix_encounter_dates(enc, [linked])

        assert self._dates(result) == {"E1": ("20200111", "20200115")}

    def test_without_linked_records_end_uses_start(self):
        enc = self._encounters(
            [("E1", "IMP", "20200110", "20200105"), ("E2", "IMP", "20200110", None)]
        )

        result = _fix_encounter_dates(enc, [self._linked([("E9", "20200101")])])

        assert self._dates(result) == {
            "E1": ("20200110", "20200110"),
            "E2": ("20200110", "20200110"),
        }

    def test_valid_encounters_are_unchanged(self):
        enc = self._encounters([("E1", "IMP", "20200110", "20200112")])
        linked = self._linked([("E1", "20200101"), ("E1", "20200130")])

        result = _fix_encounter_dates(enc, [linked])

        assert self._dates(result) == {"E1": ("20200110", "20200112")}
        assert result.columns == enc.columns


def test_clean_batch_repairs_encounter_dates():
    """Uses diagnosis and procedure dates but not medication dates."""
    tables = batch(
        {
            tbl.tnx_patient: PATIENT + "P1,F,1950,\n",
            tbl.tnx_encounter: ENCOUNTER + "E1,P1,20200110,20200105,IMP\n",
            tbl.tnx_diagnosis: DIAGNOSIS + "P1,E1,ICD-10-CM,I10,20200108\n",
            tbl.tnx_procedure: DIAGNOSIS + "P1,E1,CPT,99223,20200113\n",
            tbl.tnx_medication_ingredient: "patient_id,encounter_id,unique_id,"
            "code_system,code,start_date\nP1,E1,M1,RxNorm,1191,20200301\n",
        }
    )

    encounter = clean_batch(tables)[tbl.tnx_encounter]

    assert encounter.select(col.start_date, col.end_date).row(0) == (
        date(2020, 1, 8),
        date(2020, 1, 13),
    )


def test_clean_batch_repairs_dates_from_records_of_the_same_person_only():
    """Persons with the same encounter_id do not share linked record dates."""
    tables = batch(
        {
            tbl.tnx_patient: PATIENT + "P1,F,1950,\nP2,F,1950,\n",
            tbl.tnx_encounter: ENCOUNTER
            + "E1,P1,20200110,20200105,IMP\nE1,P2,20200110,20200105,IMP\n",
            tbl.tnx_diagnosis: DIAGNOSIS
            + "P1,E1,ICD-10-CM,I10,20200103\nP2,E1,ICD-10-CM,I10,20200107\n",
        }
    )

    encounter = clean_batch(tables)[tbl.tnx_encounter].sort(col.person_id)

    assert encounter.select(col.start_date, col.end_date).rows() == [
        (date(2020, 1, 3), date(2020, 1, 3)),
        (date(2020, 1, 7), date(2020, 1, 7)),
    ]


def test_clean_batch_nulls_unknown_encounter_ids():
    """Empty or unknown encounter_id becomes null and the rows are kept."""
    tables = batch(
        {
            tbl.tnx_patient: PATIENT + "P1,F,1950,\n",
            tbl.tnx_encounter: ENCOUNTER + "E1,P1,20200101,20200101,AMB\n",
            tbl.tnx_diagnosis: QUOTED_DIAGNOSIS
            + '"P1","E1","ICD-10-CM","I10","20200101"\n'
            + '"P1","","ICD-10-CM","E11.9","20200102"\n'
            + '"P1","E9","ICD-10-CM","J45","20200103"\n',
        }
    )

    diagnosis = clean_batch(tables)[tbl.tnx_diagnosis].sort(col.date)

    assert diagnosis[col.encounter_id].to_list() == ["E1", None, None]


def test_clean_batch_nulls_encounter_ids_of_other_persons_and_sources():
    """An encounter matches only records of its own source and patient."""
    s1 = {
        tbl.tnx_patient: PATIENT + "P1,F,1950,\nP2,F,1950,\n",
        tbl.tnx_encounter: ENCOUNTER + "E1,P1,20200101,20200101,AMB\n",
        tbl.tnx_diagnosis: DIAGNOSIS
        + "P1,E1,ICD-10-CM,I10,20200101\nP2,E1,ICD-10-CM,I10,20200101\n",
    }
    s2 = {
        tbl.tnx_patient: PATIENT + "P1,F,1950,\n",
        tbl.tnx_diagnosis: DIAGNOSIS + "P1,E1,ICD-10-CM,I10,20200101\n",
    }

    diagnosis = clean_batch(batch(s1, s2))[tbl.tnx_diagnosis].sort(col.person_id)

    assert diagnosis.select(
        col._source_id, col.patient_id, col.encounter_id
    ).rows() == [
        ("s1", "P1", "E1"),
        ("s1", "P2", None),
        ("s2", "P1", None),
    ]


def test_clean_batch_drops_rows_that_become_identical_after_nulling():
    """Rows differing only in an unknown encounter_id collapse to one row."""
    tables = batch(
        {
            tbl.tnx_patient: PATIENT + "P1,F,1950,\n",
            tbl.tnx_encounter: ENCOUNTER + "E1,P1,20200101,20200101,AMB\n",
            tbl.tnx_diagnosis: QUOTED_DIAGNOSIS
            + '"P1","E8","ICD-10-CM","I10","20200101"\n'
            + '"P1","E9","ICD-10-CM","I10","20200101"\n'
            + '"P1","","ICD-10-CM","I10","20200101"\n'
            + '"P1","E1","ICD-10-CM","I10","20200101"\n'
            + '"P1","E1","ICD-10-CM","I10","20200101"\n',
        }
    )

    diagnosis = clean_batch(tables)[tbl.tnx_diagnosis].sort(
        col.encounter_id, nulls_last=True
    )

    assert diagnosis[col.encounter_id].to_list() == ["E1", None]


def test_clean_batch_keeps_identical_rows_of_different_sources():
    """The same rows in two sources stay two records."""
    source = {
        tbl.tnx_patient: PATIENT + "P1,F,1950,\n",
        tbl.tnx_procedure: DIAGNOSIS + "P1,,CPT,99213,20200101\n",
    }

    procedure = clean_batch(batch(source, source))[tbl.tnx_procedure]

    assert sorted(procedure[col._source_id].to_list()) == ["s1", "s2"]


def test_clean_batch_nulls_encounter_ids_without_encounter_table():
    """Without encounter.csv no encounter_id can be matched."""
    tables = batch(
        {
            tbl.tnx_patient: PATIENT + "P1,F,1950,\n",
            tbl.tnx_procedure: DIAGNOSIS + "P1,E1,CPT,99213,20200101\n",
        }
    )

    procedure = clean_batch(tables)[tbl.tnx_procedure]

    assert procedure[col.encounter_id].to_list() == [None]


def test_clean_batch_drops_records_from_reported_death_month():
    """Keeps records before the reported month and drops the rest."""
    tables = batch(
        {
            tbl.tnx_patient: PATIENT + "P1,F,1950,202003\nP2,M,1950,\n",
            tbl.tnx_diagnosis: DIAGNOSIS
            + "P1,,ICD-10-CM,I10,20200229\n"
            + "P1,,ICD-10-CM,I10,20200301\n"
            + "P1,,ICD-10-CM,I10,20200415\n"
            + "P2,,ICD-10-CM,I10,20200415\n",
        }
    )

    diagnosis = clean_batch(tables)[tbl.tnx_diagnosis]

    assert sorted(diagnosis.select(col.patient_id, col.date).rows()) == [
        ("P1", date(2020, 2, 29)),
        ("P2", date(2020, 4, 15)),
    ]


def test_clean_batch_death_of_one_source_does_not_affect_another():
    """The same patient_id in two sources has two death dates."""
    s1 = {
        tbl.tnx_patient: PATIENT + "P1,F,1950,202003\n",
        tbl.tnx_diagnosis: DIAGNOSIS + "P1,,ICD-10-CM,I10,20200415\n",
    }
    s2 = {
        tbl.tnx_patient: PATIENT + "P1,F,1950,\n",
        tbl.tnx_diagnosis: DIAGNOSIS + "P1,,ICD-10-CM,I10,20200415\n",
    }

    diagnosis = clean_batch(batch(s1, s2))[tbl.tnx_diagnosis]

    assert diagnosis[col._source_id].to_list() == ["s2"]


def test_clean_batch_ends_encounters_before_death():
    """Encounters that run into the reported death month end the day before."""
    tables = batch(
        {
            tbl.tnx_patient: PATIENT + "P1,F,1950,202003\n",
            tbl.tnx_encounter: ENCOUNTER
            + "E1,P1,20200210,20200215,IMP\n"
            + "E2,P1,20200225,20200305,IMP\n"
            + "E3,P1,20200229,20200301,AMB\n",
        }
    )

    encounter = clean_batch(tables)[tbl.tnx_encounter].sort(col.encounter_id)

    assert encounter.select(col.start_date, col.end_date).rows() == [
        (date(2020, 2, 10), date(2020, 2, 15)),
        (date(2020, 2, 25), date(2020, 2, 29)),
        (date(2020, 2, 29), date(2020, 2, 29)),
    ]


def test_clean_batch_input_unchanged():
    tables = batch(
        {
            tbl.tnx_patient: PATIENT + "P1,F,1950,\n",
            tbl.tnx_diagnosis: DIAGNOSIS + "P1,E1,ICD-10-CM,I10,20200101\n",
        }
    )

    clean_batch(tables)

    assert tables[tbl.tnx_diagnosis][col.encounter_id].to_list() == ["E1"]
