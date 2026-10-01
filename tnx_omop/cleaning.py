"""Data cleaning module for TriNetX data before OMOP transformation."""

from typing import Dict, List, Optional

import polars as pl

from .util import tables as tbl
from .util import columns as col
from .util.date_utils import parse_month_year_death_column


# Table files mapping (module-level for use in functions)
TABLE_FILES = {
    tbl.tnx_patient: f"{tbl.tnx_patient}.csv",
    tbl.tnx_encounter: f"{tbl.tnx_encounter}.csv",
    tbl.tnx_diagnosis: f"{tbl.tnx_diagnosis}.csv",
    tbl.tnx_lab_result: f"{tbl.tnx_lab_result}.csv",
    tbl.tnx_vitals_signs: f"{tbl.tnx_vitals_signs}.csv",
    tbl.tnx_medication_ingredient: f"{tbl.tnx_medication_ingredient}.csv",
    tbl.tnx_procedure: f"{tbl.tnx_procedure}.csv",
}

# Date columns per table for post-death filtering
DATE_COLUMN_MAP = {
    tbl.tnx_encounter: col.start_date,
    tbl.tnx_diagnosis: col.date,
    tbl.tnx_lab_result: col.date,
    tbl.tnx_vitals_signs: col.date,
    tbl.tnx_medication_ingredient: col.start_date,
    tbl.tnx_procedure: col.date,
}


# Encounter types that map to an inpatient visit (HL7 ActEncounterCode)
INPATIENT_ENCOUNTER_TYPES = ["IMP", "ACUTE", "NONAC"]

# Tables whose dates are used to rebuild visit dates, joined on person and encounter
VISIT_DATE_SOURCES = {
    tbl.tnx_diagnosis: col.date,
    tbl.tnx_procedure: col.date,
}

_VALID_ENCOUNTER = "_valid_encounter"

# Encounters belong to one patient of one source; person_id identifies both
_ENCOUNTER_MATCH = [col._source_id, col.patient_id, col.encounter_id]
_ENCOUNTER_KEY = [col.person_id, col.encounter_id]


def clean_batch(tables: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
    """Clean the tables of one batch of persons before OMOP transformation.

    The tables come from ingest (typed, with _source_id and person_id, restricted to
    persons that have records), so every step works within one patient. Steps, in
    order:
    1. Drop patient rows without year_of_birth and duplicate patient and encounter
       rows
    2. Drop records dated in or after the death month TriNetX reports
    3. Set encounter_id to null where it matches no encounter of the same patient
       (rows are kept), then drop duplicate rows from the other tables, so rows
       that became identical are removed too
    4. Fix encounter dates that are missing or end before they start
    5. End encounters that run into the death month the day before it

    Args:
        tables: TriNetX tables of the batch keyed by table name, including patient.

    Returns:
        New dict with the cleaned tables.
    """
    result = dict(tables)
    patient_df = _drop_missing_birth_year(result[tbl.tnx_patient]).unique()
    result[tbl.tnx_patient] = patient_df
    death_lf = _death_dates(patient_df)

    for table_name, date_col in DATE_COLUMN_MAP.items():
        df = result.get(table_name)
        if df is None:
            continue
        lf = df.lazy()

        # Other tables are deduplicated after encounter IDs are nulled
        if table_name == tbl.tnx_encounter:
            lf = lf.unique()

        if death_lf is not None:
            lf = _filter_post_death_records(lf, date_col, death_lf)

        result[table_name] = lf.collect()

    _null_unknown_encounter_ids(result)
    for table_name in DATE_COLUMN_MAP:
        if table_name != tbl.tnx_encounter and result.get(table_name) is not None:
            result[table_name] = result[table_name].unique()

    if result.get(tbl.tnx_encounter) is not None:
        result[tbl.tnx_encounter] = _fix_encounter_dates(
            result[tbl.tnx_encounter],
            [
                result[t].lazy().select(*_ENCOUNTER_KEY, pl.col(c).alias(col.date))
                for t, c in VISIT_DATE_SOURCES.items()
                if result.get(t) is not None
            ],
        )
        if death_lf is not None:
            result[tbl.tnx_encounter] = _truncate_encounters_at_death(
                result[tbl.tnx_encounter], death_lf
            )

    return result


def _drop_missing_birth_year(patient_df: pl.DataFrame) -> pl.DataFrame:
    yob = pl.col(col.year_of_birth).cast(pl.Utf8)
    return patient_df.filter(yob.is_not_null() & (yob != ""))


def _death_dates(patient_df: pl.DataFrame) -> Optional[pl.LazyFrame]:
    """person_id and death_date of the patients with a reported death month."""
    if col.month_year_death not in patient_df.columns:
        return None
    death_df = (
        patient_df.select(
            pl.col(col.person_id),
            parse_month_year_death_column(pl.col(col.month_year_death)).alias(
                "death_date"
            ),
        )
        .filter(pl.col("death_date").is_not_null())
        .unique()
    )
    return None if death_df.is_empty() else death_df.lazy()


def _null_unknown_encounter_ids(
    tables: Dict[str, Optional[pl.DataFrame]],
) -> None:
    """Set encounter_id to null where it matches no cleaned encounter.

    Covers empty strings (TriNetX writes "" for records without an encounter)
    and IDs absent from encounter.csv, so OMOP visit_occurrence_id is either
    null or points to an existing visit. An encounter matches only records of its
    own source and patient. Updates tables in place.
    """
    encounter_df = tables.get(tbl.tnx_encounter)
    match_key = [pl.col(c).cast(pl.Utf8) for c in _ENCOUNTER_MATCH]
    if encounter_df is None:
        valid_ids = pl.DataFrame(
            schema={
                **{c: pl.Utf8 for c in _ENCOUNTER_MATCH},
                _VALID_ENCOUNTER: pl.Boolean,
            }
        )
    else:
        valid_ids = encounter_df.select(
            *match_key, pl.lit(True).alias(_VALID_ENCOUNTER)
        ).unique(_ENCOUNTER_MATCH)

    for table_name in DATE_COLUMN_MAP:
        df = tables.get(table_name)
        if (
            table_name == tbl.tnx_encounter
            or df is None
            or col.encounter_id not in df.columns
        ):
            continue
        # A left join marks the known IDs; null IDs never match and stay null
        tables[table_name] = (
            df.with_columns(match_key)
            .join(valid_ids, on=_ENCOUNTER_MATCH, how="left", maintain_order="left")
            .with_columns(
                pl.when(pl.col(_VALID_ENCOUNTER).is_not_null())
                .then(pl.col(col.encounter_id))
                .alias(col.encounter_id)
            )
            .drop(_VALID_ENCOUNTER)
        )


def _truncate_encounters_at_death(
    encounter_df: pl.DataFrame, death_lf: pl.LazyFrame
) -> pl.DataFrame:
    """Set end_date to the day before death_date where it is on or after it.

    Encounters starting on or after death_date were already removed, so the new
    end date is never before the start date.
    """
    last_day = pl.col("death_date") - pl.duration(days=1)
    end = pl.col(col.end_date)
    return (
        encounter_df.lazy()
        .join(death_lf, on=col.person_id, how="left")
        .with_columns(
            pl.when(end >= pl.col("death_date"))
            .then(last_day)
            .otherwise(end)
            .alias(col.end_date)
        )
        .drop("death_date")
        .collect()
    )


def _fix_encounter_dates(
    encounter_df: pl.DataFrame, linked_dates: List[pl.LazyFrame]
) -> pl.DataFrame:
    """Repair encounter start and end dates.

    - End date before start date (any type), or inpatient without an end date:
      start and end become the earliest and latest date of the diagnosis and
      procedure records linked to the encounter. Without linked records, the
      end date becomes the start date.
    - Other encounters without an end date: the end date becomes the start date.

    Args:
        encounter_df: Encounter table with Date start_date and end_date
        linked_dates: LazyFrames with person_id, encounter_id and a Date column

    Returns:
        Encounter table with corrected start_date and end_date
    """
    start = pl.col(col.start_date)
    end = pl.col(col.end_date)
    is_inpatient = pl.col(col.type).is_in(INPATIENT_ENCOUNTER_TYPES).fill_null(False)
    needs_linked = (end < start).fill_null(False) | (end.is_null() & is_inpatient)

    lf = encounter_df.lazy().with_columns(
        needs_linked.alias("_needs_linked"), start.alias("_start"), end.alias("_end")
    )

    if linked_dates:
        linked = (
            pl.concat(linked_dates)
            .filter(pl.col(col.encounter_id).is_not_null())
            .filter(pl.col(col.date).is_not_null())
            .group_by(_ENCOUNTER_KEY)
            .agg(
                pl.col(col.date).min().alias("_linked_start"),
                pl.col(col.date).max().alias("_linked_end"),
            )
        )
        # Only the encounters being repaired need the linked dates
        repair_ids = lf.filter(pl.col("_needs_linked")).select(_ENCOUNTER_KEY)
        linked = linked.join(repair_ids.unique(), on=_ENCOUNTER_KEY, how="semi")
        lf = lf.join(linked, on=_ENCOUNTER_KEY, how="left")
    else:
        lf = lf.with_columns(
            pl.lit(None, dtype=pl.Date).alias("_linked_start"),
            pl.lit(None, dtype=pl.Date).alias("_linked_end"),
        )

    use_linked = pl.col("_needs_linked") & pl.col("_linked_start").is_not_null()
    new_start = (
        pl.when(use_linked).then(pl.col("_linked_start")).otherwise(pl.col("_start"))
    )
    new_end = (
        pl.when(use_linked)
        .then(pl.col("_linked_end"))
        .when(pl.col("_needs_linked") | pl.col("_end").is_null())
        .then(pl.col("_start"))
        .otherwise(pl.col("_end"))
    )

    return (
        lf.with_columns(
            new_start.alias(col.start_date),
            new_end.alias(col.end_date),
        )
        .drop("_needs_linked", "_start", "_end", "_linked_start", "_linked_end")
        .collect()
    )


def _filter_post_death_records(
    lf: pl.LazyFrame, date_col: str, death_lf: pl.LazyFrame
) -> pl.LazyFrame:
    """Filter out records that occurred after the patient's death.

    death_date is the first day of the month TriNetX reports, which is the month
    after death, so records on or after that date are removed.

    Args:
        lf: LazyFrame to filter
        date_col: Name of the Date column
        death_lf: LazyFrame with person_id and death_date columns

    Returns:
        Filtered LazyFrame with post-death records removed
    """
    lf_with_death = lf.join(death_lf, on=col.person_id, how="left")

    lf_filtered = lf_with_death.filter(
        pl.col("death_date").is_null() | (pl.col(date_col) < pl.col("death_date"))
    )

    return lf_filtered.drop("death_date")
