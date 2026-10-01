"""Map TriNetX clinical records to standard concepts and route them by domain.

apply_lookup joins records to the vocabulary lookup: one mapped row per standard
target, and one excluded row per record without a usable target (or per
unsupported target domain). A record whose code maps to a Condition and an Episode
concept yields one mapped Condition row and one excluded row.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import polars as pl

from ..util import columns as col
from ..util import tables as tbl
from ..schema.domains import DOMAIN_TABLES, NO_SOURCE_CONCEPT
from ..omop_vocab.vocabulary import VocabularyLookup
from .condition import build_condition
from .device import build_device
from .drug import build_drug
from .measurement import build_measurement
from .observation import build_observation
from .procedure import build_procedure
from .sources import SOURCE_ADAPTERS

_LOOKUP_JOIN_KEYS: List[str] = [col.code_system, col.code]

BUILDERS: Dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {
    tbl.omop_condition_occurrence: build_condition,
    tbl.omop_procedure_occurrence: build_procedure,
    tbl.omop_drug_exposure: build_drug,
    tbl.omop_measurement: build_measurement,
    tbl.omop_observation: build_observation,
    tbl.omop_device_exposure: build_device,
}


def apply_lookup(
    df: pl.DataFrame, codes: pl.DataFrame, tnx_table: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Join records to their standard targets.

    Args:
        df: Cleaned TriNetX table with code_system and code columns.
        codes: Lookup rows of this table (VocabularyLookup.codes_for).
        tnx_table: TriNetX table name.

    Returns:
        (mapped, excluded). mapped has the original columns plus
        _source_concept_id, _concept_id and _domain_id, one row per
        standard target. excluded has the original columns plus exclusion_reason,
        target_domain_id and source_concept_id (0 when no source concept exists).
    """
    # Codes are strings in the lookup; an all-null or empty column may not be.
    df = df.with_columns(pl.col(*_LOOKUP_JOIN_KEYS).cast(pl.String))
    targets = codes.select(
        *_LOOKUP_JOIN_KEYS,
        col.source_concept_id,
        col.concept_id,
        col.domain_id,
        col.exclusion_reason,
    )
    joined = df.join(targets, on=_LOOKUP_JOIN_KEYS, how="left").with_columns(
        pl.col(col.source_concept_id).fill_null(0),
        pl.when(pl.col(col.concept_id).is_null())
        .then(pl.col(col.exclusion_reason).fill_null(NO_SOURCE_CONCEPT))
        .alias(col.exclusion_reason),
    )
    is_mapped = pl.col(col.exclusion_reason).is_null()

    mapped = (
        joined.filter(is_mapped)
        .rename(
            {
                col.source_concept_id: col._source_concept_id,
                col.concept_id: col._concept_id,
                col.domain_id: col._domain_id,
            }
        )
        .drop(col.exclusion_reason)
    )
    excluded = joined.filter(~is_mapped).select(
        *df.columns,
        col.exclusion_reason,
        pl.col(col.domain_id).alias(col.target_domain_id),
        col.source_concept_id,
    )
    return mapped, excluded


# Coverage part of one batch: output rows per code and outcome. Codes are kept so the
# merge can count distinct codes over all batches.
COVERAGE_PART_SCHEMA = pl.Schema(
    {
        col.tnx_table: pl.String,
        col.code_system: pl.String,
        col.code: pl.String,
        col.outcome: pl.String,
        col.rows: pl.Int64,
    }
)

# coverage.csv; the percentages are of the rows and distinct codes of the TriNetX table
COVERAGE_SCHEMA = pl.Schema(
    {
        col.tnx_table: pl.String,
        col.code_system: pl.String,
        col.outcome: pl.String,
        col.rows: pl.Int64,
        col.rows_pct: pl.Float64,
        col.codes: pl.Int64,
        col.codes_pct: pl.Float64,
    }
)


def record_coverage(
    df: pl.DataFrame, tnx_table: str, lookup: VocabularyLookup
) -> pl.DataFrame:
    """Count the output rows transform_clinical produces, by code and outcome.

    The outcome is the OMOP table a row lands in or the exclusion reason. Counts
    are output rows, so a record mapping to two targets counts twice.

    Returns:
        Frame with COVERAGE_PART_SCHEMA sorted by code_system, code and outcome.
    """
    codes = lookup.codes_for(tnx_table).select(
        *_LOOKUP_JOIN_KEYS, col.domain_id, col.concept_id, col.exclusion_reason
    )
    outcome = (
        pl.when(pl.col(col.concept_id).is_not_null())
        .then(
            pl.col(col.domain_id).replace_strict(
                DOMAIN_TABLES, default=None, return_dtype=pl.String
            )
        )
        .otherwise(pl.col(col.exclusion_reason).fill_null(NO_SOURCE_CONCEPT))
    )
    return (
        df.lazy()
        .select(pl.col(*_LOOKUP_JOIN_KEYS).cast(pl.String))
        .group_by(_LOOKUP_JOIN_KEYS)
        .agg(pl.len().cast(pl.Int64).alias(col.rows))
        .join(codes.lazy(), on=_LOOKUP_JOIN_KEYS, how="left")
        .group_by(col.code_system, col.code, outcome.alias(col.outcome))
        .agg(pl.col(col.rows).sum())
        .select(
            pl.lit(tnx_table, dtype=pl.String).alias(col.tnx_table),
            col.code_system,
            col.code,
            col.outcome,
            col.rows,
        )
        .sort(col.code_system, col.code, col.outcome, nulls_last=True)
        .collect()
        .cast(dict(COVERAGE_PART_SCHEMA))
    )


def clinical_events(
    df: pl.DataFrame, tnx_table: str, lookup: VocabularyLookup
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Map one TriNetX clinical table to events.

    Returns:
        (events, excluded): events has EVENT_SCHEMA plus _domain_id; excluded is
        the excluded frame of apply_lookup.
    """
    mapped, excluded = apply_lookup(df, lookup.codes_for(tnx_table), tnx_table)
    events = SOURCE_ADAPTERS[tnx_table](mapped, lookup.units).with_columns(
        mapped.get_column(col._domain_id)
    )
    return events, excluded


def transform_clinical(
    df: pl.DataFrame, tnx_table: str, lookup: VocabularyLookup
) -> Dict[str, pl.DataFrame]:
    """Map one TriNetX clinical table to OMOP tables by target concept domain.

    Args:
        df: Cleaned TriNetX table (diagnosis, procedure, medication_ingredient,
            lab_result or vitals_signs).
        tnx_table: TriNetX table name, a key of SOURCE_ADAPTERS.
        lookup: Vocabulary lookup covering the codes and units of df.

    Returns:
        OMOP table name -> rows, only for tables that received rows, plus
        tables.excluded -> excluded records (always present, possibly empty).
    """
    events, excluded = clinical_events(df, tnx_table, lookup)
    result: Dict[str, pl.DataFrame] = {}
    for (domain,), part in sorted(
        events.partition_by(col._domain_id, as_dict=True).items()
    ):
        table = DOMAIN_TABLES[domain]
        result[table] = BUILDERS[table](part.drop(col._domain_id))
    result[tbl.excluded] = excluded
    return result
