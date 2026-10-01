"""Map TriNetX codes and units to standard OMOP concepts using an Athena download.

A TriNetX (table, code_system, code) is resolved to an Athena source concept via the
code system to vocabulary config, then followed through valid "Maps to" relationships
to standard concepts. The lookup has one row per standard target, so a code mapping
to two standard concepts produces two output records. Codes without a usable target
get one row with concept_id null and an exclusion_reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Union

import polars as pl

from ..util import columns as col
from ..util import tables as tbl
from .athena import scan_concepts, scan_maps_to, unique_concepts
from ..util.concept_mappings import SOURCE_VOCABULARY_MAP, UNIT_UCUM_MAP
from ..domains import (
    DOMAIN_TABLES,
    NO_SOURCE_CONCEPT,
    NO_STANDARD_MAPPING,
    UNSUPPORTED_DOMAIN,
)

logger = logging.getLogger(__name__)

RXNORM = "RxNorm"
RXNORM_EXTENSION = "RxNorm Extension"
# RxNorm codes with this prefix are RxNorm Extension concept codes, not RxNorm CUIs.
RXNORM_EXTENSION_PREFIX = "OMOP"
UCUM = "UCUM"
STANDARD = "S"

UNIT_TABLES: List[str] = [tbl.tnx_lab_result, tbl.tnx_vitals_signs]

CODES_SCHEMA = pl.Schema(
    {
        col._tnx_table: pl.String,
        col.code_system: pl.String,
        col.code: pl.String,
        col.source_concept_id: pl.Int64,
        col.concept_id: pl.Int64,
        col.domain_id: pl.String,
        col.exclusion_reason: pl.String,
    }
)
UNITS_SCHEMA = pl.Schema(
    {col.units_of_measure: pl.String, col.unit_concept_id: pl.Int64}
)

_UNIT_LOOKUP_CODE = "_unit_lookup_code"
_SOURCE_INVALID = "_source_invalid"
_STANDARD_ONLY = "_standard_only"

Frame = Union[pl.DataFrame, pl.LazyFrame]


@dataclass
class VocabularyLookup:
    """Resolved code and unit mappings.

    Attributes:
        codes: One row per (_tnx_table, code_system, code, target). Mapped rows have
            concept_id and domain_id of a standard concept in a supported domain and a
            null exclusion_reason. Other rows have a null concept_id and an
            exclusion_reason; unsupported_domain rows keep the target domain_id.
            source_concept_id is 0 when no source concept exists.
        units: units_of_measure to UCUM unit_concept_id (0 when unmapped).
    """

    codes: pl.DataFrame
    units: pl.DataFrame

    def codes_for(self, tnx_table: str) -> pl.DataFrame:
        """Return the code rows of one TriNetX table, without the table column."""
        return self.codes.filter(pl.col(col._tnx_table) == tnx_table).drop(
            col._tnx_table
        )


def collect_source_codes(tables: Mapping[str, Frame]) -> pl.DataFrame:
    """Return distinct non-null (_tnx_table, code_system, code) from clinical tables.

    Args:
        tables: TriNetX tables keyed by table name; LazyFrames are streamed. Tables
            without a code system config entry are ignored.
    """
    parts = [
        frame.lazy()
        .select(
            pl.lit(name, dtype=pl.String).alias(col._tnx_table),
            pl.col(col.code_system).cast(pl.String),
            pl.col(col.code).cast(pl.String),
        )
        .drop_nulls()
        .unique()
        for name, frame in tables.items()
        if name in SOURCE_VOCABULARY_MAP
    ]
    if not parts:
        return pl.DataFrame(
            schema={
                c: CODES_SCHEMA[c] for c in (col._tnx_table, col.code_system, col.code)
            }
        )
    return pl.concat(parts).collect(engine="streaming")


def collect_units(tables: Mapping[str, Frame]) -> pl.DataFrame:
    """Return distinct non-null units_of_measure from lab_result and vitals_signs."""
    parts = [
        tables[name]
        .lazy()
        .select(pl.col(col.units_of_measure).cast(pl.String))
        .drop_nulls()
        for name in UNIT_TABLES
        if name in tables
    ]
    if not parts:
        return pl.DataFrame(schema={col.units_of_measure: pl.String})
    return pl.concat(parts).unique().collect(engine="streaming")


def _vocabulary_config() -> pl.DataFrame:
    return pl.DataFrame(
        [
            (table, system, vocabulary_id)
            for table, systems in SOURCE_VOCABULARY_MAP.items()
            for system, vocabulary_id in systems.items()
        ],
        schema={
            col._tnx_table: pl.String,
            col.code_system: pl.String,
            col.vocabulary_id: pl.String,
        },
        orient="row",
    )


def _attach_vocabulary(codes: pl.DataFrame) -> pl.DataFrame:
    """Add the Athena vocabulary_id each code is looked up in (null: none)."""
    vocabulary = pl.col(col.vocabulary_id)
    return codes.join(
        _vocabulary_config(), on=[col._tnx_table, col.code_system], how="left"
    ).with_columns(
        pl.when(
            (vocabulary == RXNORM)
            & pl.col(col.code).str.starts_with(RXNORM_EXTENSION_PREFIX)
        )
        .then(pl.lit(RXNORM_EXTENSION))
        .otherwise(vocabulary)
        .alias(col.vocabulary_id)
    )


def _unit_lookup_codes(units: pl.DataFrame) -> pl.DataFrame:
    overrides = pl.DataFrame(
        {
            col.units_of_measure: list(UNIT_UCUM_MAP),
            _UNIT_LOOKUP_CODE: list(UNIT_UCUM_MAP.values()),
        },
        schema={col.units_of_measure: pl.String, _UNIT_LOOKUP_CODE: pl.String},
    )
    return (
        units.select(col.units_of_measure)
        .unique()
        .join(overrides, on=col.units_of_measure, how="left")
        .with_columns(
            pl.coalesce(_UNIT_LOOKUP_CODE, col.units_of_measure).alias(
                _UNIT_LOOKUP_CODE
            )
        )
    )


def _read_candidate_concepts(
    vocab_dir: Path, codes: pl.DataFrame, unit_codes: List[str]
) -> pl.DataFrame:
    """One CONCEPT scan for source code candidates and UCUM unit candidates.

    Args:
        vocab_dir: Folder with the Athena CSV files.
        codes: Distinct (vocabulary_id, code) pairs to look up.
        unit_codes: UCUM concept codes of the units.

    Unit candidates are standard UCUM concepts only: unit_concept_id must be
    standard, and a non-standard UCUM code leaves the unit unmapped (0).
    """
    keys = pl.concat(
        [
            codes.select(
                pl.col(col.vocabulary_id),
                pl.col(col.code).alias(col.concept_code),
                pl.lit(False).alias(_STANDARD_ONLY),
            ),
            pl.DataFrame(
                {
                    col.vocabulary_id: [UCUM] * len(unit_codes),
                    col.concept_code: unit_codes,
                    _STANDARD_ONLY: [True] * len(unit_codes),
                },
                schema={
                    col.vocabulary_id: pl.String,
                    col.concept_code: pl.String,
                    _STANDARD_ONLY: pl.Boolean,
                },
            ),
        ]
    ).unique()
    candidates = scan_concepts(vocab_dir).join(
        keys.lazy(), on=[col.vocabulary_id, col.concept_code], how="inner"
    )
    return (
        unique_concepts(
            candidates.filter(
                ~pl.col(_STANDARD_ONLY) | (pl.col(col.standard_concept) == STANDARD)
            ).drop(_STANDARD_ONLY)
        )
        .select(
            pl.col(col.concept_id).cast(pl.Int64),
            col.vocabulary_id,
            col.concept_code,
            (pl.col(col.invalid_reason).fill_null("") != "").alias(_SOURCE_INVALID),
        )
        .collect(engine="streaming")
    )


def _first_by_validity(concepts: pl.DataFrame, keys: List[str]) -> pl.DataFrame:
    # Prefer a valid concept, then the lowest concept_id, for a stable choice.
    return concepts.sort([_SOURCE_INVALID, col.concept_id]).unique(
        keys, keep="first", maintain_order=True
    )


def _read_standard_targets(vocab_dir: Path, source_ids: pl.Series) -> pl.DataFrame:
    """Return (concept_id_1, concept_id, domain_id) for standard Maps to targets."""
    source_keys = source_ids.alias(col.concept_id_1).to_frame().lazy().unique()
    maps_to = (
        scan_maps_to(vocab_dir)
        .join(source_keys, on=col.concept_id_1, how="semi")
        .unique()
        .collect(engine="streaming")
    )
    target_keys = (
        maps_to.select(pl.col(col.concept_id_2).alias(col.concept_id)).unique().lazy()
    )
    targets = (
        unique_concepts(
            scan_concepts(vocab_dir)
            .filter(pl.col(col.standard_concept) == STANDARD)
            .with_columns(pl.col(col.concept_id).cast(pl.Int64, strict=False))
            .join(target_keys, on=col.concept_id, how="semi")
        )
        .select(col.concept_id, col.domain_id)
        .collect(engine="streaming")
    )
    return maps_to.join(
        targets, left_on=col.concept_id_2, right_on=col.concept_id, how="inner"
    ).select(
        col.concept_id_1,
        pl.col(col.concept_id_2).alias(col.concept_id),
        col.domain_id,
    )


def _resolve_codes(
    codes: pl.DataFrame, sources: pl.DataFrame, targets: pl.DataFrame
) -> pl.DataFrame:
    source_ids = sources.select(
        col.vocabulary_id,
        pl.col(col.concept_code).alias(col.code),
        pl.col(col.concept_id).alias(col.source_concept_id),
    )
    resolved = (
        codes.join(source_ids, on=[col.vocabulary_id, col.code], how="left")
        .join(
            targets,
            left_on=col.source_concept_id,
            right_on=col.concept_id_1,
            how="left",
        )
        .with_columns(
            pl.when(pl.col(col.source_concept_id).is_null())
            .then(pl.lit(NO_SOURCE_CONCEPT))
            .when(pl.col(col.concept_id).is_null())
            .then(pl.lit(NO_STANDARD_MAPPING))
            .when(~pl.col(col.domain_id).is_in(list(DOMAIN_TABLES)))
            .then(pl.lit(UNSUPPORTED_DOMAIN))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias(col.exclusion_reason),
            pl.col(col.source_concept_id).fill_null(0),
        )
        .with_columns(
            pl.when(pl.col(col.exclusion_reason).is_null())
            .then(pl.col(col.concept_id))
            .alias(col.concept_id)
        )
    )
    # Several unsupported targets of one code collapse to one excluded row per domain.
    return (
        resolved.select(list(CODES_SCHEMA.names()))
        .unique()
        .sort(
            [col._tnx_table, col.code_system, col.code, col.concept_id, col.domain_id],
            nulls_last=True,
        )
    )


def _resolve_units(units: pl.DataFrame, ucum: pl.DataFrame) -> pl.DataFrame:
    ucum_ids = ucum.select(
        pl.col(col.concept_code).alias(_UNIT_LOOKUP_CODE),
        pl.col(col.concept_id).alias(col.unit_concept_id),
    )
    return (
        _unit_lookup_codes(units)
        .join(ucum_ids, on=_UNIT_LOOKUP_CODE, how="left")
        .select(
            col.units_of_measure,
            pl.col(col.unit_concept_id).fill_null(0),
        )
        .sort(col.units_of_measure)
    )


def build_lookup(
    vocab_dir: Path, codes: pl.DataFrame, units: pl.DataFrame
) -> VocabularyLookup:
    """Resolve TriNetX codes and units against an Athena vocabulary download.

    Args:
        vocab_dir: Folder with the Athena CSV files.
        codes: Distinct (_tnx_table, code_system, code), see collect_source_codes.
        units: Distinct units_of_measure, see collect_units.

    Returns:
        VocabularyLookup with one code row per standard target (see its docstring).
    """
    codes = _attach_vocabulary(
        codes.select(col._tnx_table, col.code_system, col.code).unique()
    )
    unit_codes = _unit_lookup_codes(units).get_column(_UNIT_LOOKUP_CODE).to_list()
    lookup_codes = codes.select(col.vocabulary_id, col.code).drop_nulls().unique()

    candidates = _read_candidate_concepts(
        vocab_dir,
        lookup_codes,
        sorted(set(unit_codes) | set(UNIT_UCUM_MAP.values())),
    )
    is_ucum = pl.col(col.vocabulary_id) == UCUM
    ucum = _first_by_validity(candidates.filter(is_ucum), [col.concept_code])
    sources = _first_by_validity(
        candidates.filter(~is_ucum), [col.vocabulary_id, col.concept_code]
    )

    found_ucum = set(ucum.get_column(col.concept_code).to_list())
    for unit, target in UNIT_UCUM_MAP.items():
        if target not in found_ucum:
            logger.warning(
                "Unit override %s -> %s: no UCUM concept with that code", unit, target
            )

    targets = _read_standard_targets(vocab_dir, sources.get_column(col.concept_id))
    return VocabularyLookup(
        codes=_resolve_codes(codes, sources, targets),
        units=_resolve_units(units, ucum),
    )
