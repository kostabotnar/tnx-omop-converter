"""Create OMOP CDM_SOURCE table from TriNetX dataset metadata."""

from datetime import date
from pathlib import Path
from typing import Iterable

import polars as pl

from .util import columns as col
from .util import tables as tbl

CDM_VERSION = "v5.4"
CDM_VERSION_CONCEPT_ID = 756265  # OMOP CDM Version 5.4
SOURCE_ABBREVIATION = "TriNetX"
ETL_REFERENCE = "tnx-omop-converter"


def _read_column(data_dirs: Iterable[Path], table_name: str, column: str) -> list:
    values = []
    for data_dir in data_dirs:
        path = data_dir / f"{table_name}.csv"
        if path.exists():
            df = pl.read_csv(path, infer_schema=False)
            if column in df.columns:
                values.extend(df[column].drop_nulls().to_list())
    return values


def build_cdm_source(
    data_dirs: Iterable[Path], cdm_release_date: date, vocabulary_version: str
) -> pl.DataFrame:
    """Build the single CDM_SOURCE row from dataset_details and cohort_details.

    Args:
        data_dirs: Extracted TriNetX data directories, one per input ZIP
        cdm_release_date: Date the OMOP output is created
        vocabulary_version: Athena vocabulary release (see
            athena.read_vocabulary_version)

    Returns:
        CDM_SOURCE DataFrame with one row. When no export date is available,
        source_release_date falls back to cdm_release_date.
    """
    data_dirs = list(data_dirs)
    networks = sorted(
        set(_read_column(data_dirs, tbl.tnx_dataset_details, col.network_name))
    )
    cohorts = sorted(
        set(_read_column(data_dirs, tbl.tnx_cohort_details, col.cohort_name))
    )
    created = [
        d
        for d in pl.Series(
            _read_column(data_dirs, tbl.tnx_dataset_details, col.date_created),
            dtype=pl.Utf8,
        )
        .str.to_date("%Y%m%d", strict=False)
        .to_list()
        if d is not None
    ]

    source_name = "; ".join(networks) or "TriNetX export"
    description = "TriNetX EHR export"
    if cohorts:
        description += f". Cohorts: {'; '.join(cohorts)}"

    return pl.DataFrame(
        {
            col.cdm_source_name: [source_name],
            col.cdm_source_abbreviation: [SOURCE_ABBREVIATION],
            col.cdm_holder: [source_name],
            col.source_description: [description],
            col.source_documentation_reference: [None],
            col.cdm_etl_reference: [ETL_REFERENCE],
            col.source_release_date: [max(created) if created else cdm_release_date],
            col.cdm_release_date: [cdm_release_date],
            col.cdm_version: [CDM_VERSION],
            col.cdm_version_concept_id: [CDM_VERSION_CONCEPT_ID],
            col.vocabulary_version: [vocabulary_version],
        },
        schema={
            col.cdm_source_name: pl.Utf8,
            col.cdm_source_abbreviation: pl.Utf8,
            col.cdm_holder: pl.Utf8,
            col.source_description: pl.Utf8,
            col.source_documentation_reference: pl.Utf8,
            col.cdm_etl_reference: pl.Utf8,
            col.source_release_date: pl.Date,
            col.cdm_release_date: pl.Date,
            col.cdm_version: pl.Utf8,
            col.cdm_version_concept_id: pl.Int64,
            col.vocabulary_version: pl.Utf8,
        },
    )
