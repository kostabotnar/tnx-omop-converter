"""Transform TriNetX dataset metadata into the single OMOP CDM_SOURCE row."""

from collections.abc import Iterable
from datetime import date

import polars as pl

from ..schema.omop import CDM_SOURCE_SCHEMA
from ..util import columns as col

CDM_VERSION = "v5.4"
CDM_VERSION_CONCEPT_ID = 756265  # OMOP CDM Version 5.4
SOURCE_ABBREVIATION = "TriNetX"
ETL_REFERENCE = "tnx-omop-converter"


def transform_cdm_source(
    network_names: Iterable[str],
    cohort_names: Iterable[str],
    date_created: Iterable[str],
    cdm_release_date: date,
    vocabulary_version: str,
) -> pl.DataFrame:
    """Build the single CDM_SOURCE row.

    Args:
        network_names: network_name values of dataset_details, any number of
            duplicates.
        cohort_names: cohort_name values of cohort_details.
        date_created: date_created values of dataset_details as YYYYMMDD
            strings; values that do not parse are ignored.
        cdm_release_date: Date the OMOP output is created.
        vocabulary_version: Athena vocabulary release.

    Returns:
        CDM_SOURCE frame with one row. When no export date parses,
        source_release_date falls back to cdm_release_date.
    """
    networks = sorted(set(network_names))
    cohorts = sorted(set(cohort_names))
    created = [
        d
        for d in pl.Series(list(date_created), dtype=pl.Utf8)
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
        schema=CDM_SOURCE_SCHEMA.dtype_map,
    )
