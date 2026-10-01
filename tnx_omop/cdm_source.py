"""Read TriNetX dataset metadata files for the OMOP CDM_SOURCE table.

The table logic is in transformers/cdm_source.py.
"""

from datetime import date
from pathlib import Path
from typing import Iterable

import polars as pl

from .transformers.cdm_source import CDM_VERSION_CONCEPT_ID, transform_cdm_source
from .util import columns as col
from .util import tables as tbl

__all__ = ["CDM_VERSION_CONCEPT_ID", "build_cdm_source"]


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
    return transform_cdm_source(
        _read_column(data_dirs, tbl.tnx_dataset_details, col.network_name),
        _read_column(data_dirs, tbl.tnx_cohort_details, col.cohort_name),
        _read_column(data_dirs, tbl.tnx_dataset_details, col.date_created),
        cdm_release_date,
        vocabulary_version,
    )
