"""Parquet writing with the options every output file uses."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import polars as pl

ROW_GROUP_SIZE = 500_000
COMPRESSION = "zstd"
# Codecs a user can choose for the final output files; all are written and read by Polars.
COMPRESSIONS = ("zstd", "snappy", "gzip", "lz4", "brotli", "uncompressed")
# Files the pipeline reads back once and then rewrites (task outputs, batch parts) skip
# statistics and use a compression that costs far less CPU than zstd.
TEMP_COMPRESSION = "lz4"


def _options(
    row_group_size: int = ROW_GROUP_SIZE,
    temporary: bool = False,
    compression: str = COMPRESSION,
) -> dict:
    return {
        "compression": TEMP_COMPRESSION if temporary else compression,
        "statistics": not temporary,
        "row_group_size": row_group_size,
    }


def write_parquet(
    df: pl.DataFrame,
    path: Union[str, Path],
    temporary: bool = False,
    compression: str = COMPRESSION,
) -> None:
    """Write a DataFrame with statistics, fixed row groups and the given codec.

    compression is one of COMPRESSIONS (default zstd). temporary=True writes for a
    file that is read back and rewritten (see TEMP_COMPRESSION) and ignores
    compression.
    """
    df.write_parquet(path, **_options(temporary=temporary, compression=compression))


def sink_parquet(
    lf: pl.LazyFrame,
    path: Union[str, Path, pl.PartitionBy],
    row_group_size: int = ROW_GROUP_SIZE,
    temporary: bool = False,
    compression: str = COMPRESSION,
) -> None:
    """Stream a LazyFrame to a file with the same options as write_parquet.

    path may also be a pl.PartitionBy to write several files. A smaller
    row_group_size lowers the memory of files that Polars reads again in
    parallel (see ingest.WORK_ROW_GROUP_SIZE). temporary and compression are as in write_parquet.
    """
    lf.sink_parquet(path, **_options(row_group_size, temporary, compression))
