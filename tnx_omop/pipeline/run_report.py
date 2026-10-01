"""The JSON run report written to `<output>/run_report.json` after a conversion.

It records what was converted, with which tool and vocabulary versions, the row
counts, and the time and peak memory of each stage. It is separate from the resume
manifest in the working directory (see manifest.py), which is deleted after a run.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import polars as pl

from ..schema.domains import DOMAIN_TABLES
from ..util import columns as col
from ..util.concept_mappings import describe_config
from ..util.run_stats import RunStats

REPORT_FILE = "run_report.json"
REPORT_TEMP_FILE = "run_report.json.tmp"
DISTRIBUTION_NAME = "tnx-omop-converter"
UNKNOWN_VERSION = "unknown"


def tool_version() -> str:
    """Version of the installed distribution, or "unknown" when it is not installed."""
    try:
        return metadata.version(DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return UNKNOWN_VERSION


def excluded_row_counts(coverage: pl.DataFrame) -> dict[str, int]:
    """Excluded output rows per TriNetX table, from the merged coverage frame.

    A coverage row is excluded when its outcome is an exclusion reason instead of the
    name of an OMOP table.
    """
    excluded = (
        coverage.filter(~pl.col(col.outcome).is_in(list(DOMAIN_TABLES.values())))
        .group_by(col.tnx_table)
        .agg(pl.col(col.rows).sum())
        .sort(col.tnx_table)
    )
    return dict(excluded.iter_rows())


def build_report(
    *,
    started_at: datetime,
    finished_at: datetime,
    input_zips: list[Path],
    vocab_dir: Path,
    vocabulary_version: str,
    batch_rows: int,
    batch_count: int,
    person_count: int,
    ingest_stages_skipped: list[str],
    batches_already_done: int,
    row_counts: dict[str, int],
    excluded_rows: dict[str, int],
    stats: RunStats,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Collect the content of run_report.json.

    Args:
        started_at: Start of the run (timezone aware, UTC).
        finished_at: End of the run (timezone aware, UTC).
        input_zips: TriNetX ZIP files that were converted.
        vocab_dir: Athena vocabulary folder.
        vocabulary_version: Athena release, as in CDM_SOURCE.
        batch_rows: Upper bound of source rows per batch.
        batch_count: Number of batches.
        person_count: Number of persons in PERSON.
        ingest_stages_skipped: Ingest stages taken over from an earlier run.
        batches_already_done: Batches taken over from an earlier run.
        row_counts: Rows per output table.
        excluded_rows: Excluded rows per TriNetX table.
        stats: Time and memory measurements of the stages, already stopped.
        config_path: User config file with concept map overrides, or None.

    Returns:
        JSON-serializable dictionary. Peak memory values are null when the platform
        has no counter for them.
    """
    resumed = bool(ingest_stages_skipped or batches_already_done)
    return {
        "tool": {"name": DISTRIBUTION_NAME, "version": tool_version()},
        "python_version": platform.python_version(),
        "polars_version": pl.__version__,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "total_seconds": round((finished_at - started_at).total_seconds(), 3),
        "inputs": [
            {"path": str(path), "size_bytes": path.stat().st_size}
            for path in input_zips
        ],
        "vocabulary": {"path": str(vocab_dir), "version": vocabulary_version},
        "config": describe_config(config_path),
        "batch_rows": batch_rows,
        "batch_count": batch_count,
        "person_count": person_count,
        "resumed": resumed,
        "ingest_stages_skipped": ingest_stages_skipped,
        "batches_already_done": batches_already_done,
        "row_counts": row_counts,
        "excluded_rows": excluded_rows,
        "stages": {
            name: {
                "seconds": round(stage.seconds, 3),
                "peak_memory_bytes": stage.peak_memory_bytes,
            }
            for name, stage in stats.stages.items()
        },
        "process_peak_memory_bytes": stats.process_peak_bytes,
        "memory_counter": stats.counter,
    }


def write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    """Write the report to output_dir/run_report.json through a temporary file.

    The temporary file is renamed in one step, so a reader never sees a partial report.
    """
    path = output_dir / REPORT_FILE
    temp = output_dir / REPORT_TEMP_FILE
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return path
