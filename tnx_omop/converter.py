"""Main orchestration for TriNetX to OMOP conversion."""

from __future__ import annotations

import logging
import shutil
import time
from datetime import UTC, date, datetime
from pathlib import Path

from . import eras as era_builder
from . import ingest
from .batch_transform import (
    PARTS_DIR,
    SCRATCH_DIR,
    completed_batches,
    counters_from_parts,
    run_batches,
)
from .cdm_source import build_cdm_source
from .manifest import (
    Fingerprint,
    MANIFEST_FILE,
    MANIFEST_TEMP_FILE,
    PIPELINE_VERSION,
    Manifest,
    build_fingerprint,
    load_manifest,
    mark_stage_done,
    matches,
    save_manifest,
)
from .merge import merge_coverage, merge_excluded, merge_tables
from .observation_period import create_observation_period
from .omop_vocab import vocabulary_export
from .omop_vocab.athena import read_vocabulary_version
from .omop_vocab.concept_table import create_concept_table, load_terminology, table_path
from .omop_vocab.vocabulary import (
    VocabularyLookup,
    build_lookup,
    collect_source_codes,
    collect_units,
)
from .util import concept_mappings
from .run_report import build_report, excluded_row_counts, write_report
from .tnx_schema import load_data_dictionary
from .util import tables as tbl
from .util.parquet_io import write_parquet
from .util.run_stats import RunStats

logger = logging.getLogger(__name__)

# Stages timed by convert(); the ingest stages are named in ingest.STAGES
STAGE_LOOKUP = "lookup"
STAGE_TRANSFORM = "transform"
STAGE_MERGE = "merge"
STAGE_OBSERVATION_PERIOD = "observation_period"
STAGE_CDM_SOURCE = "cdm_source"
STAGE_ERAS = "eras"
STAGE_CONCEPT = "concept"
STAGE_VOCABULARY_EXPORT = "vocabulary_export"
STAGE_COVERAGE = "coverage"

# Everything convert() puts in the working directory
_WORK_ENTRIES = (
    ingest.RAW_DIR,
    ingest.META_DIR,
    ingest.BATCHES_DIR,
    ingest.GROUPS_DIR,
    ingest.PERSONS_FILE,
    PARTS_DIR,
    SCRATCH_DIR,
    era_builder.ERAS_DIR,
    MANIFEST_FILE,
    MANIFEST_TEMP_FILE,
)


def build_vocab_lookup(
    result: ingest.IngestResult, vocab_dir: Path
) -> VocabularyLookup:
    """Stage 4: map the distinct codes and units of all batches with Athena.

    The codes and units come from streaming scans of every batch file.
    """
    scans = {t: ingest.scan_table(result.work_dir, t) for t in result.tables}
    scans = {t: lf for t, lf in scans.items() if lf is not None}
    return build_lookup(vocab_dir, collect_source_codes(scans), collect_units(scans))


def _remove_work_dir(work_dir: Path) -> None:
    """Delete the pipeline files of work_dir, and work_dir itself when it is empty."""
    for name in _WORK_ENTRIES:
        path = work_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    if work_dir.is_dir() and not any(work_dir.iterdir()):
        work_dir.rmdir()


def _open_work_dir(work_dir: Path, fingerprint: Fingerprint) -> Manifest:
    """Return the manifest of work_dir when it matches this run, else start afresh.

    A missing, unreadable or different manifest means the pipeline files in work_dir
    belong to another run: they are removed (other files are kept) and a manifest
    without finished stages is written.
    """
    manifest = load_manifest(work_dir)
    if matches(manifest, fingerprint):
        return manifest
    if any((work_dir / name).exists() for name in _WORK_ENTRIES):
        logger.info(
            f"Inputs or settings changed, discarding the old files in {work_dir}"
        )
        _remove_work_dir(work_dir)
    manifest = Manifest(PIPELINE_VERSION, fingerprint)
    save_manifest(work_dir, manifest)
    return manifest


def _timing_line(seconds: float, stats: RunStats) -> str:
    """One line with the total time and the process peak memory."""
    line = f"Total time {seconds:.0f} s"
    if stats.process_peak_bytes is not None:
        line += f", peak memory {stats.process_peak_bytes / 1e9:.1f} GB"
    return line


def convert(
    input_zips: list[Path],
    output_dir: Path,
    vocab_dir: Path,
    work_dir: Path | None = None,
    batch_rows: int = ingest.DEFAULT_BATCH_ROWS,
    keep_work_dir: bool = False,
    config_path: Path | None = None,
    export_vocabulary: bool = True,
    eras: bool = True,
) -> dict[str, int]:
    """Convert TriNetX zip file(s) to OMOP parquet files.

    Runs the batched pipeline: ingest (raw Parquet, persons, batches), vocabulary
    lookup, per-batch cleaning and transformation, and the merge of the batch parts.
    A failed run leaves work_dir in place; running the same command again skips the
    finished stages and batches (see manifest.py and the ingest module docstring for the
    working directory layout).

    Args:
        input_zips: List of paths to TriNetX zip files containing CSV exports
        output_dir: Directory to write OMOP parquet files
        vocab_dir: Folder with the Athena vocabulary CSV files
        work_dir: Working directory for intermediate files (default
            <output_dir>/_work). It is deleted after a successful run unless
            keep_work_dir is set, and kept after a failure.
        batch_rows: Upper bound of source rows per batch
        keep_work_dir: Keep work_dir after a successful run
        config_path: JSON file that overrides the bundled concept maps (see
            concept_mappings.load_overrides). The maps are reset to the bundled
            content and the file is applied; they stay changed after the call.
            Without a path the maps are left as they are.
        export_vocabulary: Write the full Athena vocabulary tables (CONCEPT in
            full, CONCEPT_RELATIONSHIP, CONCEPT_ANCESTOR, VOCABULARY, DOMAIN,
            CONCEPT_CLASS, RELATIONSHIP, CONCEPT_SYNONYM, DRUG_STRENGTH) for OHDSI
            tools such as the Data Quality Dashboard (default). When False,
            CONCEPT holds only the referenced concepts. The
            export does not depend on the batches and is not part of the resume
            fingerprint. When False, the exported tables of an earlier run in
            output_dir are removed, like the era tables.
        eras: Write CONDITION_ERA and DRUG_ERA, built from the merged tables
            (see eras.py; default). Needs CONCEPT_ANCESTOR.csv in vocab_dir. They do not
            depend on the batches, so they are always rebuilt and are not part of
            the resume fingerprint. When False, the era tables of an earlier
            run in output_dir are removed, because they would not match the new
            tables.

    Returns:
        Dictionary with row counts for each output table, including the exported
        vocabulary tables and the era tables

    Raises:
        DataDictionaryError: an input ZIP has no readable datadictionary.xlsx, or
            the data dictionaries of the input ZIP files differ (before any work)
        ConfigError: config_path is unreadable or invalid (before any work)
        FileNotFoundError: eras is set and vocab_dir has no CONCEPT_ANCESTOR.csv
            (before any work)
    """
    dictionary = load_data_dictionary(input_zips)
    if eras:
        missing = era_builder.missing_vocabulary_files(vocab_dir)
        if missing:
            raise FileNotFoundError(f"{vocab_dir} is missing: {', '.join(missing)}")
    if config_path is not None:
        overrides = concept_mappings.load_overrides(config_path)
        concept_mappings.reset_mappings()
        concept_mappings.apply_overrides(overrides)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_dir = output_dir / tbl.excluded
    # A previous run may have written files this one does not.
    if excluded_dir.exists():
        shutil.rmtree(excluded_dir)
    excluded_dir.mkdir()
    if not eras:
        for name in era_builder.ERA_TABLES:
            shutil.rmtree(table_path(output_dir, name).parent, ignore_errors=True)
    if not export_vocabulary:
        # The reduced CONCEPT is written again below.
        for table in vocabulary_export.EXPORTED_TABLES:
            if table.athena_stem != tbl.athena_concept:
                path = table_path(output_dir, table.athena_stem).parent
                shutil.rmtree(path, ignore_errors=True)
    work_dir = work_dir if work_dir is not None else output_dir / "_work"
    started_at = datetime.now(UTC)
    started = time.perf_counter()

    manifest = _open_work_dir(
        work_dir, build_fingerprint(input_zips, vocab_dir, batch_rows, config_path)
    )
    stages_done = list(manifest.completed_stages)

    with RunStats() as stats:
        logger.info(f"Reading {len(input_zips)} source(s) into {work_dir}...")
        result = ingest.ingest(
            input_zips,
            work_dir,
            dictionary,
            batch_rows,
            completed=stages_done,
            on_stage_done=lambda stage: mark_stage_done(work_dir, manifest, stage),
            stage_timer=stats.stage,
        )
        logger.info(
            f"  {result.person_count:,} persons in {result.batch_count} batch(es)"
        )

        logger.info(f"Mapping codes with the Athena vocabulary in {vocab_dir}...")
        with stats.stage(STAGE_LOOKUP):
            lookup = build_vocab_lookup(result, vocab_dir)

        first_batch = completed_batches(work_dir, result.batch_count)
        if stages_done or first_batch:
            logger.info(
                f"Resuming from {work_dir}: stages {', '.join(stages_done) or 'none'} "
                f"done; {first_batch} of {result.batch_count} batches done"
            )
        logger.info("Transforming batches...")
        with stats.stage(STAGE_TRANSFORM):
            run_batches(
                result, lookup, counters_from_parts(work_dir, first_batch), first_batch
            )

        logger.info("\nWriting OMOP tables...")
        with stats.stage(STAGE_MERGE):
            row_counts = merge_tables(work_dir, output_dir)
            merge_excluded(work_dir, output_dir, result.tables)

        logger.info("\nCreating observation period and CDM source tables...")
        with stats.stage(STAGE_OBSERVATION_PERIOD):
            observation_df = create_observation_period(output_dir)
        row_counts[tbl.omop_observation_period] = len(observation_df)
        logger.info(f"  {tbl.omop_observation_period}: {len(observation_df):,} rows")

        with stats.stage(STAGE_CDM_SOURCE):
            vocabulary_version = read_vocabulary_version(vocab_dir)
            cdm_source_df = build_cdm_source(
                result.meta_dirs.values(), date.today(), vocabulary_version
            )
            cdm_source_path = table_path(output_dir, tbl.omop_cdm_source)
            cdm_source_path.parent.mkdir(parents=True, exist_ok=True)
            write_parquet(cdm_source_df, cdm_source_path)
        row_counts[tbl.omop_cdm_source] = len(cdm_source_df)
        logger.info(f"  {tbl.omop_cdm_source}: {len(cdm_source_df)} rows")

        if eras:
            logger.info("\nBuilding condition and drug eras...")
            with stats.stage(STAGE_ERAS):
                row_counts.update(
                    era_builder.build_eras(output_dir, vocab_dir, work_dir, batch_rows)
                )

        logger.info("\nCreating concept table...")
        with stats.stage(STAGE_CONCEPT):
            terminology = load_terminology(result.meta_dirs.values())
            if terminology is None:
                logger.info(
                    "  standardized_terminology.csv not found, empty names become codes"
                )
            concept_df = create_concept_table(output_dir, vocab_dir, terminology)
        row_counts[tbl.omop_concept] = len(concept_df)
        logger.info(f"  {tbl.omop_concept}: {len(concept_df):,} rows")

        if export_vocabulary:
            logger.info("\nExporting the full Athena vocabulary tables...")
            with stats.stage(STAGE_VOCABULARY_EXPORT):
                exported = vocabulary_export.export_vocabulary(
                    output_dir, vocab_dir, terminology
                )
            row_counts.update(exported)

        with stats.stage(STAGE_COVERAGE):
            coverage = merge_coverage(work_dir, output_dir)

    if not keep_work_dir:
        _remove_work_dir(work_dir)

    report = build_report(
        started_at=started_at,
        finished_at=datetime.now(UTC),
        input_zips=input_zips,
        vocab_dir=vocab_dir,
        vocabulary_version=vocabulary_version,
        batch_rows=batch_rows,
        batch_count=result.batch_count,
        person_count=result.person_count,
        ingest_stages_skipped=stages_done,
        batches_already_done=first_batch,
        row_counts=row_counts,
        excluded_rows=excluded_row_counts(coverage),
        stats=stats,
        config_path=config_path,
    )
    write_report(output_dir, report)

    logger.info(f"\nConversion complete! Output written to: {output_dir}")
    logger.info(_timing_line(time.perf_counter() - started, stats))
    return row_counts
