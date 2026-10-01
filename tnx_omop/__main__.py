"""Command line entry point: `tnx-omop` and `python -m tnx_omop`."""

import argparse
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from . import cli, dqd
from . import eras as era_builder
from .converter import convert
from .example import write_example
from .omop_vocab.athena import missing_files
from .util import concept_mappings
from .omop_vocab.vocab_coverage import build_report
from .schema.trinetx import DataDictionaryError, load_data_dictionary
from .validation import ValidationReport, coverage_report, validate

logger = logging.getLogger(__name__)

_handler: logging.Handler | None = None

# Exit code for a missing subcommand, the same as an argparse usage error
EXIT_USAGE = 2


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Send log messages to stderr, message only: INFO, or DEBUG or WARNING.

    Only the handler installed by an earlier call is replaced, so handlers added by
    others (for example a test harness) stay.
    """
    global _handler
    root = logging.getLogger()
    if _handler is not None:
        root.removeHandler(_handler)
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(_handler)
    root.setLevel(
        logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    )


def _check_inputs(
    inputs: Sequence[Path],
    vocab_dir: Path,
    eras: bool = False,
    data_dictionary: bool = False,
) -> bool:
    """Log an error and return False when the ZIP files or the vocabulary are unusable.

    eras also requires the Athena files that the era tables need. data_dictionary
    also requires a data dictionary in every ZIP, the same in all of them.
    """
    for input_path in inputs:
        if not input_path.exists():
            logger.error(f"Error: Input file not found: {input_path}")
            return False
        if input_path.suffix != ".zip":
            logger.error(f"Error: Input file must be a .zip file: {input_path}")
            return False

    # Duplicate ZIP stems: patients of two sources would merge
    stems = [p.stem for p in inputs]
    if len(stems) != len(set(stems)):
        duplicates = [s for s in stems if stems.count(s) > 1]
        logger.error(f"Error: Duplicate ZIP file names: {set(duplicates)}")
        logger.error(
            "Each ZIP file must have a unique name (stem): it identifies the "
            "source of each patient and encounter."
        )
        return False

    if data_dictionary:
        try:
            load_data_dictionary(inputs)
        except DataDictionaryError as e:
            logger.error(f"Error: {e}")
            return False

    if not vocab_dir.is_dir():
        logger.error(f"Error: Vocabulary folder not found: {vocab_dir}")
        return False
    missing = missing_files(vocab_dir)
    if missing:
        logger.error(
            f"Error: Vocabulary folder {vocab_dir} is missing: {', '.join(missing)}"
        )
        return False
    missing = era_builder.missing_vocabulary_files(vocab_dir) if eras else []
    if missing:
        logger.error(
            f"Error: Vocabulary folder {vocab_dir} is missing {', '.join(missing)}, "
            "which the era tables need; add it or give --no-eras"
        )
        return False
    return True


def _check_config(config_path: Path | None) -> bool:
    """Log an error and return False when the config file is unreadable or invalid."""
    if config_path is None:
        return True
    try:
        concept_mappings.load_overrides(config_path)
    except concept_mappings.ConfigError as e:
        logger.error(f"Error: {e}")
        return False
    return True


def _run_convert(args: argparse.Namespace) -> int:
    if not _check_inputs(
        args.input, args.vocab_dir, args.eras, data_dictionary=True
    ) or not _check_config(args.config):
        return 1
    convert(
        args.input,
        args.output,
        args.vocab_dir,
        work_dir=args.work_dir,
        batch_rows=args.batch_rows,
        keep_work_dir=args.keep_work_dir,
        config_path=args.config,
        export_vocabulary=args.export_vocabulary,
        eras=args.eras,
    )
    return 0


def _format_summary(report: ValidationReport) -> str:
    """One line per issue of each failed check, then the counts."""
    lines = [
        f"FAILED {result.name}: {issue}"
        for result in report.failed
        for issue in result.issues
    ]
    lines.append(
        f"{len(report.passed)} passed, {len(report.failed)} failed, "
        f"{len(report.skipped)} skipped"
    )
    return "\n".join(lines)


def _run_validate(args: argparse.Namespace) -> int:
    if not args.output_dir.is_dir():
        logger.error(f"Error: Output folder not found: {args.output_dir}")
        return 1
    report = validate(args.output_dir)
    print(_format_summary(report))
    if args.coverage:
        print("\n" + coverage_report(args.output_dir))
    return 0 if report.ok else 1


def _run_coverage(args: argparse.Namespace) -> int:
    if not _check_inputs(args.input, args.vocab_dir) or not _check_config(args.config):
        return 1
    if args.config is not None:
        concept_mappings.reset_mappings()
        concept_mappings.apply_overrides(concept_mappings.load_overrides(args.config))
    report = build_report(args.input, args.vocab_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stdout.write(report)
    logger.info("Report written to %s", args.output)
    return 0


def _run_dqd(args: argparse.Namespace) -> int:
    if args.install_r_packages:
        rscript = args.rscript or shutil.which("Rscript")
        if rscript is None:
            logger.error(
                "Error: Rscript not found on PATH; install R or give --rscript"
            )
            return 1
        return dqd.install_r_packages(rscript)
    if args.output_dir is None:
        logger.error("Error: Output folder missing (or give --install-r-packages)")
        return 1
    if not args.output_dir.is_dir():
        logger.error(f"Error: Output folder not found: {args.output_dir}")
        return 1
    results_dir = args.results_dir or args.output_dir / dqd.RESULTS_DIR
    try:
        accepted = dqd.load_accepted(args.accepted)
        if args.no_run:
            results_path = results_dir / dqd.RESULTS_FILE
            if not results_path.exists():
                logger.error(f"Error: No DQD results in {results_dir}")
                return 1
        else:
            rscript = args.rscript or shutil.which("Rscript")
            if rscript is None:
                logger.error(
                    "Error: Rscript not found on PATH; install R or give --rscript"
                )
                return 1
            results_path = dqd.run_dqd(
                args.output_dir,
                results_dir,
                rscript,
                source_name=args.source_name,
                keep_database=args.keep_database,
            )
        summary = dqd.summarize(dqd.load_results(results_path), accepted)
    except dqd.DqdError as e:
        logger.error(f"Error: {e}")
        return 1
    for pattern in summary.unused_patterns:
        logger.warning("Accepted DQD check matches no failure: %s", pattern)
    print(dqd.format_summary(summary))
    logger.info("DQD results: %s", results_path)
    return 0 if summary.ok else 1


def _run_example(args: argparse.Namespace) -> int:
    zip_path, athena_dir = write_example(args.output_dir)
    logger.info("Wrote %s and %s", zip_path, athena_dir)
    return 0


_COMMANDS = {
    cli.CONVERT: _run_convert,
    cli.VALIDATE: _run_validate,
    cli.COVERAGE: _run_coverage,
    cli.DQD: _run_dqd,
    cli.EXAMPLE: _run_example,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the subcommand named in the arguments.

    Args:
        argv: Arguments without the program name; defaults to sys.argv

    Returns:
        Process exit code: 0 on success, 1 on an input error or a failed
        validation, 2 when no subcommand is given
    """
    args = cli.parse_args(argv)
    configure_logging(args.verbose, args.quiet)
    if args.command is None:
        cli.build_parser().print_help(sys.stderr)
        return EXIT_USAGE
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
