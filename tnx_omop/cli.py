"""CLI argument parsing for the `tnx-omop` command and its subcommands."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import dqd
from .ingest import DEFAULT_BATCH_ROWS

CONVERT = "convert"
VALIDATE = "validate"
COVERAGE = "coverage"
DQD = "dqd"
EXAMPLE = "example"
DEFAULT_COVERAGE_OUTPUT = Path("build") / "vocab_coverage_report.md"


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "JSON file that adds or replaces entries of the bundled concept maps "
            "(see README.md)"
        ),
    )


def _add_convert_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        CONVERT,
        help="Convert TriNetX ZIP exports to OMOP CDM 5.4 Parquet files",
        description="Convert TriNetX EHR data to OMOP CDM 5.4 format",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to TriNetX zip file(s) containing CSV exports",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for OMOP parquet files",
    )

    parser.add_argument(
        "--vocab-dir",
        type=Path,
        required=True,
        help=(
            "Folder with the Athena OMOP vocabulary download (CONCEPT.csv, "
            "CONCEPT_RELATIONSHIP.csv, VOCABULARY.csv, CONCEPT_ANCESTOR.csv unless "
            "--no-eras, optional CONCEPT_CPT4.csv)"
        ),
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Folder for intermediate files (default: <output>/_work). Deleted after "
            "a successful run unless --keep-work-dir is given, kept after a failure"
        ),
    )

    parser.add_argument(
        "--batch-rows",
        type=_positive_int,
        default=DEFAULT_BATCH_ROWS,
        help=(
            "Upper bound of source rows per batch of patients "
            f"(default: {DEFAULT_BATCH_ROWS:,}); smaller batches use less memory"
        ),
    )

    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the working folder after a successful run",
    )
    # Both default to on so that the output is complete for OHDSI tools such as
    # the Data Quality Dashboard (see dqd.py).
    parser.add_argument(
        "--export-vocabulary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write the full Athena vocabulary tables (CONCEPT, "
            "CONCEPT_RELATIONSHIP, CONCEPT_ANCESTOR, VOCABULARY, DOMAIN, "
            "CONCEPT_CLASS, RELATIONSHIP, CONCEPT_SYNONYM, DRUG_STRENGTH) for "
            "OHDSI tools (default: on). --no-export-vocabulary writes only the "
            "referenced concepts and saves disk space and run time"
        ),
    )
    parser.add_argument(
        "--eras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write CONDITION_ERA and DRUG_ERA (OHDSI standard logic, 30 day "
            "persistence window; default: on); needs CONCEPT_ANCESTOR.csv in "
            "--vocab-dir, --no-eras skips them"
        ),
    )
    _add_config_argument(parser)


def _add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        VALIDATE,
        help="Check a converted output folder against the OMOP CDM 5.4 schema",
        description=(
            "Run the OMOP checks on an output folder. Exit code 0 when every check "
            "passes, 1 otherwise."
        ),
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output folder written by `tnx-omop convert`"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Also print rows per OMOP table and excluded rows per reason",
    )


def _add_coverage_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        COVERAGE,
        help="Report how well TriNetX codes map to the Athena vocabulary",
        description=(
            "Measure how well the codes in TriNetX exports map to the Athena OMOP "
            "vocabulary, using the converter's own lookup. Writes a Markdown report "
            "to stdout and to --output."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        nargs="+",
        required=True,
        help="TriNetX ZIP exports",
    )
    parser.add_argument(
        "--vocab-dir", type=Path, required=True, help="Athena vocabulary folder"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_COVERAGE_OUTPUT,
        help="Markdown report path (default: %(default)s)",
    )
    _add_config_argument(parser)


def _add_dqd_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        DQD,
        help="Run the OHDSI Data Quality Dashboard checks on an output folder",
        description=(
            "Load an output folder into a DuckDB database with the official OMOP CDM "
            "5.4 tables and run the OHDSI Data Quality Dashboard (DQD) checks with R. "
            "Needs Rscript and the R packages that `--install-r-packages` installs. Exit "
            "code 0 when every failed check is accepted, 1 otherwise."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Output folder written by `tnx-omop convert` (not needed with "
            "--install-r-packages)"
        ),
    )
    parser.add_argument(
        "--install-r-packages",
        action="store_true",
        help=(
            "Install the R packages that DQD needs (runs the bundled "
            "install_packages.R) and exit with its exit code; no output folder needed"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=f"Folder for the DQD results (default: <output_dir>/{dqd.RESULTS_DIR})",
    )
    parser.add_argument(
        "--rscript",
        default=None,
        help="Rscript executable (default: Rscript found on PATH)",
    )
    parser.add_argument(
        "--accepted",
        type=Path,
        default=dqd.ACCEPTED_FILE,
        help=(
            "JSON file of accepted failed checks with their reasons "
            "(default: the bundled tnx_omop/config/dqd_accepted.json)"
        ),
    )
    parser.add_argument(
        "--source-name",
        default=dqd.DEFAULT_SOURCE_NAME,
        help="Source name in the DQD results (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-database",
        action="store_true",
        help=f"Keep the DuckDB file ({dqd.DATABASE_FILE}) in the results folder",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Summarize the results of an earlier run without running R",
    )


def _add_example_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        EXAMPLE,
        help="Write a small synthetic export and vocabulary to try the tool",
        description=(
            "Write example_export.zip (a tiny synthetic TriNetX export) and an athena "
            "folder (a tiny vocabulary, not an Athena download) into the output "
            "folder. Both work with `tnx-omop convert` and its default options."
        ),
    )
    parser.add_argument("output_dir", type=Path, help="Folder to write the files into")


def build_parser() -> argparse.ArgumentParser:
    """Build the `tnx-omop` parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog="tnx-omop",
        description="Convert TriNetX EHR data to OMOP CDM 5.4 format",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log debug messages (give before the command)",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Log warnings and errors only (give before the command)",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_convert_parser(subparsers)
    _add_validate_parser(subparsers)
    _add_coverage_parser(subparsers)
    _add_dqd_parser(subparsers)
    _add_example_parser(subparsers)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Arguments without the program name; defaults to sys.argv

    Returns:
        Namespace with `verbose` and `quiet`, `command` (None when no subcommand
        is given) and the arguments of that subcommand
    """
    return build_parser().parse_args(argv)
