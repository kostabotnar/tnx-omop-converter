"""Measure how well TriNetX codes in TriNetX exports map to the Athena OMOP vocabulary.

Analysis script. It reads the diagnosis, procedure, medication_ingredient, lab_result
and vitals_signs tables from one or more TriNetX ZIP exports, resolves each
(table, code_system, code) and unit with the converter's own vocabulary lookup
(`omop_vocab.vocabulary.build_lookup` and `tnx_omop/config/source_vocabularies.json`), and
writes a Markdown coverage report to stdout and to a file. A code counts as mapped
exactly when the converter writes it to an OMOP table.

Usage:
    uv run tnx-omop coverage --input a.zip b.zip --vocab-dir <athena_dir> [--output report.md]
"""

from __future__ import annotations

import logging
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..domains import NO_SOURCE_CONCEPT
from ..util import columns as c
from ..util import tables as t
from ..util.concept_mappings import SOURCE_VOCABULARY_MAP
from .athena import (
    athena_path,
    scan_athena,
    scan_concepts,
    scan_maps_to,
    unique_concepts,
)
from .vocabulary import (
    RXNORM_EXTENSION,
    UCUM,
    UNIT_TABLES,
    _attach_vocabulary,
    _vocabulary_config,
    build_lookup,
)

logger = logging.getLogger(__name__)

# Internal working columns.
TABLE = c._tnx_table
ROWS = c.rows
SOURCE_INVALID = "source_invalid"
SOURCE_STANDARD = "source_standard"
SOURCE_CONCEPT_NAME = "source_concept_name"
N_TARGETS = "n_targets"
NO_VOCABULARY = "no_vocabulary"

SOURCE_TABLES = [
    t.tnx_diagnosis,
    t.tnx_procedure,
    t.tnx_medication_ingredient,
    t.tnx_lab_result,
    t.tnx_vitals_signs,
]

EXPECTED_DOMAIN = {
    t.tnx_diagnosis: "Condition",
    t.tnx_procedure: "Procedure",
    t.tnx_medication_ingredient: "Drug",
    t.tnx_lab_result: "Measurement",
    t.tnx_vitals_signs: "Measurement",
}

TOP_UNMAPPED = 20
TOP_TEXT_VALUES = 30


@dataclass
class SourceData:
    """Aggregated source data from all input ZIPs."""

    codes: pl.DataFrame  # table, code_system, code, rows
    units: pl.DataFrame  # table, units_of_measure, rows
    text_values: pl.DataFrame  # lab_result_text_val, rows


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def _scan_source(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, infer_schema=False)


def load_source(inputs: list[Path]) -> SourceData:
    """Extract the needed tables from each ZIP and aggregate codes, units and text values."""
    code_parts: list[pl.DataFrame] = []
    unit_parts: list[pl.DataFrame] = []
    text_parts: list[pl.DataFrame] = []
    for zip_path in inputs:
        with (
            tempfile.TemporaryDirectory(prefix="vocab_cov_") as tmp,
            zipfile.ZipFile(zip_path) as zf,
        ):
            names = set(zf.namelist())
            for table in SOURCE_TABLES:
                member = f"{table}.csv"
                if member not in names:
                    logger.warning("%s: %s not found", zip_path.name, member)
                    continue
                logger.info("%s: reading %s", zip_path.name, member)
                csv_path = Path(zf.extract(member, tmp))
                lf = _scan_source(csv_path)
                code_parts.append(
                    lf.group_by(c.code_system, c.code)
                    .agg(pl.len().cast(pl.Int64).alias(ROWS))
                    .with_columns(pl.lit(table).alias(TABLE))
                    .collect(engine="streaming")
                )
                if table in UNIT_TABLES:
                    unit_parts.append(
                        lf.group_by(c.units_of_measure)
                        .agg(pl.len().cast(pl.Int64).alias(ROWS))
                        .with_columns(pl.lit(table).alias(TABLE))
                        .collect(engine="streaming")
                    )
                if table == t.tnx_lab_result:
                    text_parts.append(
                        lf.group_by(c.lab_result_text_val)
                        .agg(pl.len().cast(pl.Int64).alias(ROWS))
                        .collect(engine="streaming")
                    )
                csv_path.unlink()

    def combine(parts: list[pl.DataFrame], keys: list[str]) -> pl.DataFrame:
        return pl.concat(parts, how="diagonal").group_by(keys).agg(pl.col(ROWS).sum())

    return SourceData(
        codes=combine(code_parts, [TABLE, c.code_system, c.code]),
        units=combine(unit_parts, [TABLE, c.units_of_measure]),
        text_values=combine(text_parts, [c.lab_result_text_val]),
    )


# ---------------------------------------------------------------------------
# Vocabulary access
# ---------------------------------------------------------------------------


def available_vocabularies(vocab_dir: Path) -> set[str]:
    return set(
        scan_athena(athena_path(vocab_dir, t.athena_vocabulary))
        .select(c.vocabulary_id)
        .collect()
        .get_column(c.vocabulary_id)
        .to_list()
    )


def count_vocab_concepts(
    concepts: pl.LazyFrame, vocabularies: list[str]
) -> pl.DataFrame:
    return (
        concepts.filter(pl.col(c.vocabulary_id).is_in(vocabularies))
        .group_by(c.vocabulary_id)
        .agg(
            pl.len().alias("concepts"),
            (pl.col(c.concept_name).fill_null("") == "").sum().alias("empty_names"),
        )
        .collect(engine="streaming")
        .sort(c.vocabulary_id)
    )


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def summarize_codes(lookup_codes: pl.DataFrame) -> pl.DataFrame:
    """Reduce the converter's code rows to one row per (table, code_system, code).

    n_targets counts the standard targets the converter writes to an OMOP table.
    off_domain is true when any target lies outside the table's expected domain.
    """
    expected = pl.DataFrame(
        {
            TABLE: list(EXPECTED_DOMAIN),
            "expected_domain": list(EXPECTED_DOMAIN.values()),
        }
    )
    return (
        lookup_codes.join(expected, on=TABLE)
        .group_by(TABLE, c.code_system, c.code)
        .agg(
            pl.col(c.source_concept_id).max(),
            pl.col(c.exclusion_reason).is_null().sum().cast(pl.Int64).alias(N_TARGETS),
            (pl.col(c.domain_id) != pl.col("expected_domain"))
            .any()
            .alias("off_domain"),
            pl.col(c.exclusion_reason).drop_nulls().first(),
        )
    )


def read_source_concepts(vocab_dir: Path, source_ids: list[int]) -> pl.DataFrame:
    """Return name, invalid and standard flags of the given source concepts."""
    ids = pl.LazyFrame(
        {c.concept_id: [str(i) for i in source_ids]}, schema={c.concept_id: pl.String}
    )
    return (
        unique_concepts(scan_concepts(vocab_dir).join(ids, on=c.concept_id, how="semi"))
        .select(
            pl.col(c.concept_id).cast(pl.Int64).alias(c.source_concept_id),
            pl.col(c.concept_name).alias(SOURCE_CONCEPT_NAME),
            (pl.col(c.invalid_reason).fill_null("") != "").alias(SOURCE_INVALID),
            (pl.col(c.standard_concept).fill_null("") == "S").alias(SOURCE_STANDARD),
        )
        .collect(engine="streaming")
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def md_table(df: pl.DataFrame) -> str:
    """Render a DataFrame as a Markdown table."""
    if df.is_empty():
        return "_(none)_\n"

    def cell(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.1f}"
        if isinstance(value, int) and not isinstance(value, bool):
            return f"{value:,}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join([header, sep, *body]) + "\n"


def pct(num: str, den: str) -> pl.Expr:
    return (pl.col(num) / pl.col(den) * 100).round(1)


def build_report(inputs: list[Path], vocab_dir: Path) -> str:
    start = time.perf_counter()
    out: list[str] = ["# TriNetX to OMOP vocabulary coverage\n\n"]
    out.append(
        "Inputs: "
        + ", ".join(p.name for p in inputs)
        + f"  \nVocabulary: `{vocab_dir}`\n\n"
    )

    source = load_source(inputs)
    known_pairs = (
        _vocabulary_config()
        .select(TABLE, c.code_system)
        .with_columns(pl.lit(True).alias("known"))
    )
    codes = (
        _attach_vocabulary(source.codes)
        .join(known_pairs, on=[TABLE, c.code_system], how="left")
        .with_columns(pl.col("known").fill_null(False))
    )

    # --- Code systems present
    pairs = (
        codes.group_by(TABLE, c.code_system, c.vocabulary_id, "known")
        .agg(pl.col(ROWS).sum(), pl.len().alias("distinct_codes"))
        .sort(TABLE, ROWS, descending=[False, True])
    )
    out.append("## Code systems present\n\n")
    out.append(md_table(pairs.rename({"known": "in_mapping_list"})))
    unknown = pairs.filter(~pl.col("known"))
    if unknown.height:
        out.append(
            f"\n**{unknown.height} (table, code_system) pairs are not in the mapping list.**\n"
        )
    else:
        out.append("\nAll (table, code_system) pairs are in the mapping list.\n")
    out.append(
        f"\nRxNorm codes starting with 'OMOP' are looked up in {RXNORM_EXTENSION}, "
        "as the converter does.\n"
    )

    # --- Vocabulary availability
    concepts = scan_concepts(vocab_dir)
    vocab_ids = sorted(
        {v for s in SOURCE_VOCABULARY_MAP.values() for v in s.values() if v is not None}
        | {RXNORM_EXTENSION, UCUM}
    )
    available = available_vocabularies(vocab_dir)
    logger.info("Counting concepts per vocabulary")
    vocab_counts = count_vocab_concepts(concepts, vocab_ids)
    avail_df = pl.DataFrame({c.vocabulary_id: vocab_ids}).with_columns(
        pl.col(c.vocabulary_id).is_in(list(available)).alias("in_VOCABULARY_csv")
    )
    avail_df = avail_df.join(vocab_counts, on=c.vocabulary_id, how="left").with_columns(
        pl.col("concepts").fill_null(0), pl.col("empty_names").fill_null(0)
    )
    out.append("\n## Vocabulary availability\n\n")
    out.append(md_table(avail_df))
    out.append(
        "\nCPT4 concepts are read from CONCEPT_CPT4.csv (not yet merged into CONCEPT.csv).\n"
    )

    # --- Converter lookup
    logger.info("Building the converter vocabulary lookup")
    unit_values = (
        source.units.select(c.units_of_measure)
        .drop_nulls()
        .filter(pl.col(c.units_of_measure) != "")
        .unique()
    )
    lookup = build_lookup(
        vocab_dir,
        source.codes.select(TABLE, c.code_system, c.code).drop_nulls().unique(),
        unit_values,
    )
    per_code = summarize_codes(lookup.codes)
    source_ids = (
        per_code.filter(pl.col(c.source_concept_id) > 0)
        .get_column(c.source_concept_id)
        .unique()
        .to_list()
    )
    logger.info("Reading %s source concepts", f"{len(source_ids):,}")
    source_concepts = read_source_concepts(vocab_dir, source_ids)

    mapped = (
        codes.join(per_code, on=[TABLE, c.code_system, c.code], how="left")
        .join(source_concepts, on=c.source_concept_id, how="left")
        .with_columns(
            pl.col(N_TARGETS).fill_null(0),
            pl.col(c.source_concept_id).fill_null(0),
            pl.col("off_domain").fill_null(False),
            pl.col(SOURCE_INVALID).fill_null(False),
        )
        .with_columns((pl.col(c.source_concept_id) > 0).alias("has_source"))
    )

    # --- Self-mapping check for standard source concepts
    std_sources = source_concepts.filter(pl.col(SOURCE_STANDARD)).get_column(
        c.source_concept_id
    )
    self_maps = (
        scan_maps_to(vocab_dir)
        .filter(
            (pl.col(c.concept_id_1) == pl.col(c.concept_id_2))
            & pl.col(c.concept_id_1).is_in(std_sources.to_list())
        )
        .select(c.concept_id_1)
        .unique()
        .collect(engine="streaming")
        .get_column(c.concept_id_1)
    )
    missing_self = std_sources.filter(~std_sources.is_in(self_maps.to_list()))
    out.append("\n## Standard source concepts mapping to themselves\n\n")
    out.append(
        f"Source concepts that are themselves standard: {std_sources.len():,}. "
        f"With a valid 'Maps to' row to themselves: {std_sources.len() - missing_self.len():,}. "
        f"Without: {missing_self.len():,}.\n"
    )
    if missing_self.len():
        detail = (
            mapped.filter(pl.col(c.source_concept_id).is_in(missing_self.to_list()))
            .select(c.vocabulary_id, c.code, c.source_concept_id, SOURCE_CONCEPT_NAME)
            .unique()
            .head(20)
        )
        out.append("\n" + md_table(detail))

    # --- Coverage summary
    rows_expr = pl.col(ROWS)
    summary = (
        mapped.group_by(TABLE, c.code_system)
        .agg(
            pl.len().alias("codes"),
            pl.col("has_source").sum().alias("codes_src"),
            pl.col(SOURCE_INVALID).sum().alias("codes_src_invalid"),
            (pl.col(N_TARGETS) > 0).sum().alias("codes_std"),
            (pl.col(N_TARGETS) > 1).sum().alias("codes_multi"),
            pl.col("off_domain").sum().alias("codes_off_domain"),
            rows_expr.sum().alias("rows"),
            rows_expr.filter(pl.col("has_source")).sum().alias("rows_src"),
            rows_expr.filter(pl.col(SOURCE_INVALID)).sum().alias("rows_src_invalid"),
            rows_expr.filter(pl.col(N_TARGETS) > 0).sum().alias("rows_std"),
            rows_expr.filter(pl.col(N_TARGETS) > 1).sum().alias("rows_multi"),
            (rows_expr * (pl.col(N_TARGETS) - 1).clip(lower_bound=0))
            .sum()
            .alias("extra_rows"),
            rows_expr.filter(pl.col("off_domain")).sum().alias("rows_off_domain"),
        )
        .sort(TABLE, "rows", descending=[False, True])
    )
    out.append("\n## Coverage by distinct code\n\n")
    out.append(
        "`src` = has a source concept, `src_invalid` = that source concept has invalid_reason set, "
        "`std` = mapped by the converter (at least one standard target in a supported domain), "
        "`multi` = maps to more than one such target, "
        "`off_domain` = at least one standard target outside the table's usual domain.\n\n"
    )
    out.append(
        md_table(
            summary.select(
                TABLE,
                c.code_system,
                "codes",
                "codes_src",
                pct("codes_src", "codes").alias("src_%"),
                "codes_src_invalid",
                "codes_std",
                pct("codes_std", "codes").alias("std_%"),
                "codes_multi",
                "codes_off_domain",
            )
        )
    )
    out.append("\n## Coverage by row\n\n")
    out.append(
        "`extra_rows` = additional rows created because every standard target gets its own row "
        "(sum of rows x (targets - 1)).\n\n"
    )
    out.append(
        md_table(
            summary.select(
                TABLE,
                c.code_system,
                "rows",
                "rows_src",
                pct("rows_src", "rows").alias("src_%"),
                "rows_src_invalid",
                "rows_std",
                pct("rows_std", "rows").alias("std_%"),
                "rows_multi",
                "extra_rows",
                "rows_off_domain",
                pct("rows_off_domain", "rows").alias("off_domain_%"),
            )
        )
    )
    totals = (
        summary.group_by(TABLE)
        .agg(pl.col("rows", "rows_std", "rows_off_domain", "extra_rows").sum())
        .with_columns(pct("rows_std", "rows").alias("std_%"))
        .sort(TABLE)
    )
    out.append("\nTotals per table:\n\n")
    out.append(md_table(totals))

    # --- Domain breakdown
    domain = (
        lookup.codes.filter(pl.col(c.domain_id).is_not_null())
        .join(source.codes, on=[TABLE, c.code_system, c.code])
        .join(
            pl.DataFrame(
                {
                    TABLE: list(EXPECTED_DOMAIN),
                    "expected_domain": list(EXPECTED_DOMAIN.values()),
                }
            ),
            on=TABLE,
        )
        .group_by(TABLE, c.code_system, "expected_domain", c.domain_id)
        .agg(pl.col(c.code).n_unique().alias("codes"), pl.col(ROWS).sum().alias("rows"))
        .sort(TABLE, c.code_system, "rows", descending=[False, False, True])
    )
    out.append("\n## Standard target domains\n\n")
    out.append(
        "One line per (table, code_system, target domain). Counts are code to target pairs, so a code "
        "mapping to two domains is counted under both, and `rows` is source rows per target.\n\n"
    )
    out.append(md_table(domain))

    # --- Multi-target examples
    multi = (
        mapped.filter(pl.col(N_TARGETS) > 1)
        .sort(ROWS, descending=True)
        .select(TABLE, c.code_system, c.code, SOURCE_CONCEPT_NAME, N_TARGETS, ROWS)
        .head(15)
    )
    out.append("\n## Top codes mapping to more than one standard concept\n\n")
    out.append(md_table(multi))

    # --- Unmapped codes
    out.append(f"\n## Top {TOP_UNMAPPED} unmapped codes per table\n\n")
    out.append(
        "Unmapped = not written to an OMOP table by the converter. `reason`: no_vocabulary (code "
        "system has no OMOP vocabulary or it is missing from the download), then the converter's "
        "exclusion reasons: no_source_concept, no_standard_mapping, unsupported_domain.\n"
    )
    unmapped = mapped.filter(pl.col(N_TARGETS) == 0).with_columns(
        pl.when(
            pl.col(c.vocabulary_id).is_null()
            | ~pl.col(c.vocabulary_id).is_in(list(available))
        )
        .then(pl.lit(NO_VOCABULARY))
        .otherwise(pl.col(c.exclusion_reason).fill_null(NO_SOURCE_CONCEPT))
        .alias("reason")
    )
    for table in SOURCE_TABLES:
        sub = unmapped.filter(pl.col(TABLE) == table)
        out.append(f"\n### {table}\n\n")
        out.append(
            f"Unmapped distinct codes: {sub.height:,}, rows: {sub.get_column(ROWS).sum():,}. "
            "By reason: "
            + ", ".join(
                f"{r}: {n:,} codes / {rows:,} rows"
                for r, n, rows in sub.group_by("reason")
                .agg(pl.len(), pl.col(ROWS).sum())
                .sort("reason")
                .iter_rows()
            )
            + "\n\n"
        )
        out.append(
            md_table(
                sub.sort(ROWS, descending=True)
                .select(
                    c.code_system,
                    c.code,
                    ROWS,
                    "reason",
                    c.source_concept_id,
                    SOURCE_CONCEPT_NAME,
                )
                .head(TOP_UNMAPPED)
            )
        )

    # --- Units
    units = (
        source.units.with_columns(pl.col(c.units_of_measure).fill_null(""))
        .join(lookup.units, on=c.units_of_measure, how="left")
        .with_columns(
            (pl.col(c.unit_concept_id).fill_null(0) > 0).alias("ucum"),
            (pl.col(c.units_of_measure) == "").alias("empty"),
        )
    )
    unit_summary = (
        units.group_by(TABLE)
        .agg(
            pl.len().alias("distinct_units"),
            pl.col("empty").sum().alias("distinct_empty"),
            pl.col("ucum").sum().alias("distinct_ucum"),
            pl.col(ROWS).sum().alias("rows"),
            pl.col(ROWS).filter(pl.col("empty")).sum().alias("rows_empty"),
            pl.col(ROWS).filter(pl.col("ucum")).sum().alias("rows_ucum"),
        )
        .with_columns(pct("rows_ucum", "rows").alias("ucum_%"))
        .sort(TABLE)
    )
    out.append("\n## Measurement units vs UCUM\n\n")
    out.append(
        "`ucum` = the converter maps the unit to a standard UCUM concept (exact code match or a "
        "configured override). Empty unit values are counted in `empty`.\n\n"
    )
    out.append(md_table(unit_summary))
    top_units = (
        units.group_by(c.units_of_measure, "ucum")
        .agg(pl.col(ROWS).sum())
        .sort(ROWS, descending=True)
    )
    out.append("\nTop 30 units by rows (both tables):\n\n")
    out.append(md_table(top_units.head(30)))
    out.append("\nTop 30 unmatched non-empty units by rows:\n\n")
    out.append(
        md_table(
            top_units.filter(~pl.col("ucum") & (pl.col(c.units_of_measure) != "")).head(
                30
            )
        )
    )

    # --- Text results
    text = source.text_values.with_columns(pl.col(c.lab_result_text_val).fill_null(""))
    non_empty = text.filter(pl.col(c.lab_result_text_val) != "")
    out.append(f"\n## lab_result text values (top {TOP_TEXT_VALUES})\n\n")
    out.append(
        f"Rows with a text value: {non_empty.get_column(ROWS).sum():,} of {text.get_column(ROWS).sum():,}. "
        f"Distinct non-empty text values: {non_empty.height:,}.\n\n"
    )
    out.append(md_table(non_empty.sort(ROWS, descending=True).head(TOP_TEXT_VALUES)))

    out.append(f"\n_Runtime: {time.perf_counter() - start:.0f} s_\n")
    return "".join(out)
