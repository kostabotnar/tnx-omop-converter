"""Tiny synthetic Athena vocabulary download for tests.

write_athena_fixture(dir) writes tab-separated CONCEPT, CONCEPT_CPT4,
CONCEPT_RELATIONSHIP and VOCABULARY files in the Athena layout. Concept IDs for
TriNetX codes are synthetic and exposed as constants so tests can assert on them.
Config concept IDs (gender, race, visit, route, ...) are the real OMOP IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tnx_omop.util import columns as col
from tnx_omop.util import tables as tbl
from tnx_omop.omop_vocab.athena import CONCEPT_COLUMNS
from tnx_omop.cdm_source import CDM_VERSION_CONCEPT_ID
from tnx_omop.util.concept_mappings import (
    CONDITION_STATUS_CONCEPT_MAP,
    ETHNICITY_CONCEPT_MAP,
    GENDER_CONCEPT_MAP,
    LAB_VALUE_CONCEPT_MAP,
    RACE_CONCEPT_MAP,
    ROUTE_CONCEPT_MAP,
    VISIT_TYPE_CONCEPT_MAP,
)
from tnx_omop.transformers.base import ConceptIds

VOCABULARY_VERSION = "v5.0 TEST-FIXTURE"

# Source concepts (TriNetX codes)
ICD10_E11_65 = 1001  # maps to two Condition concepts
ICD10_Z87_891 = 1002  # maps to Observation
ICD10_C79_51 = 1003  # no Maps to
ICD10_C81_90 = 1004  # maps to Episode (unsupported domain)
ICD10_I10 = 1005  # one valid Maps to, one invalid (ignored)
ICD10_R05 = 1006  # only Maps to a non-standard target
ICD10_K21_0 = 1007  # invalid source concept with a valid Maps to
ICD10_E78_5_INVALID = 1008  # same code as E78_5, invalid, not chosen
ICD10_E78_5 = 1009
ICD10_C91_00 = 1010  # maps to a Condition and an Episode concept
ICD9CM_38_93 = 1101  # same code as ICD9PROC_38_93
ICD9PROC_38_93 = 1102
CPT4_80053 = 1201  # only in CONCEPT_CPT4, empty name, standard Measurement
CPT4_99213 = 1202  # only in CONCEPT_CPT4, empty name, standard Procedure
CPT4_99214 = 1203  # in both CONCEPT and CONCEPT_CPT4
HCPCS_J1100 = 1301  # maps to Drug
HCPCS_E0601 = 1302  # maps to Device
RXNORM_1191 = 1401  # standard, self map
RXNORM_EXT_OMOP123 = 1402  # RxNorm Extension, standard, self map
RXNORM_999999 = 1403  # no Maps to
RXNORM_197361 = 1404
RXNORM_312961 = 1405
LOINC_2160_0 = 1501  # Measurement, self map
LOINC_72166_2 = 1502  # Observation, self map
LOINC_2345_7 = 1503
LOINC_8480_6 = 1504
LOINC_8462_4 = 1505

# Standard targets
COND_T2DM = 2001
COND_HYPERGLYCEMIA = 2002
OBS_EX_SMOKER = 2003
EPISODE_HODGKIN = 2004
COND_HYPERTENSION = 2005
COND_HYPERTENSION_OLD = 2006  # target of the invalid Maps to
COND_COUGH_NONSTD = 2007  # non-standard target
COND_GERD = 2008
COND_HLD_FROM_INVALID = 2009
COND_HLD = 2010
COND_ALL = 2011
EPISODE_ALL = 2012
COND_ICD9_TARGET = 2101
PROC_ICD9_TARGET = 2102
DRUG_DEXAMETHASONE = 2301
DEVICE_CPAP = 2302

# UCUM
UCUM_MG_DL = 8840
UCUM_U_L = 8645
UCUM_MM_HG = 8876
UCUM_PPM_NONSTD = 8999  # non-standard UCUM concept

EHR_TYPE = ConceptIds.EHR_TYPE

_START = "19700101"
_END = "20991231"
_INVALID_END = "20200101"

# concept_id, name, domain, vocabulary, class, standard, code, invalid_reason
ConceptRow = Tuple[int, str, str, str, str, Optional[str], str, Optional[str]]

_CODE_CONCEPTS: List[ConceptRow] = [
    (ICD10_E11_65, "Type 2 DM with hyperglycemia", "Condition", "ICD10CM", "ICD10 code", None, "E11.65", None),
    (ICD10_Z87_891, "Personal history of nicotine dependence", "Observation", "ICD10CM", "ICD10 code", None, "Z87.891", None),
    (ICD10_C79_51, "Secondary malignant neoplasm of bone", "Condition", "ICD10CM", "ICD10 code", None, "C79.51", None),
    (ICD10_C81_90, "Hodgkin lymphoma, unspecified", "Condition", "ICD10CM", "ICD10 code", None, "C81.90", None),
    (ICD10_I10, "Essential (primary) hypertension", "Condition", "ICD10CM", "ICD10 code", None, "I10", None),
    (ICD10_R05, "Cough", "Condition", "ICD10CM", "ICD10 code", None, "R05", None),
    (ICD10_K21_0, "GERD with esophagitis", "Condition", "ICD10CM", "ICD10 code", None, "K21.0", "D"),
    (ICD10_E78_5_INVALID, "Hyperlipidemia (old)", "Condition", "ICD10CM", "ICD10 code", None, "E78.5", "U"),
    (ICD10_E78_5, "Hyperlipidemia, unspecified", "Condition", "ICD10CM", "ICD10 code", None, "E78.5", None),
    (ICD10_C91_00, "Acute lymphoblastic leukemia not in remission", "Condition", "ICD10CM", "ICD10 code", None, "C91.00", None),
    (ICD9CM_38_93, "ICD-9-CM diagnosis 38.93", "Condition", "ICD9CM", "4-dig nonbill code", None, "38.93", None),
    (ICD9PROC_38_93, "Venous catheterization, NEC", "Procedure", "ICD9Proc", "4-dig billing code", None, "38.93", None),
    (CPT4_99214, "Office visit, established patient", "Procedure", "CPT4", "CPT4", "S", "99214", None),
    (HCPCS_J1100, "Injection, dexamethasone sodium phosphate, 1 mg", "Drug", "HCPCS", "HCPCS", None, "J1100", None),
    (HCPCS_E0601, "Continuous positive airway pressure device", "Device", "HCPCS", "HCPCS", None, "E0601", None),
    (RXNORM_1191, "aspirin", "Drug", "RxNorm", "Ingredient", "S", "1191", None),
    (RXNORM_EXT_OMOP123, "Extension drug", "Drug", "RxNorm Extension", "Ingredient", "S", "OMOP123", None),
    (RXNORM_999999, "Unmapped drug", "Drug", "RxNorm", "Ingredient", None, "999999", "D"),
    (RXNORM_197361, "amlodipine 5 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S", "197361", None),
    (RXNORM_312961, "simvastatin 20 MG Oral Tablet", "Drug", "RxNorm", "Clinical Drug", "S", "312961", None),
    (LOINC_2160_0, "Creatinine [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "2160-0", None),
    (LOINC_72166_2, "Tobacco smoking status", "Observation", "LOINC", "Clinical Observation", "S", "72166-2", None),
    (LOINC_2345_7, "Glucose [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "2345-7", None),
    (LOINC_8480_6, "Systolic blood pressure", "Measurement", "LOINC", "Clinical Observation", "S", "8480-6", None),
    (LOINC_8462_4, "Diastolic blood pressure", "Measurement", "LOINC", "Clinical Observation", "S", "8462-4", None),
]  # fmt: skip

_TARGET_CONCEPTS: List[ConceptRow] = [
    (COND_T2DM, "Type 2 diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "44054006", None),
    (COND_HYPERGLYCEMIA, "Hyperglycemia", "Condition", "SNOMED", "Clinical Finding", "S", "80394007", None),
    (OBS_EX_SMOKER, "Ex-smoker", "Observation", "SNOMED", "Context-dependent", "S", "8517006", None),
    (EPISODE_HODGKIN, "Hodgkin lymphoma episode", "Episode", "Cancer Modifier", "Staging/Grading", "S", "OMOP9001", None),
    (COND_HYPERTENSION, "Essential hypertension", "Condition", "SNOMED", "Clinical Finding", "S", "59621000", None),
    (COND_HYPERTENSION_OLD, "Hypertensive disorder (old)", "Condition", "SNOMED", "Clinical Finding", "S", "38341003", None),
    (COND_COUGH_NONSTD, "Cough (non-standard)", "Condition", "SNOMED", "Clinical Finding", None, "49727002", None),
    (COND_GERD, "Gastroesophageal reflux disease with esophagitis", "Condition", "SNOMED", "Clinical Finding", "S", "235595009", None),
    (COND_HLD_FROM_INVALID, "Hyperlipidemia (via invalid source)", "Condition", "SNOMED", "Clinical Finding", "S", "55822004", None),
    (COND_HLD, "Hyperlipidemia", "Condition", "SNOMED", "Clinical Finding", "S", "55822005", None),
    (COND_ALL, "Acute lymphoid leukemia", "Condition", "SNOMED", "Clinical Finding", "S", "91857003", None),
    (EPISODE_ALL, "Acute lymphoid leukemia episode", "Episode", "Cancer Modifier", "Staging/Grading", "S", "OMOP9002", None),
    (COND_ICD9_TARGET, "Condition from ICD9CM 38.93", "Condition", "SNOMED", "Clinical Finding", "S", "900001", None),
    (PROC_ICD9_TARGET, "Venous catheterization", "Procedure", "SNOMED", "Procedure", "S", "392230005", None),
    (DRUG_DEXAMETHASONE, "dexamethasone phosphate 1 MG Injection", "Drug", "RxNorm", "Clinical Drug", "S", "1116927", None),
    (DEVICE_CPAP, "CPAP device", "Device", "SNOMED", "Physical Object", "S", "702172008", None),
]  # fmt: skip

_UCUM_CONCEPTS: List[ConceptRow] = [
    (UCUM_MG_DL, "milligram per deciliter", "Unit", "UCUM", "Unit", "S", "mg/dL", None),
    (UCUM_U_L, "unit per liter", "Unit", "UCUM", "Unit", "S", "[U]/L", None),
    (UCUM_MM_HG, "millimeter mercury column", "Unit", "UCUM", "Unit", "S", "mm[Hg]", None),
    (UCUM_PPM_NONSTD, "parts per million", "Unit", "UCUM", "Unit", None, "[ppm]", None),
]  # fmt: skip

# Only in CONCEPT_CPT4.csv (CPT4_99214 is also in CONCEPT.csv, with a name there).
_CPT4_CONCEPTS: List[ConceptRow] = [
    (CPT4_80053, "", "Measurement", "CPT4", "CPT4", "S", "80053", None),
    (CPT4_99213, "", "Procedure", "CPT4", "CPT4", "S", "99213", None),
    (CPT4_99214, "", "Procedure", "CPT4", "CPT4", "S", "99214", None),
]  # fmt: skip

# Config concept IDs by config map, with the domain each belongs to.
_CONFIG_DOMAINS: List[Tuple[Dict[str, int], str, str]] = [
    (GENDER_CONCEPT_MAP, "Gender", "Gender"),
    (RACE_CONCEPT_MAP, "Race", "Race"),
    (ETHNICITY_CONCEPT_MAP, "Ethnicity", "Ethnicity"),
    (VISIT_TYPE_CONCEPT_MAP, "Visit", "Visit"),
    (CONDITION_STATUS_CONCEPT_MAP, "Condition Status", "Condition Status"),
    (ROUTE_CONCEPT_MAP, "Route", "SNOMED"),
    (LAB_VALUE_CONCEPT_MAP, "Meas Value", "SNOMED"),
]

# concept_id_1, concept_id_2, relationship_id, invalid_reason
RelationshipRow = Tuple[int, int, str, Optional[str]]

_MAPS_TO: List[RelationshipRow] = [
    (ICD10_E11_65, COND_T2DM, "Maps to", None),
    (ICD10_E11_65, COND_HYPERGLYCEMIA, "Maps to", None),
    (ICD10_Z87_891, OBS_EX_SMOKER, "Maps to", None),
    (ICD10_C81_90, EPISODE_HODGKIN, "Maps to", None),
    (ICD10_I10, COND_HYPERTENSION, "Maps to", None),
    (ICD10_I10, COND_HYPERTENSION_OLD, "Maps to", "D"),
    (ICD10_R05, COND_COUGH_NONSTD, "Maps to", None),
    (ICD10_K21_0, COND_GERD, "Maps to", None),
    (ICD10_E78_5_INVALID, COND_HLD_FROM_INVALID, "Maps to", None),
    (ICD10_E78_5, COND_HLD, "Maps to", None),
    (ICD10_C91_00, COND_ALL, "Maps to", None),
    (ICD10_C91_00, EPISODE_ALL, "Maps to", None),
    (ICD9CM_38_93, COND_ICD9_TARGET, "Maps to", None),
    (ICD9PROC_38_93, PROC_ICD9_TARGET, "Maps to", None),
    (CPT4_80053, CPT4_80053, "Maps to", None),
    (CPT4_99213, CPT4_99213, "Maps to", None),
    (CPT4_99214, CPT4_99214, "Maps to", None),
    (HCPCS_J1100, DRUG_DEXAMETHASONE, "Maps to", None),
    (HCPCS_E0601, DEVICE_CPAP, "Maps to", None),
    (RXNORM_1191, RXNORM_1191, "Maps to", None),
    (RXNORM_EXT_OMOP123, RXNORM_EXT_OMOP123, "Maps to", None),
    (RXNORM_197361, RXNORM_197361, "Maps to", None),
    (RXNORM_312961, RXNORM_312961, "Maps to", None),
    (LOINC_2160_0, LOINC_2160_0, "Maps to", None),
    (LOINC_72166_2, LOINC_72166_2, "Maps to", None),
    (LOINC_2345_7, LOINC_2345_7, "Maps to", None),
    (LOINC_8480_6, LOINC_8480_6, "Maps to", None),
    (LOINC_8462_4, LOINC_8462_4, "Maps to", None),
]

_VOCABULARIES: List[str] = [
    "Cancer Modifier",
    "Condition Status",
    "CPT4",
    "Ethnicity",
    "Gender",
    "HCPCS",
    "ICD10CM",
    "ICD9CM",
    "ICD9Proc",
    "LOINC",
    "Metadata",
    "Race",
    "RxNorm",
    "RxNorm Extension",
    "SNOMED",
    "Type Concept",
    "UCUM",
    "Visit",
]


def _config_concepts() -> List[ConceptRow]:
    rows: Dict[int, ConceptRow] = {}
    for mapping, domain, vocabulary in _CONFIG_DOMAINS:
        for name, concept_id in mapping.items():
            if concept_id and concept_id not in rows:
                rows[concept_id] = (
                    concept_id,
                    name,
                    domain,
                    vocabulary,
                    domain,
                    "S",
                    f"CFG{concept_id}",
                    None,
                )
    rows[EHR_TYPE] = (
        EHR_TYPE,
        "EHR",
        "Type Concept",
        "Type Concept",
        "Type Concept",
        "S",
        "OMOP4976890",
        None,
    )
    rows[CDM_VERSION_CONCEPT_ID] = (
        CDM_VERSION_CONCEPT_ID,
        "OMOP CDM Version 5.4.0",
        "Metadata",
        "CDM",
        "CDM",
        "S",
        "CDM v5.4.0",
        None,
    )
    return list(rows.values())


def _concept_line(row: ConceptRow) -> List[str]:
    concept_id, name, domain, vocab, cls, standard, code, invalid = row
    end = _INVALID_END if invalid else _END
    return [
        str(concept_id),
        name,
        domain,
        vocab,
        cls,
        standard or "",
        code,
        _START,
        end,
        invalid or "",
    ]


def _write_tsv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_optional_tables(directory: Path) -> None:
    """Write the vocabulary files that only `--export-vocabulary` reads."""
    _write_tsv(
        directory / f"{tbl.athena_concept_ancestor}.csv",
        [
            col.ancestor_concept_id,
            col.descendant_concept_id,
            col.min_levels_of_separation,
            col.max_levels_of_separation,
        ],
        [
            [str(COND_T2DM), str(COND_T2DM), "0", "0"],
            [str(COND_HYPERGLYCEMIA), str(COND_T2DM), "1", "2"],
            # Ingredients: self rows, and a clinical drug with the ingredient 1191
            [str(RXNORM_1191), str(RXNORM_1191), "0", "0"],
            [str(RXNORM_EXT_OMOP123), str(RXNORM_EXT_OMOP123), "0", "0"],
            [str(RXNORM_1191), str(DRUG_DEXAMETHASONE), "1", "1"],
        ],
    )
    _write_tsv(
        directory / f"{tbl.athena_domain}.csv",
        [col.domain_id, col.domain_name, col.domain_concept_id],
        [["Condition", "Condition", "19"], ["Drug", "Drug", "13"]],
    )
    _write_tsv(
        directory / f"{tbl.athena_concept_class}.csv",
        [
            col.concept_class_id,
            col.concept_class_name,
            col.concept_class_concept_id,
        ],
        [["Clinical Finding", "Clinical Finding", "44819016"]],
    )
    _write_tsv(
        directory / f"{tbl.athena_relationship}.csv",
        [
            col.relationship_id,
            col.relationship_name,
            col.is_hierarchical,
            col.defines_ancestry,
            col.reverse_relationship_id,
            col.relationship_concept_id,
        ],
        [["Maps to", "Maps to", "0", "0", "Mapped from", "44818721"]],
    )
    _write_tsv(
        directory / f"{tbl.athena_concept_synonym}.csv",
        [col.concept_id, col.concept_synonym_name, col.language_concept_id],
        [[str(COND_T2DM), "Type 2 diabetes", "4180186"]],
    )
    # Ingredient strength with an amount, and one with a numerator and denominator
    # (empty fields in the Athena file are null)
    _write_tsv(
        directory / f"{tbl.athena_drug_strength}.csv",
        [
            col.drug_concept_id,
            col.ingredient_concept_id,
            col.amount_value,
            col.amount_unit_concept_id,
            col.numerator_value,
            col.numerator_unit_concept_id,
            col.denominator_value,
            col.denominator_unit_concept_id,
            col.box_size,
            col.valid_start_date,
            col.valid_end_date,
            col.invalid_reason,
        ],
        [
            [str(RXNORM_197361), str(RXNORM_1191), "5", "8576", "", "", "", "", "", _START, _END, ""],
            [str(DRUG_DEXAMETHASONE), str(RXNORM_1191), "", "", "0.5", "8576", "1", "8587", "10", _START, _END, ""],
        ],
    )  # fmt: skip


def write_athena_fixture(directory: Path, optional_tables: bool = True) -> Path:
    """Write the fixture vocabulary files into directory and return it.

    optional_tables=False leaves out the files only the vocabulary export reads
    (CONCEPT_ANCESTOR, DOMAIN, CONCEPT_CLASS, RELATIONSHIP, CONCEPT_SYNONYM,
    DRUG_STRENGTH).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    concepts = _CODE_CONCEPTS + _TARGET_CONCEPTS + _UCUM_CONCEPTS + _config_concepts()
    _write_tsv(
        directory / f"{tbl.athena_concept}.csv",
        CONCEPT_COLUMNS,
        [_concept_line(r) for r in concepts],
    )
    _write_tsv(
        directory / f"{tbl.athena_concept_cpt4}.csv",
        CONCEPT_COLUMNS,
        [_concept_line(r) for r in _CPT4_CONCEPTS],
    )

    relationships: List[List[str]] = []
    for source, target, relationship, invalid in _MAPS_TO:
        end = _INVALID_END if invalid else _END
        relationships.append(
            [str(source), str(target), relationship, _START, end, invalid or ""]
        )
        reverse = "Mapped from"
        relationships.append(
            [str(target), str(source), reverse, _START, end, invalid or ""]
        )
    # Not a Maps to row: must be ignored even though it points to a standard concept.
    relationships.append([str(ICD10_C79_51), str(COND_T2DM), "Is a", _START, _END, ""])
    _write_tsv(
        directory / f"{tbl.athena_concept_relationship}.csv",
        [
            col.concept_id_1,
            col.concept_id_2,
            col.relationship_id,
            col.valid_start_date,
            col.valid_end_date,
            col.invalid_reason,
        ],
        relationships,
    )

    vocab_rows = [
        [
            "None",
            "OMOP Standardized Vocabularies",
            "OMOP generated",
            VOCABULARY_VERSION,
            "44819096",
        ]
    ]
    vocab_rows += [[v, v, "", "", "0"] for v in _VOCABULARIES]
    _write_tsv(
        directory / f"{tbl.athena_vocabulary}.csv",
        [
            col.vocabulary_id,
            "vocabulary_name",
            "vocabulary_reference",
            col.vocabulary_version,
            "vocabulary_concept_id",
        ],
        vocab_rows,
    )
    if optional_tables:
        _write_optional_tables(directory)
    return directory
