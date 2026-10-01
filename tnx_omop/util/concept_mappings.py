"""Load OMOP concept mappings and source vocabulary config from JSON."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import polars as pl

_CONFIG_DIR = Path(__file__).parents[1] / "config"
_CONFIG_PATH = _CONFIG_DIR / "concept_mappings.json"
_SOURCE_VOCABULARIES_PATH = _CONFIG_DIR / "source_vocabularies.json"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_config() -> dict:
    return _load_json(_CONFIG_PATH)


_config = _load_config()

GENDER_CONCEPT_MAP: dict[str, int] = _config["gender"]
RACE_CONCEPT_MAP: dict[str, int] = _config["race"]
ETHNICITY_CONCEPT_MAP: dict[str, int] = _config["ethnicity"]
VISIT_TYPE_CONCEPT_MAP: dict[str, int] = _config["visit_type"]
CONDITION_STATUS_CONCEPT_MAP: dict[str, int] = _config["condition_status"]
ROUTE_CONCEPT_MAP: dict[str, int] = _config["route"]
UNIT_UCUM_MAP: dict[str, str] = _config["unit_ucum"]
LAB_VALUE_CONCEPT_MAP: dict[str, int] = _config["lab_value"]
DEFAULT_CONCEPT_ID: int = _config["default_concept_id"]

# TriNetX table -> code_system -> Athena vocabulary_id (None: no OMOP vocabulary).
SOURCE_VOCABULARY_MAP: dict[str, dict[str, Optional[str]]] = _load_json(
    _SOURCE_VOCABULARIES_PATH
)

_CONCEPT_MAPS: dict[str, dict[str, int]] = {
    "gender": GENDER_CONCEPT_MAP,
    "race": RACE_CONCEPT_MAP,
    "ethnicity": ETHNICITY_CONCEPT_MAP,
    "visit_type": VISIT_TYPE_CONCEPT_MAP,
    "condition_status": CONDITION_STATUS_CONCEPT_MAP,
    "route": ROUTE_CONCEPT_MAP,
    "lab_value": LAB_VALUE_CONCEPT_MAP,
}
SOURCE_VOCABULARIES = "source_vocabularies"
UNIT_UCUM = "unit_ucum"
OVERRIDE_KEYS = (*_CONCEPT_MAPS, UNIT_UCUM, SOURCE_VOCABULARIES)

# Overrides update the dicts above in place, not rebind them: other modules bind these
# names with `from ... import`, so a new dict object would not reach them.
_BUNDLED = copy.deepcopy(
    {
        **_CONCEPT_MAPS,
        UNIT_UCUM: UNIT_UCUM_MAP,
        SOURCE_VOCABULARIES: SOURCE_VOCABULARY_MAP,
    }
)


class ConfigError(ValueError):
    """The user supplied config file is unreadable or invalid."""


def _check_map(key: str, value: Any, value_type: type, type_name: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"Config key '{key}' must be a JSON object")
    for name, entry in value.items():
        # bool is an int subclass but never a concept ID
        if not isinstance(entry, value_type) or isinstance(entry, bool):
            raise ConfigError(
                f"Config key '{key}', entry '{name}': expected {type_name}, "
                f"got {json.dumps(entry)}"
            )


def _check_source_vocabularies(value: Any) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"Config key '{SOURCE_VOCABULARIES}' must be a JSON object")
    for table, systems in value.items():
        if not isinstance(systems, dict):
            raise ConfigError(
                f"Config key '{SOURCE_VOCABULARIES}', table '{table}': expected a "
                "JSON object of code system to vocabulary"
            )
        for system, vocabulary in systems.items():
            if vocabulary is not None and not isinstance(vocabulary, str):
                raise ConfigError(
                    f"Config key '{SOURCE_VOCABULARIES}', table '{table}', code "
                    f"system '{system}': expected string or null, "
                    f"got {json.dumps(vocabulary)}"
                )


def load_overrides(path: Path) -> dict[str, Any]:
    """Read and validate a user config file with overrides of the bundled maps.

    Args:
        path: JSON file with any subset of the keys gender, race, ethnicity,
            visit_type, condition_status, route, lab_value (value to integer concept
            ID), unit_ucum (unit to UCUM code) and source_vocabularies (TriNetX table
            to code system to Athena vocabulary ID or null).

    Returns:
        The parsed content.

    Raises:
        ConfigError: The file is unreadable or not JSON, or has an unknown key or a
            value of the wrong type. The message names the key.
    """
    try:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"Cannot read config file {path}: {e}") from e
    except ValueError as e:
        raise ConfigError(f"Config file {path} is not valid JSON: {e}") from e
    if not isinstance(overrides, dict):
        raise ConfigError(f"Config file {path} must contain a JSON object")
    for key, value in overrides.items():
        if key in _CONCEPT_MAPS:
            _check_map(key, value, int, "integer concept ID")
        elif key == UNIT_UCUM:
            _check_map(key, value, str, "string")
        elif key == SOURCE_VOCABULARIES:
            _check_source_vocabularies(value)
        else:
            raise ConfigError(
                f"Unknown config key '{key}'; allowed: {', '.join(OVERRIDE_KEYS)}"
            )
    return overrides


def describe_config(path: Path | None) -> dict[str, str] | None:
    """Absolute path and SHA-256 (hex) of the config file, or None without one."""
    if path is None:
        return None
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def apply_overrides(overrides: dict[str, Any]) -> None:
    """Merge validated overrides into the bundled maps, in place.

    Entries are added or replaced and the others stay. source_vocabularies merges per
    TriNetX table, then per code system; null keeps its meaning (no OMOP vocabulary).
    """
    for key, value in overrides.items():
        if key in _CONCEPT_MAPS:
            _CONCEPT_MAPS[key].update(value)
        elif key == UNIT_UCUM:
            UNIT_UCUM_MAP.update(value)
        else:
            for table, systems in value.items():
                SOURCE_VOCABULARY_MAP.setdefault(table, {}).update(systems)


def reset_mappings() -> None:
    """Restore every map to the bundled content, in place."""
    targets = {
        **_CONCEPT_MAPS,
        UNIT_UCUM: UNIT_UCUM_MAP,
        SOURCE_VOCABULARIES: SOURCE_VOCABULARY_MAP,
    }
    for key, target in targets.items():
        target.clear()
        target.update(copy.deepcopy(_BUNDLED[key]))


def map_concept_id(source_col: str, mapping: dict[str, int]) -> pl.Expr:
    """Map a source column to concept IDs with a dictionary.

    Args:
        source_col: Name of the source column to map from
        mapping: Dictionary mapping source values to concept IDs

    Returns:
        Int64 expression with the concept ID of each value; null and values
        missing from mapping give DEFAULT_CONCEPT_ID
    """
    return (
        pl.col(source_col)
        .cast(pl.Utf8)
        .replace_strict(mapping, default=DEFAULT_CONCEPT_ID, return_dtype=pl.Int64)
    )
