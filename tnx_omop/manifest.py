"""Run manifest of the working directory, used to resume an interrupted run.

<work>/manifest.json records a pipeline version, a fingerprint of the inputs and the
ingest stages that finished. A later run resumes from the working directory only when
the version and the fingerprint are equal; otherwise the pipeline files are discarded.
Finished batches are not recorded here: the coverage part of a batch is its completion
marker (see batch_transform). The lookup, terminology, CDM_SOURCE and the merge are
rebuilt on every run. The file is written atomically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .omop_vocab.athena import athena_path
from .util.concept_mappings import describe_config
from .util import tables as tbl

# Increase when a code change makes the files of an older working directory invalid
PIPELINE_VERSION = 2

MANIFEST_FILE = "manifest.json"
MANIFEST_TEMP_FILE = "manifest.json.tmp"

# Athena files the converter reads
_VOCAB_FILES = (
    tbl.athena_concept,
    tbl.athena_concept_cpt4,
    tbl.athena_concept_relationship,
    tbl.athena_vocabulary,
)

Fingerprint = dict[str, Any]


@dataclass
class Manifest:
    """Content of manifest.json.

    Attributes:
        version: Pipeline version that wrote the working directory.
        fingerprint: Inputs of the run, see build_fingerprint.
        completed_stages: Ingest stages that finished, in order.
    """

    version: int
    fingerprint: Fingerprint
    completed_stages: list[str] = field(default_factory=list)


def _file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_fingerprint(
    input_zips: list[Path],
    vocab_dir: Path,
    batch_rows: int,
    config_path: Path | None = None,
) -> Fingerprint:
    """Describe the inputs of a run.

    Args:
        input_zips: TriNetX ZIP files in the given order.
        vocab_dir: Folder with the Athena files.
        batch_rows: Upper bound of source rows per batch.
        config_path: User config file with concept map overrides, or None.

    Returns:
        A JSON compatible dict with path, size and modification time of each ZIP,
        the same for each Athena file the converter reads (CPT4 only when present),
        batch_rows, and path and SHA-256 of the config file (null without one).
    """
    return {
        "inputs": [{"path": str(p.resolve()), **_file_state(p)} for p in input_zips],
        "vocabulary": {
            "path": str(vocab_dir.resolve()),
            "files": {
                athena_path(vocab_dir, stem).name: _file_state(
                    athena_path(vocab_dir, stem)
                )
                for stem in _VOCAB_FILES
                if athena_path(vocab_dir, stem).is_file()
            },
        },
        "batch_rows": batch_rows,
        "config": describe_config(config_path),
    }


def load_manifest(work_dir: Path) -> Manifest | None:
    """Read <work>/manifest.json, or None when it is missing or unreadable."""
    try:
        data = json.loads((work_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        stages = data["completed_stages"]
        if not isinstance(stages, list) or not all(isinstance(s, str) for s in stages):
            return None
        return Manifest(int(data["version"]), dict(data["fingerprint"]), stages)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_manifest(work_dir: Path, manifest: Manifest) -> None:
    """Write <work>/manifest.json atomically (temporary file, then os.replace)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    temp = work_dir / MANIFEST_TEMP_FILE
    temp.write_text(
        json.dumps(
            {
                "version": manifest.version,
                "fingerprint": manifest.fingerprint,
                "completed_stages": manifest.completed_stages,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temp, work_dir / MANIFEST_FILE)


def matches(manifest: Manifest | None, fingerprint: Fingerprint) -> bool:
    """True when manifest was written by this pipeline version for these inputs."""
    return (
        manifest is not None
        and manifest.version == PIPELINE_VERSION
        and manifest.fingerprint == fingerprint
    )


def mark_stage_done(work_dir: Path, manifest: Manifest, stage: str) -> None:
    """Record a finished ingest stage in manifest and on disk."""
    if stage not in manifest.completed_stages:
        manifest.completed_stages.append(stage)
    save_manifest(work_dir, manifest)
