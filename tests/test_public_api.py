"""Tests for the names importable from the tnx_omop package."""

import tnx_omop
from tnx_omop import convert, validate
from tnx_omop.pipeline.converter import convert as pipeline_convert
from tnx_omop.quality.validation import validate as quality_validate


def test_convert_and_validate_are_exported():
    assert convert is pipeline_convert
    assert validate is quality_validate
    assert set(tnx_omop.__all__) == {"convert", "validate"}
