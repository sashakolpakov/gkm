"""Content-distinct latent-program/style-pose split contract tests."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_program_split import (  # noqa: E402
    HybridProgramSplit,
    program_digest,
    sample_basic_program_splits,
)


def _dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "downloads" / "Bongard-LOGO"


def test_basic_split_uses_24_content_disjoint_programs_and_round_trips() -> None:
    split = sample_basic_program_splits(
        str(_dataset_root()), limit=1, seed=20260805)[0]
    groups = (
        split.support_pos, split.support_neg,
        split.query_pos, split.query_neg,
    )
    digests = [program_digest(program) for group in groups for program in group]
    assert len(digests) == 24
    assert len(set(digests)) == 24

    manifest = split.to_manifest()
    restored = HybridProgramSplit.from_manifest(manifest)
    assert restored == split
    assert restored.to_manifest() == manifest

    support = restored.render("support", 20260805)
    query = restored.render("query", 20260806)
    assert len(support.pos) == len(support.neg) == 6
    assert len(query.pos) == len(query.neg) == 6
    assert all(panel.dtype == np.uint8 and panel.shape == (128, 128)
               for panel, _label in support.panels() + query.panels())
    assert all(not np.array_equal(left, right)
               for left, right in zip(support.pos + support.neg,
                                      query.pos + query.neg))


def test_program_manifest_rejects_content_and_digest_tampering() -> None:
    split = sample_basic_program_splits(
        str(_dataset_root()), limit=1, seed=20260805)[0]
    manifest = split.to_manifest()

    changed_program = copy.deepcopy(manifest)
    changed_program["query"]["pos"][0]["program"][0][0] += "-tampered"
    with pytest.raises(ValueError, match="digest"):
        HybridProgramSplit.from_manifest(changed_program)

    changed_allocation = copy.deepcopy(manifest)
    changed_allocation["selection"]["query_indices"][0] = 5
    unsigned = {key: value for key, value in changed_allocation.items()
                if key != "program_split_digest"}
    from hybrid_program_split import canonical_digest
    changed_allocation["program_split_digest"] = canonical_digest(unsigned)
    with pytest.raises(ValueError, match="selection"):
        HybridProgramSplit.from_manifest(changed_allocation)
