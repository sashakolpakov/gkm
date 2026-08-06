from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import pytest

from bongard.artifacts import canonical_json
from bongard.canonical_cache import cached_content_bytes, cached_content_digest
import bongard.semantic_calibration_campaign as campaign_module
from bongard.semantic_calibration_campaign import (
    SemanticCalibrationProposalArchive,
)


@dataclass
class _NumericArtifact:
    value: object

    def _content(self) -> dict[str, object]:
        return {"value": self.value}

    def payload(self) -> bytes:
        return cached_content_bytes(self, (self.value,), self._content)

    def digest(self) -> str:
        return cached_content_digest(self, (self.value,), self._content)


@pytest.mark.parametrize("changed", (True, 1.0))
def test_cache_anchor_distinguishes_json_scalar_types(changed: object) -> None:
    artifact = _NumericArtifact(1)
    original_payload = artifact.payload()
    original_digest = artifact.digest()

    artifact.value = changed
    changed_payload = canonical_json({"value": changed})

    assert changed_payload != original_payload
    assert artifact.payload() == changed_payload
    assert artifact.digest() == hashlib.sha256(changed_payload).hexdigest()
    assert artifact.digest() != original_digest


def test_cache_anchor_rejects_mutable_aliases() -> None:
    artifact = _NumericArtifact(1)
    mutable_anchor = ([1],)

    with pytest.raises(TypeError, match="exact immutable"):
        cached_content_digest(artifact, mutable_anchor, artifact._content)


def test_proposal_archive_anchor_tracks_ambient_historical_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = object.__new__(SemanticCalibrationProposalArchive)
    values = {
        "selection_algorithm": "selection",
        "protocol": SimpleNamespace(
            digest=lambda: "protocol",
            assert_untampered=lambda: None,
        ),
        "execution_config": SimpleNamespace(digest="execution"),
        "selection_seed": "seed",
        "selection_seed_digest": "seed-digest",
        "candidate_count": 6,
        "families": ("bd",),
        "semantic_cohort": "drill",
        "source_corpus_manifest_digest": "source",
        "development_manifest_digest": "development",
        "split_source_digest": "split-source",
        "split_manifest_digest": "split-manifest",
        "historical_seed_digest": "historical",
        "resolver_policy_digest": "resolver",
        "cohort_report_digest": "cohort",
        "clean_cohort_whitelist_digest": "whitelist-digest",
        "clean_cohort_whitelist": (),
        "blocked_policy_digest": "blocked-policy",
        "blocked_exclusion_digest": "blocked-exclusion",
        "blocked_excluded_task_ids": (),
        "blocked_morphology_clusters": (),
        "exposure_predecessor": SimpleNamespace(digest="predecessor"),
        "exposure_successor": SimpleNamespace(digest="successor"),
        "records": (
            SimpleNamespace(
                digest="record",
                assert_matches_protocol=lambda protocol: None,
            ),
        ),
    }
    for name, value in values.items():
        object.__setattr__(archive, name, value)

    monkeypatch.setattr(
        campaign_module,
        "_historical_exposure_cache_identity",
        lambda: "a" * 64,
    )
    first = archive._canonical_anchor()
    monkeypatch.setattr(
        campaign_module,
        "_historical_exposure_cache_identity",
        lambda: "b" * 64,
    )

    assert archive._canonical_anchor() != first

    monkeypatch.setattr(
        SemanticCalibrationProposalArchive,
        "_uncached_content_data",
        lambda self: {"candidate_count": self.candidate_count},
    )
    sealed = archive.digest
    object.__setattr__(archive, "_sealed_digest", sealed)
    object.__setattr__(archive, "candidate_count", 6.0)

    with pytest.raises(
        campaign_module.SemanticCalibrationCampaignError,
        match="changed after sealing",
    ):
        archive.assert_untampered()
