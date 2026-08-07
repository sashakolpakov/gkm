from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from typing import cast
import zipfile

import pytest

from bongard import prototype_pair_campaign_cli as campaign_cli
from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger, semantic_resolver_policy_digest
from bongard.historical_exposure import DEFAULT_SEED_PATH, load_historical_exposure
from bongard.official_panel_archive import OfficialPanelArchive, ReleasedOfficialPanel
from bongard.prototype_pair_campaign import run_prototype_pair_campaign
from bongard.prototype_pair_campaign_store import PrototypePairCampaignStore
from bongard.prototype_pair_campaign_cli import (
    DEFAULT_ABSENT_UPPER_PPM,
    DEFAULT_PRESENT_LOWER_PPM,
    PythonRuntimeIdentity,
    dispatch_prepared_prototype_pair_campaign,
    prepare_prototype_pair_campaign_launch,
    verify_prototype_pair_campaign_metadata,
)
from bongard.prototype_pair_cohort import (
    BIRD_FAMILIES,
    OFFICIAL_BASIC_GENERATOR_SHA256,
    OFFICIAL_BASIC_SAMPLER_SHA256,
    OFFICIAL_UPSTREAM_COMMIT,
    OFFICIAL_UPSTREAM_REPOSITORY,
    plan_prototype_pair_cohort,
    prototype_pair_seed_commitment,
    task_id_inventory_digest,
)
from bongard.prototype_pair_execution_precommit import (
    PrototypePairExecutionPrecommit,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.release import OfficialReleaseDescriptor
from bongard.transport import CloudPolicyCacheSnapshot


_SEED = "metadata-only campaign launcher test seed"
_LAUNCHER_SHA256 = hashlib.sha256(b"authenticated test launcher").hexdigest()


@dataclass(frozen=True, slots=True)
class _Paths:
    preregistration: Path
    preregistration_digest: str
    plan: Path
    release: Path
    split: Path
    historical: Path
    predecessor: Path
    archive: Path
    store: Path

    def metadata_kwargs(self) -> dict[str, object]:
        return {
            "preregistration_path": self.preregistration,
            "expected_preregistration_digest": self.preregistration_digest,
            "cohort_plan_path": self.plan,
            "release_descriptor_path": self.release,
            "split_path": self.split,
            "historical_seed_path": self.historical,
            "exposure_predecessor_path": self.predecessor,
        }


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _fixture(tmp_path: Path) -> _Paths:
    historical = load_historical_exposure(DEFAULT_SEED_PATH)
    unused = [
        shape
        for shape in historical.unused_basic_shape_families
        if shape not in BIRD_FAMILIES
    ]
    targets = ["bird1", *unused[:5]]
    partner_cursor = 5
    task_ids: set[str] = set()
    for candidate_index in range(3):
        shape_a = targets[2 * candidate_index]
        shape_b = targets[2 * candidate_index + 1]
        task_ids.update(
            {
                f"bd_{shape_a}-{shape_b}_0000",
                f"bd_{shape_a}_0000",
                f"bd_{shape_b}_0000",
            }
        )
        for shape in (shape_a, shape_b):
            for _ in range(14):
                partner = unused[partner_cursor]
                partner_cursor += 1
                task_ids.add(f"bd_{shape}-{partner}_0000")
    inventory = tuple(sorted(task_ids))
    split_bytes = canonical_json(
        {
            "train": list(inventory),
            "val": [],
            "test_ff": [],
            "test_bd": [],
            "test_hd_comb": [],
            "test_hd_novel": [],
        }
    )

    archive_path = tmp_path / "ShapeBongard_V2.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("metadata-only.txt", b"no panel bytes")
    archive_bytes = archive_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-metadata-launcher-test",
        archive_filename=archive_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename="ShapeBongard_V2_split.json",
        split_sha256="sha256:" + hashlib.sha256(split_bytes).hexdigest(),
        split_size_bytes=len(split_bytes),
        upstream_repository=OFFICIAL_UPSTREAM_REPOSITORY,
        upstream_commit=OFFICIAL_UPSTREAM_COMMIT,
        family_counts=(("bd", len(inventory)),),
        primary_split_counts=(
            ("test", 0),
            ("train", len(inventory)),
            ("val", 0),
        ),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256=task_id_inventory_digest(inventory),
        corpus_manifest_sha256="sha256:" + hashlib.sha256(b"corpus").hexdigest(),
    )
    exposure = ExposureLedger.create(descriptor.corpus_manifest_sha256)
    plan = plan_prototype_pair_cohort(
        release_descriptor=descriptor,
        split_bytes=split_bytes,
        task_ids=inventory,
        exposure_predecessor=exposure,
        historical_seed=historical,
        selection_seed=_SEED,
        expected_seed_commitment=prototype_pair_seed_commitment(_SEED),
        expected_release_descriptor_digest=descriptor.digest,
        expected_corpus_manifest_digest=descriptor.corpus_manifest_sha256,
        expected_split_source_digest=descriptor.split_sha256,
        expected_task_inventory_digest=descriptor.task_ids_sha256,
        expected_exposure_predecessor_digest=exposure.digest,
        expected_historical_seed_digest=historical.seed_digest,
        expected_resolver_policy_digest=semantic_resolver_policy_digest(historical),
        expected_basic_sampler_sha256=OFFICIAL_BASIC_SAMPLER_SHA256,
        expected_basic_generator_sha256=OFFICIAL_BASIC_GENERATOR_SHA256,
    )

    plan_path = tmp_path / "cohort.plan.json"
    release_path = tmp_path / "release.json"
    split_path = tmp_path / descriptor.split_filename
    predecessor_path = tmp_path / "predecessor.exposure.json"
    _write_canonical(plan_path, plan.to_data())
    _write_canonical(release_path, descriptor.to_dict())
    split_path.write_bytes(split_bytes)
    predecessor_path.write_text(exposure.to_json(), encoding="utf-8")

    preregistration = {
        "schema": (
            "gkm.bongard-prototype-pair-targeted-engineering-preregistration.v1"
        ),
        "created_at": "2026-08-07T00:00:00Z",
        "scope": "exact-unused-train-semantics-reused-targeted-engineering",
        "seed": {
            "value": _SEED,
            "provenance": "preexisting-test-commitment",
            "namespace": plan.namespace,
            "commitment": plan.selection_seed_commitment,
        },
        "source": {
            "release_descriptor_digest": descriptor.digest,
            "corpus_manifest_digest": descriptor.corpus_manifest_sha256,
            "split_source_digest": descriptor.split_sha256,
            "task_inventory_digest": descriptor.task_ids_sha256,
            "historical_seed_digest": historical.seed_digest,
            "exposure_predecessor_digest": exposure.digest,
        },
        "planner": {
            "algorithm_id": plan.algorithm_id,
            "source_sha256": plan.planner_source_sha256,
            "algorithm_digest": plan.planner_algorithm_digest,
        },
        "selection": {
            "candidate_count": len(plan.candidates),
            "selected_task_count": len(plan.selected_task_ids),
            "drill_task_id": plan.drill.task_id,
            "drill_shape_families": list(plan.drill.ordered_shapes),
            "plan_digest": plan.record_digest,
        },
        "statistics": {
            "opaque_tag_count": 2,
            "calibration_task_clusters_per_tag": plan.clusters_per_hypothesis,
            "hypothesis_count": plan.hypothesis_count,
            "confidence_level_ppm": plan.confidence_level_ppm,
            "zero_error_family_upper_ppm": plan.zero_error_family_upper_ppm,
            "targeted_engineering_tolerance_ppm": (
                plan.targeted_engineering_tolerance_ppm
            ),
            "zero_errors_required": plan.zero_errors_required_for_tolerance,
            "stronger_250k_claim_authorized": (
                plan.stronger_250k_claim_authorized
            ),
        },
        "execution": {
            "metadata_only_selection": True,
            "panel_bytes_opened_before_preregistration": False,
            "action_program_json_authorized": False,
            "thresholds_must_be_frozen_before_calibration": True,
            "formula_must_be_frozen_before_query_pixels": True,
            "cold_replay_must_be_model_free": True,
            "official_test_authorized": False,
        },
        "authority": {
            "predicate_authority_id": plan.predicate_authority_id,
            "python_is_canonical_authority": True,
            "lean_required": False,
            "lean_defines_artifact_identity": False,
            "lean_affects_selection_or_decision": False,
            "optional_secondary_checker_detachable": True,
        },
        "claims": {
            "targeted_engineering_only": True,
            "semantics_reused": True,
            "benchmark_claim_authorized": False,
            "unseen_claim_authorized": False,
        },
    }
    preregistration_digest = canonical_digest(preregistration)
    preregistration["record_digest"] = preregistration_digest
    preregistration_path = tmp_path / "campaign.prereg.json"
    _write_canonical(preregistration_path, preregistration)
    return _Paths(
        preregistration=preregistration_path,
        preregistration_digest=preregistration_digest,
        plan=plan_path,
        release=release_path,
        split=split_path,
        historical=DEFAULT_SEED_PATH,
        predecessor=predecessor_path,
        archive=archive_path,
        store=tmp_path / "campaign-store",
    )


def _forbidden(label: str):
    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"forbidden during metadata preflight: {label}")

    return fail


def test_metadata_verification_is_dry_and_never_touches_pixels_store_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(PrototypePairCampaignStore, "open", _forbidden("store"))
    monkeypatch.setattr(OfficialPanelArchive, "load", _forbidden("archive load"))
    monkeypatch.setattr(OfficialPanelArchive, "read_panel", _forbidden("panel read"))
    monkeypatch.setattr(ReleasedOfficialPanel, "release", _forbidden("panel release"))
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.snapshot_cloud_policy_cache",
        _forbidden("policy snapshot"),
    )
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.codex_cli_authenticated_fingerprint",
        _forbidden("Codex fingerprint"),
    )

    verified = verify_prototype_pair_campaign_metadata(**paths.metadata_kwargs())

    assert verified.cohort_plan.record_digest == verified.pins.plan_digest
    assert verified.exposure_predecessor.digest == (
        verified.pins.exposure_predecessor_digest
    )
    assert not paths.store.exists()


def test_prepare_freezes_python_defaults_and_dispatches_once_without_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    model_catalog, no_tools_attestation = canonical_no_tools_runtime(
        _LAUNCHER_SHA256
    )
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.snapshot_cloud_policy_cache",
        lambda: CloudPolicyCacheSnapshot(None),
    )
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.codex_cli_authenticated_fingerprint",
        lambda _executable, *, expected_launcher_digest: {
            "version": "codex-cli 1.2.3-test",
            "launcher_digest": expected_launcher_digest,
        },
    )
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.snapshot_pinned_model_catalog",
        lambda: model_catalog,
    )
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.attest_codex_no_tools",
        lambda **_kwargs: no_tools_attestation,
    )
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.snapshot_python_runtime_identity",
        lambda: PythonRuntimeIdentity(
            runtime_id="cpython-3.13.7-test",
            identity_digest=hashlib.sha256(b"python-runtime").hexdigest(),
            executable_sha256=hashlib.sha256(b"python-executable").hexdigest(),
        ),
    )
    monkeypatch.setattr(OfficialPanelArchive, "read_panel", _forbidden("panel read"))
    monkeypatch.setattr(ReleasedOfficialPanel, "release", _forbidden("panel release"))

    prepared = prepare_prototype_pair_campaign_launch(
        **paths.metadata_kwargs(),
        official_archive_path=paths.archive,
        store_root=paths.store,
        expected_codex_launcher_sha256=_LAUNCHER_SHA256,
    )

    assert prepared.identities.observer_model_id == "gpt-5.6-sol"
    assert prepared.identities.observer_reasoning_effort == "medium"
    assert prepared.identities.ranker_model_id == "gpt-5.6-sol"
    assert prepared.identities.ranker_reasoning_effort == "medium"
    assert (
        prepared.identities.execution_configuration_digest
        == prepared.configuration.record_digest
    )
    assert prepared.configuration.observer_minutes == 15
    assert prepared.configuration.observer_verbose is False
    assert prepared.configuration.observer_executable == "codex"
    assert prepared.configuration.ranker_minutes == 15
    assert prepared.configuration.ranker_verbose is False
    assert prepared.configuration.ranker_executable == "codex"
    assert prepared.ranker.minutes == prepared.configuration.ranker_minutes
    assert prepared.ranker.verbose is prepared.configuration.ranker_verbose
    assert prepared.ranker.executable == prepared.configuration.ranker_executable
    assert prepared.model_catalog_snapshot == model_catalog
    assert prepared.no_tools_attestation == no_tools_attestation
    assert (
        prepared.model_catalog_snapshot
        is prepared.precommit.identities.codex_model_catalog_snapshot
    )
    assert (
        prepared.no_tools_attestation
        is prepared.precommit.identities.codex_no_tools_attestation
    )
    assert prepared.identities is prepared.precommit.identities
    assert prepared.identities.codex_model_catalog_snapshot == model_catalog
    assert prepared.identities.codex_no_tools_attestation == no_tools_attestation
    assert {
        (item.absent_upper_ppm, item.present_lower_ppm)
        for item in prepared.identities.thresholds
    } == {(DEFAULT_ABSENT_UPPER_PPM, DEFAULT_PRESENT_LOWER_PPM)}
    runtime_sources = dict(prepared.identities.runtime_source_digests)
    assert runtime_sources["campaign-cli"] == hashlib.sha256(
        Path(campaign_cli.__file__).read_bytes()
    ).hexdigest()
    assert not list((paths.store / "objects").rglob("*.json"))

    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_campaign(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        precommit = cast(PrototypePairExecutionPrecommit, kwargs["precommit"])
        store = cast(PrototypePairCampaignStore, kwargs["store"])
        assert store is prepared.store
        payload = canonical_json(precommit.to_data()) + b"\n"
        store.persist_execution_precommit(payload, precommit.record_digest)
        return sentinel

    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli._campaign_entrypoint",
        lambda: fake_campaign,
    )
    assert dispatch_prepared_prototype_pair_campaign(prepared) is sentinel
    assert len(calls) == 1
    assert set(calls[0]) == set(
        inspect.signature(run_prototype_pair_campaign).parameters
    )
    assert calls[0]["expected_precommit_digest"] == prepared.precommit.record_digest
    assert calls[0]["observed_codex_cli_version"] == "codex-cli 1.2.3-test"
    assert calls[0]["ranker"] is prepared.ranker


def test_no_tools_attestation_failure_precedes_store_archive_and_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    model_catalog, _attestation = canonical_no_tools_runtime(_LAUNCHER_SHA256)
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.snapshot_cloud_policy_cache",
        lambda: CloudPolicyCacheSnapshot(None),
    )
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.codex_cli_authenticated_fingerprint",
        lambda _executable, *, expected_launcher_digest: {
            "version": "codex-cli 0.147.0",
            "launcher_digest": expected_launcher_digest,
        },
    )
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.snapshot_pinned_model_catalog",
        lambda: model_catalog,
    )

    def preflight_failure(**_kwargs: object) -> object:
        raise RuntimeError("synthetic no-tools preflight failed")

    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.attest_codex_no_tools",
        preflight_failure,
    )
    monkeypatch.setattr(PrototypePairCampaignStore, "open", _forbidden("store"))
    monkeypatch.setattr(OfficialPanelArchive, "load", _forbidden("archive load"))
    monkeypatch.setattr(OfficialPanelArchive, "read_panel", _forbidden("panel read"))
    monkeypatch.setattr(ReleasedOfficialPanel, "release", _forbidden("panel release"))
    monkeypatch.setattr(
        "bongard.prototype_pair_campaign_cli.snapshot_python_runtime_identity",
        _forbidden("Python runtime snapshot after failed preflight"),
    )

    with pytest.raises(RuntimeError, match="no-tools preflight failed"):
        prepare_prototype_pair_campaign_launch(
            **paths.metadata_kwargs(),
            official_archive_path=paths.archive,
            store_root=paths.store,
            expected_codex_launcher_sha256=_LAUNCHER_SHA256,
        )
    assert not paths.store.exists()
