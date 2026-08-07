from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger, semantic_resolver_policy_digest
from bongard.historical_exposure import DEFAULT_SEED_PATH, load_historical_exposure
from bongard.prototype_pair_campaign_cli import (
    verify_prototype_pair_campaign_metadata,
)
from bongard.prototype_pair_cohort import (
    BIRD_FAMILIES,
    OFFICIAL_BASIC_GENERATOR_SHA256,
    OFFICIAL_BASIC_SAMPLER_SHA256,
    OFFICIAL_UPSTREAM_COMMIT,
    OFFICIAL_UPSTREAM_REPOSITORY,
    PrototypePairCohortPlan,
    plan_prototype_pair_cohort,
    prototype_pair_seed_commitment,
    task_id_inventory_digest,
)
from bongard.prototype_pair_recovery_preregister import (
    EXPECTED_FAILED_CAMPAIGN_STATUS,
    EXPECTED_FAILED_OBSERVER_STATUS,
    PrototypePairRecoveryError,
    RECOVERY_GENERATOR_SOURCE_SHA256,
    RECOVERY_POLICY_ID,
    generate_prototype_pair_recovery_preregistration,
)
from bongard.release import OfficialReleaseDescriptor


_SEED = "preexisting synthetic recovery seed"
_NAMESPACE = "bongard-prototype-pair-synthetic-recovery-v1"
_FAILED_CAMPAIGN_DIGEST = "sha256:" + hashlib.sha256(
    b"synthetic failed campaign"
).hexdigest()


@dataclass(frozen=True, slots=True)
class _RecoveryFixture:
    preregistration: Path
    preregistration_digest: str
    old_plan_path: Path
    old_plan: PrototypePairCohortPlan
    release: Path
    split: Path
    predecessor_path: Path
    predecessor: ExposureLedger
    successor_object: Path
    successor: ExposureLedger


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _preregistration(
    *,
    descriptor: OfficialReleaseDescriptor,
    historical_seed_digest: str,
    predecessor: ExposureLedger,
    plan: PrototypePairCohortPlan,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": (
            "gkm.bongard-prototype-pair-targeted-engineering-preregistration.v1"
        ),
        "created_at": "2026-08-07T00:00:00Z",
        "scope": "exact-unused-train-semantics-reused-targeted-engineering",
        "seed": {
            "value": _SEED,
            "provenance": "preexisting synthetic test commitment",
            "namespace": _NAMESPACE,
            "commitment": plan.selection_seed_commitment,
        },
        "source": {
            "release_descriptor_digest": descriptor.digest,
            "corpus_manifest_digest": descriptor.corpus_manifest_sha256,
            "split_source_digest": descriptor.split_sha256,
            "task_inventory_digest": descriptor.task_ids_sha256,
            "historical_seed_digest": historical_seed_digest,
            "exposure_predecessor_digest": predecessor.digest,
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
    return {**body, "record_digest": canonical_digest(body)}


def _fixture(tmp_path: Path) -> _RecoveryFixture:
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
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-metadata-only-recovery-test",
        archive_filename="deliberately-absent-archive.zip",
        archive_sha256="sha256:" + "a" * 64,
        archive_size_bytes=1,
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
    predecessor = ExposureLedger.create(descriptor.corpus_manifest_sha256)
    commitment = prototype_pair_seed_commitment(_SEED, namespace=_NAMESPACE)
    old_plan = plan_prototype_pair_cohort(
        release_descriptor=descriptor,
        split_bytes=split_bytes,
        task_ids=inventory,
        exposure_predecessor=predecessor,
        historical_seed=historical,
        selection_seed=_SEED,
        expected_seed_commitment=commitment,
        expected_release_descriptor_digest=descriptor.digest,
        expected_corpus_manifest_digest=descriptor.corpus_manifest_sha256,
        expected_split_source_digest=descriptor.split_sha256,
        expected_task_inventory_digest=descriptor.task_ids_sha256,
        expected_exposure_predecessor_digest=predecessor.digest,
        expected_historical_seed_digest=historical.seed_digest,
        expected_resolver_policy_digest=semantic_resolver_policy_digest(historical),
        expected_basic_sampler_sha256=OFFICIAL_BASIC_SAMPLER_SHA256,
        expected_basic_generator_sha256=OFFICIAL_BASIC_GENERATOR_SHA256,
        namespace=_NAMESPACE,
    )
    preregistration = _preregistration(
        descriptor=descriptor,
        historical_seed_digest=historical.seed_digest,
        predecessor=predecessor,
        plan=old_plan,
    )
    successor = predecessor.record(
        phase="prototype_pair_selected_task_release",
        actor="synthetic campaign",
        purpose="record the selected prototype-pair task release",
        task_ids=old_plan.selected_task_ids,
        source=_FAILED_CAMPAIGN_DIGEST,
        observed_at="2026-08-07T01:00:00Z",
        known_task_ids=inventory,
        require_unseen=True,
    )

    release_path = tmp_path / "release.json"
    split_path = tmp_path / descriptor.split_filename
    predecessor_path = tmp_path / "old.exposure.json"
    old_plan_path = tmp_path / "old.plan.json"
    preregistration_path = tmp_path / "old.prereg.json"
    successor_object = tmp_path / "successor-object.json"
    _write_canonical(release_path, descriptor.to_dict())
    split_path.write_bytes(split_bytes)
    predecessor_path.write_text(predecessor.to_json(), encoding="utf-8")
    _write_canonical(old_plan_path, old_plan.to_data())
    _write_canonical(preregistration_path, preregistration)
    _write_canonical(successor_object, successor.to_dict())
    return _RecoveryFixture(
        preregistration=preregistration_path,
        preregistration_digest=str(preregistration["record_digest"]),
        old_plan_path=old_plan_path,
        old_plan=old_plan,
        release=release_path,
        split=split_path,
        predecessor_path=predecessor_path,
        predecessor=predecessor,
        successor_object=successor_object,
        successor=successor,
    )


def _generation_kwargs(
    fixture: _RecoveryFixture,
    tmp_path: Path,
    *,
    successor_object: Path | None = None,
    successor_digest: str | None = None,
    failed_campaign_status: str = EXPECTED_FAILED_CAMPAIGN_STATUS,
) -> dict[str, object]:
    return {
        "old_preregistration_path": fixture.preregistration,
        "expected_old_preregistration_digest": fixture.preregistration_digest,
        "old_cohort_plan_path": fixture.old_plan_path,
        "expected_old_cohort_plan_digest": fixture.old_plan.record_digest,
        "old_exposure_predecessor_path": fixture.predecessor_path,
        "successor_exposure_object_path": (
            successor_object or fixture.successor_object
        ),
        "expected_successor_exposure_digest": (
            successor_digest or fixture.successor.digest
        ),
        "failed_campaign_digest": _FAILED_CAMPAIGN_DIGEST,
        "failed_campaign_status": failed_campaign_status,
        "failed_observer_status": EXPECTED_FAILED_OBSERVER_STATUS,
        "release_descriptor_path": fixture.release,
        "split_path": fixture.split,
        "historical_seed_path": DEFAULT_SEED_PATH,
        "output_exposure_predecessor_path": tmp_path / "new.exposure.json",
        "output_cohort_plan_path": tmp_path / "new.plan.json",
        "output_preregistration_path": tmp_path / "new.prereg.json",
        "created_at": "2026-08-07T02:00:00Z",
    }


def test_same_seed_successor_is_disjoint_exclusive_and_cold_verified(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    kwargs = _generation_kwargs(fixture, tmp_path)

    artifacts = generate_prototype_pair_recovery_preregistration(**kwargs)

    assert artifacts.predecessor_digest == fixture.successor.digest
    assert len(artifacts.selected_task_ids) == 31
    assert set(artifacts.selected_task_ids).isdisjoint(
        fixture.old_plan.selected_task_ids
    )
    assert Path(kwargs["output_exposure_predecessor_path"]).read_bytes() == (
        fixture.successor.to_json().encode("utf-8")
    )
    plan = PrototypePairCohortPlan.from_data(
        json.loads(Path(kwargs["output_cohort_plan_path"]).read_bytes())
    )
    assert plan.namespace == fixture.old_plan.namespace == _NAMESPACE
    assert plan.selection_seed_digest == fixture.old_plan.selection_seed_digest
    assert plan.selection_seed_commitment == (
        fixture.old_plan.selection_seed_commitment
    )
    assert plan.exposure_predecessor_digest == fixture.successor.digest
    assert set(fixture.old_plan.selected_task_ids).issubset(
        plan.excluded_exact_used_train_basic_task_ids
    )
    preregistration = json.loads(
        Path(kwargs["output_preregistration_path"]).read_bytes()
    )
    provenance = preregistration["seed"]["provenance"]
    for fact in (
        RECOVERY_POLICY_ID,
        f"prior_preregistration_digest={fixture.preregistration_digest}",
        f"prior_plan_digest={fixture.old_plan.record_digest}",
        f"failed_campaign_digest={_FAILED_CAMPAIGN_DIGEST}",
        f"failed_campaign_status={EXPECTED_FAILED_CAMPAIGN_STATUS}",
        f"failed_observer_status={EXPECTED_FAILED_OBSERVER_STATUS}",
        f"successor_exposure_digest={fixture.successor.digest}",
        "seed_and_namespace_reused=true",
        "prior_attempt_retained_in_denominator=true",
        f"generator_source_sha256={RECOVERY_GENERATOR_SOURCE_SHA256}",
    ):
        assert fact in provenance
    verified = verify_prototype_pair_campaign_metadata(
        preregistration_path=kwargs["output_preregistration_path"],
        expected_preregistration_digest=artifacts.preregistration_digest,
        cohort_plan_path=kwargs["output_cohort_plan_path"],
        release_descriptor_path=fixture.release,
        split_path=fixture.split,
        historical_seed_path=DEFAULT_SEED_PATH,
        exposure_predecessor_path=kwargs["output_exposure_predecessor_path"],
    )
    assert verified.cohort_plan == plan

    before = {
        Path(kwargs[name]): Path(kwargs[name]).read_bytes()
        for name in (
            "output_exposure_predecessor_path",
            "output_cohort_plan_path",
            "output_preregistration_path",
        )
    }
    with pytest.raises(PrototypePairRecoveryError, match="already exists"):
        generate_prototype_pair_recovery_preregistration(**kwargs)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("tamper", ("missing_task", "wrong_phase"))
def test_tampered_successor_fails_before_outputs(
    tmp_path: Path,
    tamper: str,
) -> None:
    fixture = _fixture(tmp_path)
    tasks = fixture.old_plan.selected_task_ids
    tampered = fixture.predecessor.record(
        phase=(
            "wrong_release_phase"
            if tamper == "wrong_phase"
            else "prototype_pair_selected_task_release"
        ),
        actor="synthetic campaign",
        purpose="tampered successor",
        task_ids=tasks[:-1] if tamper == "missing_task" else tasks,
        source=_FAILED_CAMPAIGN_DIGEST,
        observed_at="2026-08-07T01:00:00Z",
        require_unseen=True,
    )
    successor_object = tmp_path / f"tampered-{tamper}.json"
    _write_canonical(successor_object, tampered.to_dict())
    kwargs = _generation_kwargs(
        fixture,
        tmp_path,
        successor_object=successor_object,
        successor_digest=tampered.digest,
    )

    with pytest.raises(
        PrototypePairRecoveryError,
        match="does not expose exactly the old selected tasks",
    ):
        generate_prototype_pair_recovery_preregistration(**kwargs)

    assert not Path(kwargs["output_exposure_predecessor_path"]).exists()
    assert not Path(kwargs["output_cohort_plan_path"]).exists()
    assert not Path(kwargs["output_preregistration_path"]).exists()


def test_wrong_failure_status_fails_before_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    kwargs = _generation_kwargs(
        fixture,
        tmp_path,
        failed_campaign_status="completed",
    )

    with pytest.raises(PrototypePairRecoveryError, match="not description_gap"):
        generate_prototype_pair_recovery_preregistration(**kwargs)

    assert not Path(kwargs["output_exposure_predecessor_path"]).exists()
    assert not Path(kwargs["output_cohort_plan_path"]).exists()
    assert not Path(kwargs["output_preregistration_path"]).exists()


def test_generator_has_no_archive_panel_model_or_store_input_surface() -> None:
    parameters = set(
        inspect.signature(
            generate_prototype_pair_recovery_preregistration
        ).parameters
    )
    assert not any(
        forbidden in parameter
        for parameter in parameters
        for forbidden in ("archive", "panel", "model", "store")
    )
    source = inspect.getsource(
        inspect.getmodule(generate_prototype_pair_recovery_preregistration)
    )
    parsed = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(parsed)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(parsed)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        forbidden in module
        for module in imported_modules
        for forbidden in (
            "official_panel_archive",
            "prototype_pair_campaign_store",
            "transport",
        )
    )
