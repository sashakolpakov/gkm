from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger
from bongard.prototype_pair_campaign_store import (
    PrototypePairCallAlreadyFinished,
    PrototypePairCampaignStore,
    PrototypePairCampaignStoreError,
)
from bongard.prototype_pair_cohort import plan_prototype_pair_cohort
from bongard.prototype_pair_execution_precommit import (
    prepare_prototype_pair_execution_precommit,
)
from bongard.prototype_scene_headless_runner import (
    PrototypeSceneCandidateFreeze,
    PrototypeSceneFreezeCommitReceipt,
    run_prototype_scene_headless,
)
from bongard.tests.test_prototype_pair_cohort import _fixture, _kwargs
from bongard.tests.test_prototype_pair_execution_precommit import _identities
from bongard.tests.test_prototype_scene_headless_pipeline import (
    _conjunction_only_support,
    _panel,
    _rank_response,
    _verifier,
    scene_authority,
)


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _campaign_inputs():
    historical, release, split, inventory, predecessor, _candidate_ids = _fixture()
    plan = plan_prototype_pair_cohort(
        **_kwargs(historical, release, split, inventory, predecessor)
    )
    identities = _identities(plan)
    precommit = prepare_prototype_pair_execution_precommit(
        cohort_plan=plan,
        identities=identities,
        expected_cohort_plan_digest=plan.record_digest,
        expected_identity_bundle_digest=identities.record_digest,
        expected_exposure_predecessor_digest=predecessor.digest,
    )
    return plan, predecessor, precommit


def _authorized_store(tmp_path: Path):
    plan, predecessor, precommit = _campaign_inputs()
    store = PrototypePairCampaignStore.open(tmp_path / "store")
    precommit_bytes = canonical_json(precommit.to_data()) + b"\n"
    precommit_receipt = store.persist_execution_precommit(
        precommit_bytes, precommit.record_digest
    )
    authorization = store.authorize_release(
        plan,
        predecessor,
        precommit_receipt,
        expected_plan_digest=plan.record_digest,
        expected_execution_precommit_digest=precommit.record_digest,
        expected_exposure_predecessor_digest=predecessor.digest,
        actor="campaign-store-test",
        observed_at="2026-08-07T18:00:00Z",
    )
    return store, plan, predecessor, precommit, precommit_receipt, authorization


def _record(value: int = 1):
    body = {"schema": "gkm.test-campaign-store-record.v1", "value": value}
    digest = _address(body)
    return {**body, "record_digest": digest}, digest


def test_canonical_objects_are_exclusive_fsynced_and_exactly_reloaded(
    tmp_path: Path,
) -> None:
    store = PrototypePairCampaignStore.open(tmp_path / "store")
    data, digest = _record()
    first = store.persist_canonical_object("observer_artifact", data, digest)
    second = store.persist_canonical_object("observer_artifact", data, digest)
    assert first == second
    assert store.load_canonical_object(first, digest) == data
    assert (store.root / first.relative_path).read_bytes() == canonical_json(data) + b"\n"

    (store.root / first.relative_path).write_bytes(b"{}\n")
    with pytest.raises(PrototypePairCampaignStoreError):
        store.load_canonical_object(first, digest)


def test_precommit_and_one_31_task_successor_are_durable_before_authorization(
    tmp_path: Path,
) -> None:
    store, plan, predecessor, precommit, receipt, authorization = (
        _authorized_store(tmp_path)
    )
    assert store.verify_execution_precommit(
        receipt,
        precommit.record_digest,
        canonical_json(precommit.to_data()) + b"\n",
    )
    successor = ExposureLedger.from_dict(
        store.load_canonical_object(
            authorization.exposure_successor_receipt,
            authorization.exposure_successor_digest,
        )
    )
    assert len(successor.events) == len(predecessor.events) + 1
    event = successor.events[-1]
    assert event.task_ids == plan.selected_task_ids
    assert len(event.task_ids) == 31
    assert event.panel_ids == ()
    assert authorization == store.load_release_authorization(
        authorization.record_digest
    )

    repeated = store.authorize_release(
        plan,
        predecessor,
        receipt,
        expected_plan_digest=plan.record_digest,
        expected_execution_precommit_digest=precommit.record_digest,
        expected_exposure_predecessor_digest=predecessor.digest,
        actor="campaign-store-test",
        observed_at="2026-08-07T23:59:59Z",
    )
    assert repeated == authorization
    assert repeated.observed_at == "2026-08-07T18:00:00Z"
    exposure_files = list((store.root / "objects" / "exposure_successor").iterdir())
    assert len(exposure_files) == 1

    with pytest.raises(PrototypePairCampaignStoreError, match="another precommit"):
        store.authorize_release(
            plan,
            predecessor,
            receipt,
            expected_plan_digest=plan.record_digest,
            expected_execution_precommit_digest=precommit.record_digest,
            expected_exposure_predecessor_digest=predecessor.digest,
            actor="different-actor",
            observed_at="2026-08-08T00:00:00Z",
        )


def test_plan_predecessor_root_rejects_changed_configuration_precommit(
    tmp_path: Path,
) -> None:
    store, plan, predecessor, precommit, _receipt, authorization = (
        _authorized_store(tmp_path)
    )
    changed_identities = _identities(
        plan,
        execution_configuration_digest=_address("changed execution configuration"),
    )
    changed_precommit = prepare_prototype_pair_execution_precommit(
        cohort_plan=plan,
        identities=changed_identities,
        expected_cohort_plan_digest=plan.record_digest,
        expected_identity_bundle_digest=changed_identities.record_digest,
        expected_exposure_predecessor_digest=predecessor.digest,
    )
    assert changed_precommit.record_digest != precommit.record_digest
    changed_receipt = store.persist_execution_precommit(
        canonical_json(changed_precommit.to_data()) + b"\n",
        changed_precommit.record_digest,
    )
    with pytest.raises(PrototypePairCampaignStoreError, match="another precommit"):
        store.authorize_release(
            plan,
            predecessor,
            changed_receipt,
            expected_plan_digest=plan.record_digest,
            expected_execution_precommit_digest=changed_precommit.record_digest,
            expected_exposure_predecessor_digest=predecessor.digest,
            actor=authorization.actor,
            observed_at="2026-08-07T18:10:00Z",
        )
    exposure_files = list((store.root / "objects" / "exposure_successor").iterdir())
    assert len(exposure_files) == 1


def test_nonterminal_authorization_claim_is_permanent_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bongard.prototype_pair_campaign_store as module

    plan, predecessor, precommit = _campaign_inputs()
    store = PrototypePairCampaignStore.open(tmp_path / "store")
    receipt = store.persist_execution_precommit(
        canonical_json(precommit.to_data()) + b"\n", precommit.record_digest
    )
    original = module._write_once

    def interrupt_after_claim(path: Path, payload: bytes, *, allow_identical: bool):
        created = original(path, payload, allow_identical=allow_identical)
        if path.name.endswith(".claim.json") and path.parent.name == "authorizations":
            raise RuntimeError("simulated crash after exclusive authorization claim")
        return created

    monkeypatch.setattr(module, "_write_once", interrupt_after_claim)
    with pytest.raises(RuntimeError):
        store.authorize_release(
            plan,
            predecessor,
            receipt,
            expected_plan_digest=plan.record_digest,
            expected_execution_precommit_digest=precommit.record_digest,
            expected_exposure_predecessor_digest=predecessor.digest,
            actor="campaign-store-test",
            observed_at="2026-08-07T18:00:00Z",
        )
    monkeypatch.setattr(module, "_write_once", original)
    with pytest.raises(
        PrototypePairCampaignStoreError, match="nonterminal durable claim"
    ):
        store.authorize_release(
            plan,
            predecessor,
            receipt,
            expected_plan_digest=plan.record_digest,
            expected_execution_precommit_digest=precommit.record_digest,
            expected_exposure_predecessor_digest=predecessor.digest,
            actor="campaign-store-test",
            observed_at="2026-08-07T18:01:00Z",
        )


def test_call_claim_is_one_shot_and_terminal_requires_durable_result(
    tmp_path: Path,
) -> None:
    store, _plan, _predecessor, _precommit, _receipt, authorization = (
        _authorized_store(tmp_path)
    )
    context = _address("observer-context")

    def claim(index: int):
        return store.claim_call(
            authorization,
            phase="prototype_description_observed",
            subject_id="six-prototype-reference-set",
            context_digest=context,
            claimed_at=f"2026-08-07T18:01:{index:02d}Z",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        admissions = list(pool.map(claim, range(8)))
    assert sum(item.model_eligible for item in admissions) == 1
    assert all(
        item.model_eligible or item.reason == "preexisting_nonterminal_claim"
        for item in admissions
    )
    fresh = next(item for item in admissions if item.model_eligible)

    alien = PrototypePairCampaignStore.open(tmp_path / "alien")
    result_data, result_digest = _record(2)
    alien_receipt = alien.persist_canonical_object(
        "observer_artifact", result_data, result_digest
    )
    with pytest.raises(PrototypePairCampaignStoreError):
        store.finish_call(
            fresh.claim,
            terminal_status="success",
            result_receipt=alien_receipt,
            finished_at="2026-08-07T18:02:00Z",
        )

    result_receipt = store.persist_canonical_object(
        "observer_artifact", result_data, result_digest
    )
    with pytest.raises(PrototypePairCampaignStoreError):
        store.finish_call(
            fresh.claim,
            terminal_status="success",
            result_receipt=result_receipt,
            result_digest=_address("different-result"),
            finished_at="2026-08-07T18:02:00Z",
        )
    outcome = store.finish_call(
        fresh.claim,
        terminal_status="success",
        result_receipt=result_receipt,
        result_digest=result_digest,
        finished_at="2026-08-07T18:02:00Z",
    )
    assert outcome.result_digest == result_digest
    assert store.load_call_outcome(fresh.claim) == outcome
    with pytest.raises(PrototypePairCallAlreadyFinished):
        store.finish_call(
            fresh.claim,
            terminal_status="error",
            result_receipt=result_receipt,
            finished_at="2026-08-07T18:03:00Z",
        )
    resumed = PrototypePairCampaignStore.open(store.root).claim_call(
        authorization,
        phase="prototype_description_observed",
        subject_id="six-prototype-reference-set",
        context_digest=context,
        claimed_at="2026-08-07T18:04:00Z",
    )
    assert resumed.model_eligible is False
    assert resumed.reason == "preexisting_terminal_outcome"
    assert resumed.terminal_outcome == outcome

    seal = store.seal_call_journal(
        authorization.record_digest,
        expected_terminal_key_digests=(outcome.key_digest,),
        sealed_at="2026-08-07T18:05:00Z",
    )
    assert store.verify_call_journal_seal(
        authorization.record_digest,
        expected_terminal_key_digests=(outcome.key_digest,),
    ) == seal
    assert store.enumerate_call_journal(authorization.record_digest) == (
        (fresh.claim, outcome),
    )
    assert store.claim_call(
        authorization,
        phase="prototype_description_observed",
        subject_id="six-prototype-reference-set",
        context_digest=context,
        claimed_at="2026-08-07T18:06:00Z",
    ).terminal_outcome == outcome
    with pytest.raises(PrototypePairCampaignStoreError, match="journal is sealed"):
        store.claim_call(
            authorization,
            phase="twelve_support_scenes_released_and_observed",
            subject_id="new-key-after-terminal-seal",
            context_digest=_address("new-context-after-terminal-seal"),
            claimed_at="2026-08-07T18:07:00Z",
        )


def test_journal_seal_rejects_extra_or_nonterminal_claims(tmp_path: Path) -> None:
    store, _plan, _predecessor, _precommit, _receipt, authorization = (
        _authorized_store(tmp_path)
    )
    first = store.claim_call(
        authorization,
        phase="prototype_description_observed",
        subject_id="expected-terminal",
        context_digest=_address("expected-context"),
        claimed_at="2026-08-07T18:01:00Z",
    )
    result_data, result_digest = _record(3)
    result_receipt = store.persist_canonical_object(
        "observer_artifact", result_data, result_digest
    )
    outcome = store.finish_call(
        first.claim,
        terminal_status="success",
        result_receipt=result_receipt,
        finished_at="2026-08-07T18:02:00Z",
    )
    extra = store.claim_call(
        authorization,
        phase="twelve_support_scenes_released_and_observed",
        subject_id="orphan-nonterminal",
        context_digest=_address("orphan-context"),
        claimed_at="2026-08-07T18:03:00Z",
    )
    assert extra.model_eligible is True

    with pytest.raises(PrototypePairCampaignStoreError, match="key set differs"):
        store.seal_call_journal(
            authorization.record_digest,
            expected_terminal_key_digests=(outcome.key_digest,),
            sealed_at="2026-08-07T18:04:00Z",
        )
    with pytest.raises(PrototypePairCampaignStoreError, match="nonterminal claim"):
        store.seal_call_journal(
            authorization.record_digest,
            expected_terminal_key_digests=tuple(
                sorted((outcome.key_digest, extra.claim.key_digest))
            ),
            sealed_at="2026-08-07T18:04:01Z",
        )


def test_atomic_publication_stress_has_no_partial_readers(tmp_path: Path) -> None:
    store, _plan, _predecessor, _precommit, _receipt, authorization = (
        _authorized_store(tmp_path)
    )
    workers = 8
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for iteration in range(30):
            data, digest = _record(10_000 + iteration)
            object_barrier = Barrier(workers)

            def persist(_index: int):
                object_barrier.wait()
                return store.persist_canonical_object(
                    "publication_stress", data, digest
                )

            receipts = list(pool.map(persist, range(workers)))
            assert len(set(receipts)) == 1
            assert store.load_canonical_object(receipts[0], digest) == data

            claim_barrier = Barrier(workers)
            context = _address({"stress-claim": iteration})

            def acquire(index: int):
                claim_barrier.wait()
                return store.claim_call(
                    authorization,
                    phase="twelve_support_scenes_released_and_observed",
                    subject_id=f"stress-panel-{iteration}",
                    context_digest=context,
                    claimed_at=f"2026-08-07T19:{iteration:02d}:{index:02d}Z",
                )

            admissions = list(pool.map(acquire, range(workers)))
            assert sum(item.model_eligible for item in admissions) == 1
            assert sum(
                item.reason == "preexisting_nonterminal_claim"
                for item in admissions
            ) == workers - 1
    assert not list(store.root.rglob("*.tmp"))


def test_candidate_freeze_returns_and_cold_reloads_runner_commit(
    tmp_path: Path,
) -> None:
    authority = scene_authority.__wrapped__()
    family, library, _context, _png = authority
    positives, negatives = _conjunction_only_support(authority)
    store = PrototypePairCampaignStore.open(tmp_path / "store")
    captured: list[bytes] = []

    def commit(payload: bytes):
        captured.append(payload)
        return store.persist_candidate_freeze(payload)

    archive = run_prototype_scene_headless(
        family,
        library,
        positives,
        negatives,
        artifact_verifier=_verifier([]),
        ranker=_rank_response,
        freeze_committer=commit,
        query_source=lambda _freeze: {
            "positive": _panel(authority, "query-positive", ("present", "present")),
            "negative": _panel(authority, "query-negative", ("absent", "present")),
        },
    )
    assert archive.freeze is not None and archive.freeze_commit is not None
    assert isinstance(archive.freeze_commit, PrototypeSceneFreezeCommitReceipt)
    commit_receipt, freeze_storage, commit_storage = (
        store.load_candidate_freeze_commit(archive.freeze_commit.record_digest)
    )
    assert commit_receipt == archive.freeze_commit
    assert commit_receipt.storage_id == freeze_storage.record_digest
    assert store.verify_stored_object_bytes(
        freeze_storage, archive.freeze.record_digest
    ) == captured[0]
    assert PrototypeSceneCandidateFreeze.from_data(json.loads(captured[0])) == archive.freeze
    assert store.load_canonical_object(
        commit_storage, commit_receipt.record_digest
    ) == commit_receipt.to_data()


def test_store_source_has_no_lean_or_model_transport_dependency() -> None:
    source = Path(__import__(
        "bongard.prototype_pair_campaign_store", fromlist=["x"]
    ).__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("lean" in name.lower() for name in imports)
    assert not any("transport" in name.lower() for name in imports)
