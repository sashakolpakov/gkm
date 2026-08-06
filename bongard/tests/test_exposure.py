from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from bongard.exposure import (
    basic_morphology_cluster_id,
    ExposureIntegrityError,
    ExposureLedger,
    ExposureViolation,
    SemanticDisclosureKey,
    deterministic_partition,
    import_historical_exposures,
    semantic_resolver_policy_digest,
    semantic_policy_blocked_keys,
)
from bongard.historical_exposure import load_historical_exposure


CORPUS = "sha256:" + "c" * 64
PANEL_A = "ff/ff_a_0000/1/0.png"
PANEL_B = "ff/ff_a_0000/1/1.png"
PANEL_C = "bd/bd_c_0000/0/0.png"


def test_record_returns_new_frozen_hash_chained_ledger() -> None:
    empty = ExposureLedger.create(CORPUS)
    one = empty.record(
        phase="drill",
        actor="vision-proposer",
        purpose="fit a soft predicate",
        panel_ids=[PANEL_A],
        observed_at="2026-08-05T10:00:00Z",
    )
    two = one.record(
        phase="dev",
        actor="verifier",
        purpose="held-out replay",
        task_ids=["bd_c_0000"],
        observed_at="2026-08-05T10:01:00Z",
    )

    assert empty.events == ()
    assert one.events[0].previous_digest is None
    assert two.events[1].previous_digest == two.events[0].digest
    assert two.digest.startswith("sha256:")
    with pytest.raises(FrozenInstanceError):
        two.events[0].purpose = "tamper"  # type: ignore[misc]


def test_unseen_claim_fails_for_prior_task_or_panel_exposure() -> None:
    ledger = ExposureLedger.create(CORPUS).record(
        phase="historical",
        actor="legacy",
        purpose="old visual run",
        panel_ids=[PANEL_A],
        observed_at="2026-08-05T10:00:00Z",
    )

    with pytest.raises(ExposureViolation, match="not unseen"):
        ledger.assert_unseen(panel_ids=[PANEL_A])
    # Conservatively, seeing any panel contaminates the task concept.
    with pytest.raises(ExposureViolation, match="ff_a_0000"):
        ledger.assert_unseen(task_ids=["ff_a_0000"])
    with pytest.raises(ExposureViolation, match="ff_a_0000"):
        ledger.record(
            phase="sealed",
            actor="benchmark",
            purpose="claim fresh task",
            panel_ids=[PANEL_B],
            require_unseen=True,
        )


def test_sealed_access_and_unknown_identifiers_fail_closed() -> None:
    ledger = ExposureLedger.create(CORPUS)

    with pytest.raises(ExposureViolation, match="sealed"):
        ledger.record(
            phase="drill",
            actor="proposer",
            purpose="training",
            panel_ids=[PANEL_C],
            sealed_task_ids=["bd_c_0000"],
        )
    with pytest.raises(ExposureViolation, match="sealed"):
        ledger.record(
            phase="drill",
            actor="proposer",
            purpose="training",
            task_ids=["ff_a_0000"],
            sealed_panel_ids=[PANEL_A],
        )
    with pytest.raises(ExposureViolation, match="unknown task"):
        ledger.record(
            phase="dev",
            actor="proposer",
            purpose="validation",
            task_ids=["ff_unknown_0000"],
            known_task_ids=["ff_a_0000"],
        )
    opened = ledger.record(
        phase="sealed",
        actor="one-shot-runner",
        purpose="final benchmark",
        task_ids=["bd_c_0000"],
        sealed_task_ids=["bd_c_0000"],
        allow_sealed=True,
    )
    assert opened.exposed_task_ids == {"bd_c_0000"}


def test_noncanonical_panel_ids_are_rejected() -> None:
    ledger = ExposureLedger.create(CORPUS)
    for panel_id in (
        "mystery.png",
        "ff/hd_convex_0000/1/0.png",
        "hd/hd_convex_0000/1/7.png",
    ):
        with pytest.raises(ExposureIntegrityError, match="non-canonical panel"):
            ledger.record(
                phase="drill",
                actor="proposer",
                purpose="bad record",
                panel_ids=[panel_id],
            )


def test_serialized_ledger_is_write_once_and_tamper_evident(tmp_path: Path) -> None:
    ledger = ExposureLedger.create(CORPUS).record(
        phase="drill",
        actor="vision",
        purpose="description",
        task_ids=["ff_a_0000"],
        observed_at="2026-08-05T10:00:00Z",
    )
    destination = tmp_path / "ledger.json"
    ledger.write_once(destination)

    assert ExposureLedger.load(destination) == ledger
    ledger.write_once(destination)  # identical content is idempotent

    raw = json.loads(destination.read_text(encoding="utf-8"))
    raw["events"][0]["purpose"] = "rewritten history"
    destination.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ExposureIntegrityError, match="digest mismatch"):
        ExposureLedger.load(destination)
    with pytest.raises(ExposureIntegrityError, match="refusing to overwrite"):
        ledger.write_once(destination)


def test_content_addressed_filename_matches_ledger_digest(tmp_path: Path) -> None:
    ledger = ExposureLedger.create(CORPUS)
    path = ledger.write_content_addressed(tmp_path)
    assert path.name == ledger.digest.removeprefix("sha256:") + ".exposure.json"
    assert ExposureLedger.load(path) == ledger


def test_historical_import_supports_task_lists_and_event_mappings() -> None:
    ledger = ExposureLedger.create(CORPUS)
    tasks = import_historical_exposures(
        ledger,
        ["ff_a_0000", "bd_c_0000"],
        source="symbolic-runs-2026-08-05",
        known_task_ids=["ff_a_0000", "bd_c_0000"],
    )
    panels = import_historical_exposures(
        tasks,
        {
            "events": [
                {
                    "panel_id": PANEL_A,
                    "actor": "old-vlm",
                    "purpose": "saved description",
                }
            ]
        },
        source="phase-d-manifest",
        known_task_ids=["ff_a_0000", "bd_c_0000"],
        known_panel_ids=[PANEL_A],
    )

    assert len(panels.events) == 2
    assert panels.events[0].phase == "historical"
    assert panels.events[0].source == "symbolic-runs-2026-08-05"
    assert panels.events[1].actor == "old-vlm"
    assert panels.events[1].panel_ids == (PANEL_A,)

    scalar = import_historical_exposures(
        panels,
        {"task_ids": "ff_scalar_0000"},
        source="handwritten-note",
    )
    assert scalar.events[-1].task_ids == ("ff_scalar_0000",)
    assert import_historical_exposures(
        scalar, [], source="empty-inventory"
    ) is scalar


def test_deterministic_partition_is_disjoint_exhaustive_and_order_invariant() -> None:
    eligible = [f"bd_shape_{index:04d}" for index in range(12)]
    first = deterministic_partition(
        eligible,
        drill_count=7,
        dev_count=3,
        namespace="experiment-v1/basic",
    )
    second = deterministic_partition(
        reversed(eligible),
        drill_count=7,
        dev_count=3,
        namespace="experiment-v1/basic",
    )

    assert first == second
    assert len(first.drill) == 7
    assert len(first.dev) == 3
    assert len(first.sealed) == 2
    assert set(first.drill).isdisjoint(first.dev)
    assert set(first.drill).isdisjoint(first.sealed)
    assert set(first.dev).isdisjoint(first.sealed)
    assert set(first.drill + first.dev + first.sealed) == set(eligible)
    assert first.digest.startswith("sha256:")


def test_partition_rejects_non_exhaustive_or_duplicate_input() -> None:
    with pytest.raises(ValueError, match="exhaust"):
        deterministic_partition(
            ["a", "b", "c"], drill_count=1, dev_count=1, sealed_count=0
        )
    with pytest.raises(ValueError, match="duplicates"):
        deterministic_partition(["a", "a"], drill_count=1, dev_count=0)


@pytest.fixture(scope="module")
def historical_seed():
    return load_historical_exposure()


def _semantic_ledger(task_id: str) -> ExposureLedger:
    return ExposureLedger.create(CORPUS).record(
        phase="drill",
        actor="vision",
        purpose="semantic exposure test",
        task_ids=[task_id],
        observed_at="2026-08-06T10:00:00Z",
    )


def test_abstract_pair_exposure_covers_all_twenty_sibling_instances(
    historical_seed,
) -> None:
    left, right = historical_seed.abstract_partition.drill[0]
    ledger = _semantic_ledger(f"hd_{left}-{right}_0000")
    resolution = ledger.derive_exposed_semantic_keys(
        historical_seed=historical_seed,
        expected_historical_seed_digest=historical_seed.seed_digest,
    )

    assert resolution.semantic_keys == (
        SemanticDisclosureKey("abstract_pair", (left, right)),
    )
    with pytest.raises(ExposureViolation, match="semantics are not unseen"):
        ledger.assert_semantically_unseen(
            task_ids=[f"hd_{left}-{right}_0019"],
            historical_seed=historical_seed,
            expected_historical_seed_digest=historical_seed.seed_digest,
        )


def test_basic_task_opens_each_named_family_and_overlap_collides(
    historical_seed,
) -> None:
    first, shared, third = historical_seed.partition.drill[:3]
    ledger = _semantic_ledger(f"bd_{first}-{shared}_0000")
    resolution = ledger.derive_exposed_semantic_keys(
        historical_seed=historical_seed,
        expected_historical_seed_digest=historical_seed.seed_digest,
    )

    assert set(resolution.semantic_keys) == {
        SemanticDisclosureKey("basic_family", (first,)),
        SemanticDisclosureKey("basic_family", (shared,)),
        SemanticDisclosureKey(
            "basic_morphology_cluster",
            (basic_morphology_cluster_id(first),),
        ),
        SemanticDisclosureKey(
            "basic_morphology_cluster",
            (basic_morphology_cluster_id(shared),),
        ),
    }
    with pytest.raises(ExposureViolation, match="semantics are not unseen"):
        ledger.assert_semantically_unseen(
            task_ids=[f"bd_{shared}-{third}_0000"],
            historical_seed=historical_seed,
            expected_historical_seed_digest=historical_seed.seed_digest,
        )


def test_basic_numbered_morphologies_are_not_independent(historical_seed) -> None:
    blocked = set(semantic_policy_blocked_keys(historical_seed))
    assert SemanticDisclosureKey(
        "basic_morphology_cluster", ("advanced_lamp",)
    ) in blocked
    assert SemanticDisclosureKey("basic_morphology_cluster", ("bird",)) in blocked

    empty = ExposureLedger.create(CORPUS)
    for sibling in ("advanced_lamp5", "bird2", "bird4", "bird7"):
        with pytest.raises(ExposureViolation, match="semantics are not unseen"):
            empty.assert_semantically_unseen(
                task_ids=[f"bd_{sibling}_0000"],
                historical_seed=historical_seed,
                expected_historical_seed_digest=historical_seed.seed_digest,
            )


def test_basic_v2_keeps_exact_and_cluster_keys_for_a_clean_stem(
    historical_seed,
) -> None:
    resolution = ExposureLedger.create(CORPUS).assert_semantically_unseen(
        task_ids=["bd_arc_cup_0000"],
        historical_seed=historical_seed,
        expected_historical_seed_digest=historical_seed.seed_digest,
    )
    assert set(resolution.semantic_keys) == {
        SemanticDisclosureKey("basic_family", ("arc_cup",)),
        SemanticDisclosureKey("basic_morphology_cluster", ("arc_cup",)),
    }


def test_abstract_pair_key_is_ordered_and_not_attribute_wise(historical_seed) -> None:
    first = historical_seed.admissible_abstract_pairs[0]
    second = next(
        pair
        for pair in historical_seed.admissible_abstract_pairs
        if pair[0] == first[0] and pair != first
    )
    ledger = _semantic_ledger(f"hd_{first[0]}-{first[1]}_0000")

    unseen = ledger.assert_semantically_unseen(
        task_ids=[f"hd_{second[0]}-{second[1]}_0000"],
        historical_seed=historical_seed,
        expected_historical_seed_digest=historical_seed.seed_digest,
    )
    assert unseen.semantic_keys == (
        SemanticDisclosureKey("abstract_pair", second),
    )

    # The frozen official parser does not silently canonicalize reversal.  If
    # a future vocabulary admits both orders, the key's ordered tuple still
    # keeps them distinct.
    with pytest.raises(ExposureViolation, match="cannot derive requested"):
        ledger.assert_semantically_unseen(
            task_ids=[f"hd_{first[1]}-{first[0]}_0000"],
            historical_seed=historical_seed,
            expected_historical_seed_digest=historical_seed.seed_digest,
        )
    assert SemanticDisclosureKey("abstract_pair", first) != SemanticDisclosureKey(
        "abstract_pair", tuple(reversed(first))
    )


def test_abstract_singleton_and_freeform_have_separate_exact_keys(
    historical_seed,
) -> None:
    singleton = _semantic_ledger("hd_convex_0005").derive_exposed_semantic_keys(
        historical_seed=historical_seed,
        expected_historical_seed_digest=historical_seed.seed_digest,
    )
    assert singleton.semantic_keys == (
        SemanticDisclosureKey("abstract_attribute", ("convex",)),
    )

    freeform = _semantic_ledger("ff_nact2_5_0000")
    with pytest.raises(ExposureViolation, match="semantics are not unseen"):
        freeform.assert_semantically_unseen(
            task_ids=["ff_nact2_5_0299"],
            historical_seed=historical_seed,
            expected_historical_seed_digest=historical_seed.seed_digest,
        )
    distinct = freeform.assert_semantically_unseen(
        task_ids=["ff_nact3_3_0000"],
        historical_seed=historical_seed,
        expected_historical_seed_digest=historical_seed.seed_digest,
    )
    assert distinct.semantic_keys == (
        SemanticDisclosureKey("freeform_family", ("nact3_3",)),
    )


def test_panel_only_record_implies_semantic_task_exposure(historical_seed) -> None:
    left, right = historical_seed.abstract_partition.dev[0]
    task_id = f"hd_{left}-{right}_0003"
    panel_id = f"hd/{task_id}/1/0.png"
    ledger = ExposureLedger.create(CORPUS).record(
        phase="drill",
        actor="vision",
        purpose="one panel was viewed",
        panel_ids=[panel_id],
        observed_at="2026-08-06T10:00:00Z",
    )

    resolution = ledger.derive_exposed_semantic_keys(
        historical_seed=historical_seed,
        expected_historical_seed_digest=historical_seed.seed_digest,
    )
    assert resolution.task_ids == (task_id,)
    with pytest.raises(ExposureViolation, match="semantics are not unseen"):
        ledger.assert_semantically_unseen(
            task_ids=[f"hd_{left}-{right}_0012"],
            historical_seed=historical_seed,
            expected_historical_seed_digest=historical_seed.seed_digest,
        )


def test_semantic_resolution_fails_closed_on_malformed_recorded_or_requested_ids(
    historical_seed,
) -> None:
    malformed_history = _semantic_ledger("hd_not-an-official-task_0000")
    with pytest.raises(ExposureIntegrityError, match="cannot derive recorded"):
        malformed_history.derive_exposed_semantic_keys(
            historical_seed=historical_seed,
            expected_historical_seed_digest=historical_seed.seed_digest,
        )

    with pytest.raises(ExposureViolation, match="cannot derive requested"):
        ExposureLedger.create(CORPUS).assert_semantically_unseen(
            task_ids=["hd_not-an-official-task_0000"],
            historical_seed=historical_seed,
            expected_historical_seed_digest=historical_seed.seed_digest,
        )


def test_semantic_resolution_requires_and_returns_seed_or_policy_binding(
    historical_seed,
) -> None:
    ledger = _semantic_ledger("ff_nact2_5_0000")
    policy_digest = semantic_resolver_policy_digest(historical_seed)

    with pytest.raises(ExposureViolation, match="requires a precommitted"):
        ledger.derive_exposed_semantic_keys(historical_seed=historical_seed)
    with pytest.raises(ExposureViolation, match="seed differs"):
        ledger.derive_exposed_semantic_keys(
            historical_seed=historical_seed,
            expected_historical_seed_digest="sha256:" + "0" * 64,
        )
    with pytest.raises(ExposureViolation, match="policy differs"):
        ledger.derive_exposed_semantic_keys(
            historical_seed=historical_seed,
            expected_resolver_policy_digest="sha256:" + "0" * 64,
        )

    substituted_policy = replace(
        historical_seed,
        admissible_abstract_pairs=(
            *historical_seed.admissible_abstract_pairs,
            tuple(reversed(historical_seed.admissible_abstract_pairs[0])),
        ),
    )
    with pytest.raises(ExposureViolation, match="policy differs"):
        ledger.derive_exposed_semantic_keys(
            historical_seed=substituted_policy,
            expected_resolver_policy_digest=policy_digest,
        )

    resolution = ledger.derive_exposed_semantic_keys(
        historical_seed=historical_seed,
        expected_resolver_policy_digest=policy_digest,
    )
    assert resolution.historical_seed_digest == historical_seed.seed_digest
    assert resolution.resolver_policy_digest == policy_digest
    assert resolution.ledger_digest == ledger.digest
    assert set(ledger.to_dict()) == {
        "schema",
        "corpus_digest",
        "events",
        "ledger_digest",
    }
