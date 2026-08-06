from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bongard.admission import TypedAttachmentContract
from bongard.artifacts import atom_paths, canonical_digest
from bongard.benchmark import (
    BenchmarkProtocolError,
    EpisodeStatus,
    ObservationInput,
    ProposedRule,
    SUPPORT_PROTOTYPE_PREDICATE_MODE,
    SealedMutationError,
    SealedTestGuard,
    SupportInput,
    SupportGatePolicy,
    SupportGateMeasurement,
    SupportGateMode,
    SupportGateResult,
    prepare_episode,
    run_episode,
    score_results,
)
from bongard.corpus import ShapeBongardCorpus
from bongard.evidence import Disposition, Evidence, Provenance
from bongard.ir import Atom, Relation, StaticLegCall
from bongard.legs import BOOLEAN_WITNESS, PANEL, LegContract, LegRegistry


PNG = b"\x89PNG\r\n\x1a\n"
VERIFIER = "canonical-bongard-verifier"
TEST_GATE = SupportGatePolicy.verifier_test_bypass("unit fixture")


def test_support_gate_policy_keeps_legacy_bytes_and_separates_prototype_replay() -> None:
    empirical = SupportGatePolicy.empirical()
    assert canonical_digest(empirical.to_data()) == (
        "93e976b9f18f517fc3a1d109514bb76be13866680834e8e0c72e6cdb696b15ef"
    )
    assert empirical.to_data() == {
        "version": "headless-hybrid-support-replay/v2",
        "mode": "empirical_replay",
        "reason": None,
        "call_count": 12,
        "positive_count": 6,
        "negative_count": 6,
        "image_name": "query.png",
        "positive_outcome": "present",
        "negative_outcome": "nonmatch",
        "nonmatch_certificate_semantics": (
            "archived_model_nonmatch_for_frozen_operational_claim"
        ),
        "nonmatch_reason_semantics": (
            "optional_overall_model_summary_bound_inside_certificate"
        ),
        "nonmatch_cue_keyed_findings_required": True,
        "nonmatch_visibility_statement_required": True,
        "fresh_isolated_transport_per_panel": True,
        "polarity_flip_allowed": False,
    }
    assert canonical_digest(TEST_GATE.to_data()) == (
        "353f69769bd75a3f5ed6182fb62619c744af001437758af507aeb35c372d9bd2"
    )

    prototype = SupportGatePolicy.prototype()
    assert prototype.mode is SupportGateMode.SUPPORT_PROTOTYPE_REPLAY
    assert prototype.to_data() == {
        "version": "support-prototype-replay/v1",
        "mode": "support_prototype_replay",
        "reason": None,
        "call_count": 12,
        "positive_count": 6,
        "negative_count": 6,
        "extractor_input_contract": (
            "panel_bytes_only_no_task_candidate_side_or_role_context_v1"
        ),
        "positive_outcome": "present",
        "negative_outcome": "certified_absent",
        "certified_absence_semantics": (
            "operational_contrastive_nonmatch_for_frozen_support_prototype"
        ),
        "dispositions": [
            "present",
            "certified_absent",
            "indeterminate",
            "error",
        ],
        "fresh_candidate_independent_extraction_per_panel": True,
        "fresh_frozen_predicate_evaluation_per_panel": True,
        "polarity_flip_allowed": False,
    }
    encoded = json.dumps(prototype.to_data(), sort_keys=True)
    assert "codex" not in encoded.lower()
    assert "cue" not in encoded.lower()
    assert "polarity_flip_allowed\": false" in encoded


def test_support_gate_policy_rejects_cross_version_or_bypass_semantics() -> None:
    with pytest.raises(ValueError, match="support-prototype replay policy"):
        SupportGatePolicy(SupportGateMode.SUPPORT_PROTOTYPE_REPLAY)
    with pytest.raises(ValueError, match="has no bypass reason"):
        SupportGatePolicy(
            SupportGateMode.SUPPORT_PROTOTYPE_REPLAY,
            reason="candidate requested bypass",
            version="support-prototype-replay/v1",
        )
    with pytest.raises(ValueError, match="empirical support replay policy"):
        SupportGatePolicy(
            SupportGateMode.EMPIRICAL_REPLAY,
            version="support-prototype-replay/v1",
        )


def _make_task(root: Path, task_id: str) -> None:
    for label, marker in (("1", b"POS"), ("0", b"NEG")):
        directory = root / "ff" / "images" / task_id / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(7):
            # Deliberately no task id, concept, or split string in file bytes.
            (directory / f"{index}.png").write_bytes(
                PNG + marker + b":" + str(index).encode("ascii")
            )


def _corpus(tmp_path: Path, *, test: bool = False):
    root = tmp_path / "ShapeBongard_V2"
    task_id = "ff_secret_concept_0000"
    _make_task(root, task_id)
    split = {"test_ff": [task_id]} if test else {"train": [task_id]}
    (root / "ShapeBongard_V2_split.json").write_text(
        json.dumps(split), encoding="utf-8"
    )
    corpus = ShapeBongardCorpus.from_root(root)
    return corpus, task_id


def _unused_leg(panel: object) -> Evidence[bool]:
    return Evidence.error(
        Provenance("fixture-leg", "1", "unused"),
        "UnexpectedCall",
        "injected observer should provide atom evidence",
    )


def _replacement_leg(panel: object) -> Evidence[bool]:
    return Evidence.present(True, Provenance("replacement-leg", "1", "tampered"))


def _candidate() -> ProposedRule:
    registry = LegRegistry()
    reference = registry.register(
        LegContract(
            "fixture_rule",
            "1",
            (PANEL,),
            BOOLEAN_WITNESS,
            _unused_leg,
        )
    )
    registry.freeze()
    formula = Atom(
        StaticLegCall(reference, ("panel",)),
        Relation.PRESENT,
        "the affirmative fixture mark is visible",
    )
    attachment = TypedAttachmentContract.issue(
        issued_by=VERIFIER,
        registry=registry,
        boundary_types={"panel": PANEL},
    )
    return ProposedRule(
        proposal_id="fixture-proposal",
        proposer_digest=hashlib.sha256(b"support-only fixture proposer").hexdigest(),
        formula=formula,
        registry=registry,
        attachment_contract=attachment,
    )


class RecordingProposer:
    def __init__(self, task_id: str, events: list[str]) -> None:
        self.task_id = task_id
        self.events = events
        self.calls = 0

    def propose(self, support: SupportInput) -> ProposedRule:
        self.calls += 1
        self.events.append("propose")
        assert self.task_id not in repr(support)
        assert [path.name for path in support.positive_paths] == [
            f"pos_{index}.png" for index in range(6)
        ]
        assert [path.name for path in support.negative_paths] == [
            f"neg_{index}.png" for index in range(6)
        ]
        parent = support.positive_paths[0].parent
        assert sorted(path.name for path in parent.iterdir()) == [
            *(f"neg_{index}.png" for index in range(6)),
            *(f"pos_{index}.png" for index in range(6)),
        ]
        assert all(path.parent == parent for path in support.negative_paths)
        return _candidate()


class PixelObserver:
    def __init__(self, task_id: str, events: list[str]) -> None:
        self.task_id = task_id
        self.events = events
        self.inputs: list[ObservationInput] = []

    def observe(self, query: ObservationInput):
        self.events.append("observe")
        self.inputs.append(query)
        assert self.task_id not in repr(query)
        assert query.query_id == "query"
        assert query.panel.blob_id == "query-panel"
        assert query.panel_path.name == "query.png"
        assert [item.name for item in query.panel_path.parent.iterdir()] == ["query.png"]
        assert not hasattr(query, "positive")
        assert not hasattr(query, "task_id")
        assert not hasattr(query, "split")
        payload = query.panel_path.read_bytes()
        provenance = Provenance(
            "fixture-observer",
            "1",
            "single-neutral-panel",
            input_digests=(query.panel.sha256,),
        )
        path = atom_paths(query.freeze.formula)[0]
        if b"POS:" in payload:
            return {path: Evidence.present(True, provenance)}
        return {path: Evidence.certified_absent(provenance, "NEG marker visible")}


class PixelObserverFactory:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.instances: list[PixelObserver] = []

    def create_observer(self) -> PixelObserver:
        observer = PixelObserver(self.task_id, [])
        self.instances.append(observer)
        return observer


class GateReplayObserver:
    def __init__(self, owner: "GateReplayProposer", index: int) -> None:
        self.owner = owner
        self.index = index
        self.calls = 0

    def observe_support(self, panel: ObservationInput) -> SupportGateMeasurement:
        self.calls += 1
        self.owner.support_calls += 1
        assert panel.query_id == "query"
        assert panel.panel.blob_id == "query-panel"
        assert panel.panel_path.name == "query.png"
        assert [item.name for item in panel.panel_path.parent.iterdir()] == ["query.png"]
        assert not hasattr(panel, "positive")
        payload = panel.panel_path.read_bytes()
        pixel_positive = b"POS:" in payload
        disposition = "present" if pixel_positive else "nonmatch"
        if self.owner.behaviour == "inverted":
            disposition = "nonmatch" if pixel_positive else "present"
        elif self.owner.behaviour == "partial" and self.index == 0:
            disposition = "present" if disposition == "nonmatch" else "nonmatch"
        elif self.owner.behaviour in {"indeterminate", "error"} and self.index == 0:
            disposition = self.owner.behaviour
        provenance = Provenance(
            "fixture-support-observer",
            "1",
            "one-neutral-image",
            (panel.panel.sha256,),
        )
        if disposition == "present":
            evidence = Evidence.present(True, provenance)
        elif disposition == "nonmatch":
            evidence = Evidence.certified_absent(
                provenance,
                "archived fixture model nonmatch for frozen operational claim",
            )
        elif disposition == "indeterminate":
            evidence = Evidence.indeterminate(provenance, "fixture ambiguity")
        else:
            evidence = Evidence.error(provenance, "FixtureError", "fixture failure")
        return SupportGateMeasurement(
            evidence=evidence,
            observer_artifact={
                "schema": "fixture-support-observation/v1",
                "panel_sha256": panel.panel.sha256,
                "disposition": disposition,
            },
        )


class GateReplayProposer(RecordingProposer):
    requires_empirical_support_gate = True

    def __init__(self, task_id: str, behaviour: str = "aligned") -> None:
        super().__init__(task_id, [])
        self.behaviour = behaviour
        self.support_calls = 0
        self.support_instances: list[GateReplayObserver] = []

    def create_support_observer(self) -> GateReplayObserver:
        observer = GateReplayObserver(self, len(self.support_instances))
        self.support_instances.append(observer)
        return observer


class AbstainingObserver:
    def __init__(self, disposition: str = "indeterminate") -> None:
        self.disposition = disposition

    def observe(self, query: ObservationInput):
        provenance = Provenance(
            "fixture-observer", "1", "abstain", (query.panel.sha256,)
        )
        path = atom_paths(query.freeze.formula)[0]
        if self.disposition == "error":
            return {path: Evidence.error(provenance, "VisionError", "no judgment")}
        return {path: Evidence.indeterminate(provenance, "borderline image")}


class StatefulObserver:
    """Would change its answer if one instance saw both query calls."""

    def __init__(self) -> None:
        self.calls = 0

    def observe(self, query: ObservationInput):
        self.calls += 1
        provenance = Provenance(
            "stateful-fixture", "1", "call-count", (query.panel.sha256,)
        )
        path = atom_paths(query.freeze.formula)[0]
        if self.calls == 1:
            return {path: Evidence.present(True, provenance)}
        return {path: Evidence.certified_absent(provenance, "second call")}


class CombinedEpisodeSession:
    """Fixture matching the canonical CLI's proposer/observer object shape."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.proposed = False
        self._observations: dict[str, str] = {}

    def propose(self, support: SupportInput) -> ProposedRule:
        self.proposed = True
        return _candidate()

    def observe(self, query: ObservationInput):
        assert self.proposed
        assert self._observations == {}
        self._observations[query.query_id] = query.panel.sha256
        provenance = Provenance(
            "combined-session", "1", "isolated", (query.panel.sha256,)
        )
        path = atom_paths(query.freeze.formula)[0]
        payload = query.panel_path.read_bytes()
        if b"POS:" in payload:
            return {path: Evidence.present(True, provenance)}
        return {path: Evidence.certified_absent(provenance, "NEG marker visible")}


def test_episode_selection_is_deterministic_exactly_six_plus_six(tmp_path: Path) -> None:
    corpus, task_id = _corpus(tmp_path)
    manifest = corpus.build_manifest()
    nonce = "a" * 64
    first = prepare_episode(
        corpus,
        task_id,
        seed="benchmark-seed",
        corpus_manifest=manifest,
        label_seal_nonce=nonce,
    )
    second = prepare_episode(
        corpus,
        task_id,
        seed="benchmark-seed",
        corpus_manifest=manifest,
        label_seal_nonce=nonce,
    )

    assert first.digest == second.digest
    assert first.digest == canonical_digest(first.to_data())
    assert first.to_data() == second.to_data()
    assert set(first.to_data()) == {
        "version",
        "task_id",
        "family",
        "split",
        "regime",
        "run_id",
        "verifier_id",
        "seed_digest",
        "corpus_digest",
        "task_manifest_digest",
        "support_commitment_digest",
        "latent_query_digest",
        "label_commitment_digest",
    }
    assert first.support.digest() == second.support.digest()
    assert first.latent_query_digest == second.latent_query_digest
    assert len(first.support.support) == 12
    assert sum(example.positive for example in first.support.support) == 6
    assert len(first.queries) == 2
    assert {source.positive for source in first._query_sources} == {False, True}
    support_hashes = {example.panel.sha256 for example in first.support.support}
    query_hashes = {query.panel.sha256 for query in first.queries}
    assert support_hashes.isdisjoint(query_hashes)

    hidden_first = prepare_episode(
        corpus, task_id, seed="benchmark-seed", corpus_manifest=manifest
    )
    hidden_second = prepare_episode(
        corpus, task_id, seed="benchmark-seed", corpus_manifest=manifest
    )
    assert hidden_first.latent_query_digest == hidden_second.latent_query_digest
    assert hidden_first.label_commitment_digest != hidden_second.label_commitment_digest
    assert hidden_first.digest != hidden_second.digest

    alternatives = {
        prepare_episode(corpus, task_id, seed=f"seed-{index}", corpus_manifest=manifest)
        .latent_query_digest
        for index in range(20)
    }
    assert len(alternatives) > 1


def test_episode_plan_optionally_commits_pre_support_predicate_policy(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    manifest = corpus.build_manifest()
    nonce = "b" * 64
    policy_digest = hashlib.sha256(b"fixed support-prototype policy").hexdigest()
    legacy = prepare_episode(
        corpus,
        task_id,
        seed="policy-commitment",
        corpus_manifest=manifest,
        label_seal_nonce=nonce,
    )
    committed = prepare_episode(
        corpus,
        task_id,
        seed="policy-commitment",
        corpus_manifest=manifest,
        label_seal_nonce=nonce,
        predicate_mode=SUPPORT_PROTOTYPE_PREDICATE_MODE,
        predicate_policy_digest=policy_digest,
    )

    assert "predicate_mode" not in legacy.to_data()
    assert "predicate_policy_digest" not in legacy.to_data()
    assert committed.to_data() == {
        **legacy.to_data(),
        "predicate_mode": SUPPORT_PROTOTYPE_PREDICATE_MODE,
        "predicate_policy_digest": policy_digest,
    }
    assert committed.digest == canonical_digest(committed.to_data())
    assert committed.digest != legacy.digest
    assert committed.support == legacy.support
    assert committed.queries == legacy.queries

    changed = prepare_episode(
        corpus,
        task_id,
        seed="policy-commitment",
        corpus_manifest=manifest,
        label_seal_nonce=nonce,
        predicate_mode=SUPPORT_PROTOTYPE_PREDICATE_MODE,
        predicate_policy_digest=hashlib.sha256(b"changed policy").hexdigest(),
    )
    assert changed.digest != committed.digest


@pytest.mark.parametrize(
    ("predicate_mode", "predicate_policy_digest", "message"),
    [
        (SUPPORT_PROTOTYPE_PREDICATE_MODE, None, "committed together"),
        (None, "a" * 64, "committed together"),
        ("hybrid", "a" * 64, "unsupported predicate mode"),
        (
            SUPPORT_PROTOTYPE_PREDICATE_MODE,
            "sha256:" + "a" * 64,
            "unprefixed lowercase SHA-256",
        ),
        (
            SUPPORT_PROTOTYPE_PREDICATE_MODE,
            "A" * 64,
            "unprefixed lowercase SHA-256",
        ),
    ],
)
def test_episode_plan_rejects_incomplete_or_noncanonical_predicate_policy(
    tmp_path: Path,
    predicate_mode: str | None,
    predicate_policy_digest: str | None,
    message: str,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    with pytest.raises(ValueError, match=message):
        prepare_episode(
            corpus,
            task_id,
            seed="invalid-policy",
            label_seal_nonce="c" * 64,
            predicate_mode=predicate_mode,
            predicate_policy_digest=predicate_policy_digest,
        )


def test_runner_rejects_policy_mismatch_before_support_callback(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    policy_digest = hashlib.sha256(b"committed prototype policy").hexdigest()
    plan = prepare_episode(
        corpus,
        task_id,
        seed="policy-bound-runner",
        predicate_mode=SUPPORT_PROTOTYPE_PREDICATE_MODE,
        predicate_policy_digest=policy_digest,
    )

    class MismatchedProposer:
        predicate_mode = SUPPORT_PROTOTYPE_PREDICATE_MODE
        predicate_policy_digest = hashlib.sha256(b"other policy").hexdigest()

        def __init__(self) -> None:
            self.called = False

        def propose(self, support):
            del support
            self.called = True
            raise AssertionError("support must not be released")

    proposer = MismatchedProposer()
    with pytest.raises(BenchmarkProtocolError, match="committed episode plan"):
        run_episode(
            plan,
            proposer,
            PixelObserverFactory(task_id),
            support_gate_policy=SupportGatePolicy.prototype(),
        )
    assert not proposer.called


def test_runner_freezes_before_two_isolated_queries_and_scores_both(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed="one-shot")
    events: list[str] = []
    proposer = RecordingProposer(task_id, events)
    observers = PixelObserverFactory(task_id)

    result = run_episode(plan, proposer, observers, support_gate_policy=TEST_GATE)

    assert proposer.calls == 1
    assert events == ["propose"]
    assert len(observers.instances) == 2
    assert all(instance.events == ["observe"] for instance in observers.instances)
    inputs = [instance.inputs[0] for instance in observers.instances]
    assert len({id(instance) for instance in observers.instances}) == 2
    assert len({id(item.registry) for item in inputs}) == 2
    assert len({item.panel_path.parent for item in inputs}) == 2
    assert {item.query_id for item in inputs} == {"query"}
    assert {item.panel.blob_id for item in inputs} == {"query-panel"}
    assert result.status is EpisodeStatus.COMPLETE
    assert result.phases == (
        "plan_committed",
        "support_released",
        "proposal_fixed",
        "support_gate_replayed",
        "proposal_frozen",
        "query_released",
        "predictions_committed",
        "labels_revealed",
        "cold_replay_verified",
    )
    assert result.phases.index("proposal_frozen") < result.phases.index("query_released")
    assert result.phases.index("predictions_committed") < result.phases.index(
        "labels_revealed"
    )
    assert result.score.image_accuracy == 1.0
    assert result.score.puzzle_accuracy == 1.0
    assert result.bundle is not None
    assert all("positive" not in query.to_data() for query in result.bundle.release.queries)
    assert (
        result.bundle.labels.prediction_commitment_digest
        == result.bundle.predictions.digest()
    )
    assert result.bundle.verify().predictions_match


def test_empirical_support_gate_uses_exactly_twelve_fresh_neutral_calls(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed="support-gate-aligned")
    proposer = GateReplayProposer(task_id)
    query_observers = PixelObserverFactory(task_id)

    result = run_episode(
        plan,
        proposer,
        query_observers,
        support_gate_policy=SupportGatePolicy.empirical(),
    )

    assert result.status is EpisodeStatus.COMPLETE
    assert proposer.support_calls == 12
    assert len(proposer.support_instances) == 12
    assert len({id(item) for item in proposer.support_instances}) == 12
    assert [item.calls for item in proposer.support_instances] == [1] * 12
    assert len(query_observers.instances) == 2
    assert result.support_gate is not None
    assert result.support_gate.result is SupportGateResult.ALIGNED
    assert result.support_gate.forward_matches == 12
    assert result.support_gate.reverse_matches == 0
    assert result.support_gate.transport_attempt_count == 12
    assert result.proposal_freeze is not None
    assert result.proposal_freeze.support_gate_digest == result.support_gate.digest


def test_prototype_support_gate_uses_the_same_strict_classifier(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed="prototype-support-gate")
    proposer = GateReplayProposer(task_id)
    query_observers = PixelObserverFactory(task_id)

    result = run_episode(
        plan,
        proposer,
        query_observers,
        support_gate_policy=SupportGatePolicy.prototype(),
    )

    assert result.status is EpisodeStatus.COMPLETE
    assert result.support_gate is not None
    assert result.support_gate.policy == SupportGatePolicy.prototype()
    assert result.support_gate.result is SupportGateResult.ALIGNED
    assert result.support_gate.forward_matches == 12
    assert result.support_gate.reverse_matches == 0
    assert result.support_gate.present_count == 6
    assert result.support_gate.nonmatch_count == 6
    assert result.support_gate.indeterminate_count == 0
    assert result.support_gate.error_count == 0
    assert result.support_gate.transport_attempt_count == 12
    assert proposer.support_calls == 12
    assert len(query_observers.instances) == 2


@pytest.mark.parametrize("behaviour", ["indeterminate", "error"])
def test_prototype_support_gate_preserves_unresolved_dispositions(
    tmp_path: Path,
    behaviour: str,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(
        corpus, task_id, seed=f"prototype-support-gate-{behaviour}"
    )
    proposer = GateReplayProposer(task_id, behaviour)
    query_observers = PixelObserverFactory(task_id)

    result = run_episode(
        plan,
        proposer,
        query_observers,
        support_gate_policy=SupportGatePolicy.prototype(),
    )

    assert result.status is EpisodeStatus.SUPPORT_REJECTED
    assert result.support_gate is not None
    assert result.support_gate.result is SupportGateResult.OBSERVER_FAILURE
    assert result.support_gate.indeterminate_count == (
        1 if behaviour == "indeterminate" else 0
    )
    assert result.support_gate.error_count == (1 if behaviour == "error" else 0)
    assert query_observers.instances == []
    assert "query_released" not in result.phases


@pytest.mark.parametrize(
    ("behaviour", "expected"),
    [
        ("inverted", SupportGateResult.MISORIENTED),
        ("partial", SupportGateResult.UNSUPPORTED),
        ("indeterminate", SupportGateResult.OBSERVER_FAILURE),
        ("error", SupportGateResult.OBSERVER_FAILURE),
    ],
)
def test_support_gate_rejects_without_flip_or_query_release(
    tmp_path: Path,
    behaviour: str,
    expected: SupportGateResult,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed=f"support-gate-{behaviour}")
    proposer = GateReplayProposer(task_id, behaviour)
    query_observers = PixelObserverFactory(task_id)

    result = run_episode(
        plan,
        proposer,
        query_observers,
        support_gate_policy=SupportGatePolicy.empirical(),
    )

    assert proposer.support_calls == 12
    assert result.status is EpisodeStatus.SUPPORT_REJECTED
    assert result.bundle is None
    assert result.support_gate is not None
    assert result.support_gate.result is expected
    assert "query_released" not in result.phases
    assert query_observers.instances == []
    if behaviour == "inverted":
        assert result.support_gate.reverse_matches == 12
        assert result.support_gate.forward_matches == 0
        assert result.failure is not None
        assert result.failure.reason == "misoriented"


def test_empirical_headless_marker_cannot_use_test_gate_bypass(tmp_path: Path) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed="support-gate-no-bypass")
    proposer = GateReplayProposer(task_id)
    queries = PixelObserverFactory(task_id)

    result = run_episode(
        plan,
        proposer,
        queries,
        support_gate_policy=TEST_GATE,
    )

    assert result.status is EpisodeStatus.SUPPORT_REJECTED
    assert result.bundle is None
    assert queries.instances == []
    assert result.failure is not None
    assert "cannot bypass" in result.failure.reason


def test_legacy_observer_is_snapshotted_into_independent_query_sessions(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed="state-isolation")
    observer = StatefulObserver()

    result = run_episode(
        plan, RecordingProposer(task_id, []), observer, support_gate_policy=TEST_GATE
    )

    assert result.status is EpisodeStatus.COMPLETE
    assert observer.calls == 0
    assert result.bundle is not None
    assert [
        prediction.disposition for prediction in result.bundle.predictions.predictions
    ] == [Disposition.PRESENT, Disposition.PRESENT]


def test_combined_cli_session_collects_receipts_only_after_isolated_calls(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed="combined-session")
    session = CombinedEpisodeSession(task_id)

    result = run_episode(plan, session, session, support_gate_policy=TEST_GATE)

    assert result.status is EpisodeStatus.COMPLETE
    assert result.score.image_correct == 2
    assert set(session._observations) == {"query-0", "query-1"}
    assert set(session._observations.values()) == {
        query.panel.sha256 for query in plan.queries
    }


def test_factory_reusing_one_observer_fails_before_query_release(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed="reused-observer")

    class ReusingFactory:
        def __init__(self) -> None:
            self.observer = StatefulObserver()

        def create_observer(self) -> StatefulObserver:
            return self.observer

    factory = ReusingFactory()
    result = run_episode(
        plan, RecordingProposer(task_id, []), factory, support_gate_policy=TEST_GATE
    )

    assert result.status is EpisodeStatus.PROPOSAL_ERROR
    assert "query_released" not in result.phases
    assert factory.observer.calls == 0
    assert result.failure is not None
    assert "reused one callback object" in result.failure.reason


def test_registry_implementation_is_revalidated_after_each_observer(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed="registry-mutation")

    class CapturingProposer(RecordingProposer):
        rule: ProposedRule | None = None

        def propose(self, support: SupportInput) -> ProposedRule:
            self.rule = super().propose(support)
            return self.rule

    proposer = CapturingProposer(task_id, [])

    class MutatingObserver:
        def __init__(self) -> None:
            self.calls = 0

        def observe(self, query: ObservationInput):
            self.calls += 1
            assert proposer.rule is not None
            contract = proposer.rule.registry.contracts()[0]
            object.__setattr__(contract, "implementation", _replacement_leg)
            provenance = Provenance(
                "registry-mutator", "1", "malicious", (query.panel.sha256,)
            )
            return {atom_paths(query.freeze.formula)[0]: Evidence.present(True, provenance)}

    class MutatingFactory:
        def __init__(self) -> None:
            self.instances: list[MutatingObserver] = []

        def create_observer(self) -> MutatingObserver:
            instance = MutatingObserver()
            self.instances.append(instance)
            return instance

    factory = MutatingFactory()
    with pytest.raises(BenchmarkProtocolError, match="changed a frozen registry"):
        run_episode(plan, proposer, factory, support_gate_policy=TEST_GATE)
    assert [instance.calls for instance in factory.instances] == [1, 0]


@pytest.mark.parametrize("disposition", ["indeterminate", "error"])
def test_unresolved_observation_is_wrong_even_for_the_negative_query(
    tmp_path: Path, disposition: str
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed="fail-closed")

    result = run_episode(
        plan,
        RecordingProposer(task_id, []),
        AbstainingObserver(disposition),
        support_gate_policy=TEST_GATE,
    )

    assert result.status is EpisodeStatus.COMPLETE
    assert result.score.image_correct == 0
    assert not result.score.puzzle_correct
    assert result.score.determinate == 0
    assert result.score.abstentions == 2
    assert result.score.errors == (2 if disposition == "error" else 0)
    assert result.bundle is not None
    assert all(
        prediction.positive is None
        for prediction in result.bundle.predictions.predictions
    )


def test_proposal_failure_never_releases_query_and_counts_as_two_errors(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    plan = prepare_episode(corpus, task_id, seed="proposal-fails")

    class FailedProposer:
        def propose(self, support: SupportInput) -> ProposedRule:
            raise RuntimeError("headless proposer unavailable")

    class ForbiddenObserver:
        def observe(self, query: ObservationInput):
            raise AssertionError("query was released after proposal failure")

    result = run_episode(
        plan, FailedProposer(), ForbiddenObserver(), support_gate_policy=TEST_GATE
    )

    assert result.status is EpisodeStatus.PROPOSAL_ERROR
    assert result.bundle is None
    assert "query_released" not in result.phases
    assert result.phases[-1] == "proposal_failed"
    assert result.score.image_correct == 0
    assert result.score.abstentions == result.score.errors == 2


def test_sealed_test_requires_guard_and_detects_callback_mutation(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path, test=True)
    manifest = corpus.build_manifest()
    guard = SealedTestGuard.capture(corpus, corpus_manifest=manifest)
    plan = prepare_episode(corpus, task_id, seed="sealed", corpus_manifest=manifest)

    with pytest.raises(BenchmarkProtocolError, match="require.*SealedTestGuard"):
        run_episode(
            plan,
            RecordingProposer(task_id, []),
            PixelObserver(task_id, []),
            support_gate_policy=TEST_GATE,
        )

    target = corpus.task(task_id).positive_paths[0]

    class MutatingProposer(RecordingProposer):
        def propose(self, support: SupportInput) -> ProposedRule:
            proposal = super().propose(support)
            target.write_bytes(PNG + b"MUTATED")
            return proposal

    observer = PixelObserver(task_id, [])
    with pytest.raises(SealedMutationError, match="panel bytes changed"):
        run_episode(
            plan,
            MutatingProposer(task_id, []),
            observer,
            support_gate_policy=TEST_GATE,
            sealed_guard=guard,
        )
    assert observer.inputs == []


def test_guard_detects_split_mutation_and_aggregate_includes_failures(
    tmp_path: Path,
) -> None:
    sealed_root = tmp_path / "sealed"
    corpus, task_id = _corpus(sealed_root, test=True)
    manifest = corpus.build_manifest()
    guard = SealedTestGuard.capture(corpus, corpus_manifest=manifest)
    plan = prepare_episode(corpus, task_id, seed="sealed", corpus_manifest=manifest)
    good = run_episode(
        plan,
        RecordingProposer(task_id, []),
        PixelObserver(task_id, []),
        support_gate_policy=TEST_GATE,
        sealed_guard=guard,
    )

    split_path = corpus.split.source_path
    assert split_path is not None
    split_path.write_text(json.dumps({"train": [task_id]}), encoding="utf-8")
    with pytest.raises(SealedMutationError, match="split bytes changed"):
        guard.verify_all()

    open_root = tmp_path / "open"
    other_corpus, other_task = _corpus(open_root)
    failed_plan = prepare_episode(other_corpus, other_task, seed="failure")
    failed = run_episode(
        failed_plan,
        RecordingProposer(other_task, []),
        AbstainingObserver("error"),
        support_gate_policy=TEST_GATE,
    )
    aggregate = score_results((good, failed))
    assert aggregate.episode_total == aggregate.episode_complete == 2
    assert aggregate.image_correct == 2
    assert aggregate.image_total == 4
    assert aggregate.image_accuracy == 0.5
    assert aggregate.puzzle_correct == 1
    assert aggregate.puzzle_accuracy == 0.5
    assert aggregate.abstentions == aggregate.errors == 2
    assert dict(aggregate.dispositions) == {
        "certified_absent": 1,
        "error": 2,
        "present": 1,
    }
