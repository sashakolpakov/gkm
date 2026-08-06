from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace

import pytest

from bongard.admission import (
    AdmissionEvidence,
    AdmissionPolicy,
    AdmissionVerifier,
    AntiMemorizationEvaluation,
    ArchiveEntry,
    ArchivePreservationContract,
    ArchiveReplayReceipt,
    CalibrationEvaluation,
    CandidateAttachment,
    CandidateReplayReceipt,
    GateName,
    NearMissEvaluation,
    NuisanceEvaluation,
    NuisanceTrial,
    TypedAttachmentContract,
    apply_promotion,
    charge_ast_novelty,
)
from bongard.evidence import Evidence, Provenance, Uncertainty
from bongard.ir import Atom, Quantity, Relation, StaticLegCall, formula_digest
from bongard.legs import (
    PANEL,
    AffirmativeRelation,
    LegContract,
    LegRegistry,
    Transform,
    Unit,
    ValueType,
    implementation_sha256,
)
from bongard.legs.contracts import RegistrySnapshot


ANGLE = ValueType("measurement", Unit.DEGREES)
VERIFIER = "canonical-bongard-verifier"
SUITE = hashlib.sha256(b"suite-v1").hexdigest()
CANDIDATE_SUITE = hashlib.sha256(b"candidate-suite-v1").hexdigest()


def extractor(panel: dict[str, tuple[float, float]]) -> Evidence[float]:
    lower, upper = panel["angle"]
    origin = Provenance("angle-leg", "1", "deterministic", ("panel",))
    return Evidence.present(
        (lower + upper) / 2, origin, Uncertainty(lower, upper)
    )


def alternate_extractor(panel: dict[str, tuple[float, float]]) -> Evidence[float]:
    lower, upper = panel["angle"]
    origin = Provenance("alternate-angle-leg", "1", "deterministic", ("panel",))
    return Evidence.present(lower, origin, Uncertainty(lower, upper))


def setup_contracts(existing: tuple[ArchiveEntry, ...] = ()):
    registry = LegRegistry()
    reference = registry.register(
        LegContract(
            "angle",
            "1",
            (PANEL,),
            ANGLE,
            extractor,
            affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
        )
    )
    registry.freeze()
    attachment = TypedAttachmentContract.issue(
        issued_by=VERIFIER,
        registry=registry,
        boundary_types={"panel": PANEL},
    )
    archive = ArchivePreservationContract(VERIFIER, SUITE, tuple(sorted(existing)))
    formula = Atom(
        StaticLegCall(reference, ("panel",)),
        Relation.AT_LEAST,
        "angle is oblique",
        Quantity(45.0, Unit.DEGREES),
    )
    candidate = CandidateAttachment(
        "oblique-v1",
        formula,
        reference,
        "def score(x):\n    return x + 1\n",
        inspect.getsource(extractor).strip(),
    )
    return registry, attachment, archive, candidate


def valid_evidence(candidate: CandidateAttachment, *, replay=()) -> AdmissionEvidence:
    digest = candidate.digest()
    return AdmissionEvidence(
        nuisance=NuisanceEvaluation(
            VERIFIER,
            digest,
            (
                NuisanceTrial("n1", Transform.TRANSLATION, True),
                NuisanceTrial("n2", Transform.ROTATION, True),
            ),
        ),
        calibration=CalibrationEvaluation(VERIFIER, digest, 0.10, 0.25, 50),
        near_miss=NearMissEvaluation(VERIFIER, digest, 9, 10),
        anti_memorization=AntiMemorizationEvaluation(
            VERIFIER, digest, 10, 10, 0, ()
        ),
        archive_replay=tuple(replay),
        candidate_replay=CandidateReplayReceipt(
            VERIFIER,
            CANDIDATE_SUITE,
            digest,
            formula_digest(candidate.formula),
            hashlib.sha256(b"candidate replay").hexdigest(),
            24,
            True,
        ),
    )


def policy() -> AdmissionPolicy:
    return AdmissionPolicy(
        required_nuisances=frozenset(
            {Transform.TRANSLATION, Transform.ROTATION}
        ),
        candidate_replay_suite_digest=CANDIDATE_SUITE,
        incumbent_source_sha256=hashlib.sha256(
            b"def score(x):\n    return x + 1\n"
        ).hexdigest(),
        minimum_nuisance_rate=1.0,
        minimum_calibration_gain=0.05,
        minimum_calibration_samples=20,
        minimum_near_miss_rate=0.8,
        minimum_anti_memorization_rate=0.9,
        maximum_ast_novelty=100,
    )


def test_ast_novelty_charges_same_size_rewrite_but_not_formatting() -> None:
    rewrite = charge_ast_novelty(
        "def f(x):\n    return x + 1\n", "def f(x):\n    return x - 1\n"
    )
    assert rewrite.parsed
    assert rewrite.incumbent_nodes == rewrite.candidate_nodes
    assert rewrite.same_size_rewrite
    assert rewrite.charge is not None and rewrite.charge > 0

    formatting = charge_ast_novelty(
        "def f(x):\n return x + 1\n", "def f( x ):\n    return x + 1  # same AST\n"
    )
    assert formatting.charge == 0
    assert not formatting.same_size_rewrite


def test_attachment_v2_embeds_a_recomputable_static_registry() -> None:
    registry, attachment, _archive, candidate = setup_contracts()
    assert attachment.ir_version == "positive-ir/v2"
    assert attachment.registry_snapshot == registry.snapshot()
    assert attachment.registry_snapshot.digest() == attachment.registry_digest
    snapshot = RegistrySnapshot.from_data(
        attachment.to_data()["registry_snapshot"]
    )
    assert snapshot == attachment.registry_snapshot
    assert snapshot.contracts[0].source_digest == implementation_sha256(extractor)
    assert snapshot.contracts[0].operational_digest is None
    attachment.validate_static(candidate.formula)


def test_static_registry_snapshot_rejects_mutable_containers() -> None:
    registry, attachment, _archive, _candidate = setup_contracts()
    contract = attachment.registry_snapshot.contracts[0]
    with pytest.raises(TypeError, match="immutable tuple"):
        RegistrySnapshot(list(registry.snapshot().contracts))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="parameter_names must be a frozenset"):
        replace(contract, parameter_names=set())  # type: ignore[arg-type]


def test_leg_source_digest_cannot_be_overridden_by_a_caller() -> None:
    with pytest.raises(TypeError, match="source_digest"):
        LegContract(
            "forged_source",
            "1",
            (PANEL,),
            ANGLE,
            extractor,
            affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
            source_digest="0" * 64,  # type: ignore[call-arg]
        )


def test_leg_source_digest_is_recomputed_before_contract_use() -> None:
    contract = LegContract(
        "recomputed_source",
        "1",
        (PANEL,),
        ANGLE,
        extractor,
        affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
    )
    object.__setattr__(contract, "implementation", alternate_extractor)
    with pytest.raises(ValueError, match="implementation changed"):
        contract.digest()


def test_operational_digest_is_separate_from_recomputed_source_identity() -> None:
    operational = hashlib.sha256(b"frozen visual procedure").hexdigest()
    contract = LegContract(
        "configured_angle",
        "1",
        (PANEL,),
        ANGLE,
        extractor,
        affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
        operational_digest=operational,
    )
    snapshot = contract.snapshot()
    assert snapshot.source_digest == implementation_sha256(extractor)
    assert snapshot.operational_digest == operational
    assert snapshot.source_digest != snapshot.operational_digest


def test_all_gates_pass_in_one_atomic_archive_transition() -> None:
    registry, attachment, archive, candidate = setup_contracts()
    verifier = AdmissionVerifier(
        verifier_id=VERIFIER,
        registry=registry,
        attachment_contract=attachment,
        archive_contract=archive,
        policy=policy(),
    )
    decision = verifier.decide(candidate, valid_evidence(candidate))
    assert decision.accepted
    assert all(gate.passed for gate in decision.gates)
    assert archive.entries == ()  # the input contract was not mutated

    promoted = apply_promotion(decision, archive)
    assert [entry.attachment_id for entry in promoted.entries] == ["oblique-v1"]
    assert promoted.digest() == decision.archive_after.digest()  # type: ignore[union-attr]


def test_novelty_source_is_the_registered_executable_not_candidate_decoy() -> None:
    registry, attachment, archive, candidate = setup_contracts()
    decoy = replace(
        candidate,
        candidate_source="def harmless(x):\n    return x\n",
    )
    decision = AdmissionVerifier(
        verifier_id=VERIFIER,
        registry=registry,
        attachment_contract=attachment,
        archive_contract=archive,
        policy=policy(),
    ).decide(decoy, valid_evidence(decoy))
    gate = next(result for result in decision.gates if result.gate is GateName.SOURCE_BINDING)
    assert not decision.accepted
    assert not gate.passed
    assert "registered executable" in gate.detail


def test_novelty_incumbent_is_the_verifier_precommit() -> None:
    registry, attachment, archive, candidate = setup_contracts()
    substituted = replace(
        candidate,
        incumbent_source="def score(x):\n    return x - 1\n",
    )
    decision = AdmissionVerifier(
        verifier_id=VERIFIER,
        registry=registry,
        attachment_contract=attachment,
        archive_contract=archive,
        policy=policy(),
    ).decide(substituted, valid_evidence(substituted))
    gate = next(result for result in decision.gates if result.gate is GateName.SOURCE_BINDING)
    assert not decision.accepted
    assert not gate.passed
    assert "precommit" in gate.detail


def test_failed_calibration_cannot_partially_promote() -> None:
    registry, attachment, archive, candidate = setup_contracts()
    evidence = valid_evidence(candidate)
    evidence = replace(
        evidence,
        calibration=replace(evidence.calibration, candidate_brier=0.40),
    )
    decision = AdmissionVerifier(
        verifier_id=VERIFIER,
        registry=registry,
        attachment_contract=attachment,
        archive_contract=archive,
        policy=policy(),
    ).decide(candidate, evidence)
    assert not decision.accepted
    assert decision.archive_after is None
    assert not next(
        gate for gate in decision.gates if gate.gate.value == "calibration_vs_baseline"
    ).passed
    with pytest.raises(ValueError, match="rejected"):
        apply_promotion(decision, archive)


def test_candidate_replay_is_a_real_identity_bound_gate() -> None:
    registry, attachment, archive, candidate = setup_contracts()
    evidence = valid_evidence(candidate)
    evidence = replace(
        evidence,
        candidate_replay=replace(evidence.candidate_replay, passed=False),
    )
    decision = AdmissionVerifier(
        verifier_id=VERIFIER,
        registry=registry,
        attachment_contract=attachment,
        archive_contract=archive,
        policy=policy(),
    ).decide(candidate, evidence)
    gate = next(
        result for result in decision.gates
        if result.gate is GateName.CANDIDATE_REPLAY
    )
    assert not decision.accepted
    assert not gate.passed

    wrong_suite = replace(
        evidence,
        candidate_replay=replace(
            evidence.candidate_replay,
            passed=True,
            replay_suite_digest=hashlib.sha256(b"wrong suite").hexdigest(),
        ),
    )
    second = AdmissionVerifier(
        verifier_id=VERIFIER,
        registry=registry,
        attachment_contract=attachment,
        archive_contract=archive,
        policy=policy(),
    ).decide(candidate, wrong_suite)
    assert not next(
        result for result in second.gates
        if result.gate is GateName.CANDIDATE_REPLAY
    ).passed


def test_full_archive_replay_is_mandatory_and_exact() -> None:
    old = ArchiveEntry(
        "old-leg",
        hashlib.sha256(b"old candidate").hexdigest(),
        hashlib.sha256(b"old formula").hexdigest(),
        hashlib.sha256(b"old replay").hexdigest(),
        12,
    )
    registry, attachment, archive, candidate = setup_contracts((old,))
    verifier = AdmissionVerifier(
        verifier_id=VERIFIER,
        registry=registry,
        attachment_contract=attachment,
        archive_contract=archive,
        policy=policy(),
    )
    missing = verifier.decide(candidate, valid_evidence(candidate))
    assert not missing.accepted
    archive_gate = next(
        gate for gate in missing.gates if gate.gate.value == "full_archive_replay"
    )
    assert "missing=['old-leg']" in archive_gate.detail

    receipt = ArchiveReplayReceipt(
        VERIFIER,
        SUITE,
        old.attachment_id,
        old.candidate_digest,
        old.formula_digest,
        old.replay_digest,
        old.case_count,
        True,
    )
    passed = verifier.decide(candidate, valid_evidence(candidate, replay=(receipt,)))
    assert passed.accepted


def test_receipts_are_bound_to_candidate_and_verifier() -> None:
    registry, attachment, archive, candidate = setup_contracts()
    evidence = valid_evidence(candidate)
    forged = replace(
        evidence,
        near_miss=replace(
            evidence.near_miss,
            candidate_digest=hashlib.sha256(b"other").hexdigest(),
        ),
    )
    decision = AdmissionVerifier(
        verifier_id=VERIFIER,
        registry=registry,
        attachment_contract=attachment,
        archive_contract=archive,
        policy=policy(),
    ).decide(candidate, forged)
    assert not decision.accepted
    assert not decision.gates[0].passed


def test_anti_memorization_requires_zero_overlap_and_no_identifier_hits() -> None:
    registry, attachment, archive, candidate = setup_contracts()
    evidence = valid_evidence(candidate)
    contaminated = replace(
        evidence,
        anti_memorization=replace(
            evidence.anti_memorization,
            train_query_overlap=1,
            forbidden_identifier_hits=("bd_017",),
        ),
    )
    decision = AdmissionVerifier(
        verifier_id=VERIFIER,
        registry=registry,
        attachment_contract=attachment,
        archive_contract=archive,
        policy=policy(),
    ).decide(candidate, contaminated)
    assert not decision.accepted
    gate = next(
        gate for gate in decision.gates if gate.gate.value == "anti_memorization"
    )
    assert not gate.passed


def test_apply_is_compare_and_swap_against_archive_digest() -> None:
    registry, attachment, archive, candidate = setup_contracts()
    decision = AdmissionVerifier(
        verifier_id=VERIFIER,
        registry=registry,
        attachment_contract=attachment,
        archive_contract=archive,
        policy=policy(),
    ).decide(candidate, valid_evidence(candidate))
    changed = ArchivePreservationContract(
        VERIFIER,
        hashlib.sha256(b"different-suite").hexdigest(),
    )
    with pytest.raises(ValueError, match="changed after admission"):
        apply_promotion(decision, changed)
