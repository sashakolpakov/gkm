from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from bongard.admission import ArchivePreservationContract, TypedAttachmentContract
from bongard.artifacts import (
    ArtifactTamperError,
    AtomReplayInput,
    BlobRef,
    ColdReplayInputs,
    LabelReveal,
    PredictionCommitment,
    ProposalFreeze,
    QueryPanel,
    QueryRelease,
    QueryReplayInput,
    RevealedLabel,
    RunArtifactBundle,
    SupportCommitment,
    SupportExample,
    TruthEvidenceRecord,
    VerifiedRunArchive,
    atom_paths,
    canonical_digest,
    canonical_json,
    replay_cold_payload,
    verify_archive_bytes,
    verify_archive_data,
)
from bongard.evidence import Evidence, Provenance
from bongard.ir import AllOf, Atom, Quantity, Relation, StaticLegCall
from bongard.legs import (
    PANEL,
    AffirmativeRelation,
    LegContract,
    LegRegistry,
    Unit,
    ValueType,
)


VERIFIER = "artifact-verifier"
CORPUS = hashlib.sha256(b"complete corpus manifest").hexdigest()
PROPOSER = hashlib.sha256(b"headless codex proposer source").hexdigest()
ANGLE = ValueType("measurement", Unit.DEGREES)


def origin(name: str) -> Provenance:
    return Provenance(name, "1", "frozen-atom", input_digests=("panel",))


def truth(name: str = "atom") -> Evidence[bool]:
    return Evidence.present(True, origin(name))


def false(name: str = "atom") -> Evidence[bool]:
    return Evidence.certified_absent(origin(name), "interval excludes threshold")


def build_protocol(*, query_overlap: bool = False):
    calls = {"count": 0}

    def forbidden_model_call(panel):
        calls["count"] += 1
        raise AssertionError("cold replay called a model-backed leg")

    registry = LegRegistry()
    lower_reference = registry.register(
        LegContract(
            "minimum_angle",
            "1",
            (PANEL,),
            ANGLE,
            forbidden_model_call,
            affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
        )
    )
    upper_reference = registry.register(
        LegContract(
            "maximum_angle",
            "1",
            (PANEL,),
            ANGLE,
            forbidden_model_call,
            affirmative_relations=frozenset({AffirmativeRelation.AT_MOST}),
        )
    )
    registry.freeze()
    attachment = TypedAttachmentContract.issue(
        issued_by=VERIFIER,
        registry=registry,
        boundary_types={"panel": PANEL},
    )
    lower = Atom(
        StaticLegCall(lower_reference, ("panel",)),
        Relation.AT_LEAST,
        "angle is not shallow",
        Quantity(30.0, Unit.DEGREES),
    )
    upper = Atom(
        StaticLegCall(upper_reference, ("panel",)),
        Relation.AT_MOST,
        "angle remains oblique",
        Quantity(75.0, Unit.DEGREES),
    )
    formula = AllOf((lower, upper), "the same measured angle lies in an oblique band")

    support_pos = BlobRef.from_bytes("support-pos", b"support positive", "image/png")
    support_neg = BlobRef.from_bytes("support-neg", b"support negative", "image/png")
    support = SupportCommitment(
        "run-001",
        VERIFIER,
        CORPUS,
        (
            SupportExample(support_neg, False),
            SupportExample(support_pos, True),
        ),
        "support-nonce",
    )
    freeze = ProposalFreeze.create(
        support=support,
        proposal_id="proposal-001",
        formula=formula,
        proposer_digest=PROPOSER,
        attachment_contract=attachment,
        registry=registry,
        support_gate_digest=canonical_digest({"fixture": "support-gate"}),
        verifier_nonce="freeze-nonce",
    )
    query_a_blob = (
        support_pos
        if query_overlap
        else BlobRef.from_bytes("query-a-panel", b"query a", "image/png")
    )
    query_a = QueryPanel("query-a", query_a_blob)
    query_b = QueryPanel(
        "query-b", BlobRef.from_bytes("query-b-panel", b"query b", "image/png")
    )
    release = QueryRelease.create(
        freeze, (query_a, query_b), verifier_nonce="query-nonce"
    )
    paths = atom_paths(formula)
    assert paths == ((0,), (1,))
    cold = ColdReplayInputs.capture(
        freeze=freeze,
        release=release,
        atom_evidence={
            "query-a": {paths[0]: truth("a0"), paths[1]: truth("a1")},
            "query-b": {paths[0]: truth("b0"), paths[1]: false("b1")},
        },
    )
    predictions = PredictionCommitment.create(
        freeze=freeze,
        release=release,
        cold_inputs=cold,
        verifier_nonce="prediction-nonce",
    )
    labels = LabelReveal.create(
        predictions,
        (RevealedLabel("query-a", True), RevealedLabel("query-b", False)),
        verifier_nonce="label-nonce",
    )
    bundle = RunArtifactBundle(
        support, attachment, freeze, release, cold, predictions, labels
    )
    return bundle, calls


def reseal_outer(archive: dict[str, object]) -> None:
    content = {key: value for key, value in archive.items() if key != "archive_digest"}
    archive["archive_digest"] = canonical_digest(content)


def test_canonical_json_and_digest_are_order_independent_and_finite() -> None:
    left = {"z": [2, 1], "a": {"y": True, "x": None}}
    right = {"a": {"x": None, "y": True}, "z": [2, 1]}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_digest(left) == canonical_digest(right)
    with pytest.raises(ValueError, match="canonical-JSON"):
        canonical_json({"bad": float("nan")})
    with pytest.raises(ValueError, match="keys must be strings"):
        canonical_json({1: "ambiguous"})


def test_support_blob_commitment_detects_changed_bytes() -> None:
    blob = BlobRef.from_bytes("panel", b"exact bytes", "image/png")
    blob.verify_bytes(b"exact bytes")
    with pytest.raises(ArtifactTamperError, match="bytes changed"):
        blob.verify_bytes(b"other bytes")


def test_full_chain_replays_two_predictions_without_calling_model_or_leg() -> None:
    bundle, calls = build_protocol()
    assert calls["count"] == 0
    assert [item.positive for item in bundle.predictions.predictions] == [True, False]
    receipt = bundle.verify()
    assert calls["count"] == 0
    assert receipt.predictions_match
    assert (receipt.determinate_correct, receipt.determinate_total) == (2, 2)
    assert receipt.abstentions == 0


def test_full_canonical_archive_round_trip_has_every_component() -> None:
    bundle, calls = build_protocol()
    payload = bundle.to_archive_bytes()
    decoded = json.loads(payload)
    assert set(decoded) == {
        "schema",
        "support",
        "attachment_contract",
        "proposal_freeze",
        "query_release",
        "cold_replay_inputs",
        "prediction_commitment",
        "label_reveal",
        "chain",
        "chain_digest",
        "model_free_replay_receipt",
        "archive_digest",
    }
    assert payload == canonical_json(decoded)
    verified = verify_archive_bytes(payload)
    assert isinstance(verified, VerifiedRunArchive)
    assert verified.bundle.chain_data() == bundle.chain_data()
    assert verified.replay_receipt == bundle.verify()
    assert verified.archive_digest == decoded["archive_digest"]
    assert calls["count"] == 0


def test_archive_bytes_reject_noncanonical_encoding() -> None:
    bundle, _ = build_protocol()
    pretty = json.dumps(bundle.to_archive_data(), indent=2)
    with pytest.raises(ArtifactTamperError, match="not canonical JSON"):
        verify_archive_bytes(pretty)


def test_archive_formula_tamper_survives_outer_reseal_but_not_formula_binding() -> None:
    bundle, _ = build_protocol()
    archive = json.loads(bundle.to_archive_bytes())
    archive["proposal_freeze"]["formula"]["terms"][0]["lower"]["value"] = 31.0
    reseal_outer(archive)
    with pytest.raises(ArtifactTamperError, match="formula content differs"):
        verify_archive_data(archive)


def test_archive_offline_validation_rejects_flipped_scalar_orientation() -> None:
    bundle, _ = build_protocol()
    archive = json.loads(bundle.to_archive_bytes())
    formula = archive["proposal_freeze"]["formula"]
    formula["terms"][0]["relation"] = "at_most"
    archive["proposal_freeze"]["formula_digest"] = canonical_digest(formula)
    reseal_outer(archive)
    with pytest.raises(
        ArtifactTamperError, match="not an affirmative orientation"
    ):
        verify_archive_data(archive)


def test_archive_offline_validation_rejects_threshold_unit_tamper() -> None:
    bundle, _ = build_protocol()
    archive = json.loads(bundle.to_archive_bytes())
    formula = archive["proposal_freeze"]["formula"]
    formula["terms"][0]["lower"]["unit"] = "probability"
    archive["proposal_freeze"]["formula_digest"] = canonical_digest(formula)
    reseal_outer(archive)
    with pytest.raises(ArtifactTamperError, match="threshold uses probability"):
        verify_archive_data(archive)


def test_archive_offline_validation_rejects_boundary_type_tamper() -> None:
    bundle, _ = build_protocol()
    archive = json.loads(bundle.to_archive_bytes())
    attachment = archive["attachment_contract"]
    attachment["boundary_types"][0][1] = {"name": "object", "unit": "none"}
    archive["proposal_freeze"]["attachment_contract_digest"] = canonical_digest(
        attachment
    )
    reseal_outer(archive)
    with pytest.raises(ArtifactTamperError, match="expected ValueType.*panel"):
        verify_archive_data(archive)


def test_archive_recomputes_contract_and_reference_digests() -> None:
    bundle, _ = build_protocol()
    archive = json.loads(bundle.to_archive_bytes())
    attachment = archive["attachment_contract"]
    attachment["registry_snapshot"][0]["source_digest"] = "0" * 64
    attachment["registry_digest"] = canonical_digest(
        attachment["registry_snapshot"]
    )
    archive["proposal_freeze"]["registry_digest"] = attachment[
        "registry_digest"
    ]
    archive["proposal_freeze"]["attachment_contract_digest"] = canonical_digest(
        attachment
    )
    reseal_outer(archive)
    with pytest.raises(ArtifactTamperError, match="contract digest mismatch"):
        verify_archive_data(archive)


def test_archive_cold_evidence_tamper_breaks_prediction_parent() -> None:
    bundle, _ = build_protocol()
    archive = json.loads(bundle.to_archive_bytes())
    evidence = archive["cold_replay_inputs"]["queries"][0]["atom_inputs"][0]["evidence"]
    evidence["provenance"]["producer"] = "tampered-observer"
    archive["chain"]["cold_inputs"] = canonical_digest(
        archive["cold_replay_inputs"]
    )
    archive["chain_digest"] = canonical_digest(archive["chain"])
    reseal_outer(archive)
    with pytest.raises(ArtifactTamperError, match="changed after prediction"):
        verify_archive_data(archive)


def test_archive_proposer_digest_tamper_breaks_query_parent() -> None:
    bundle, _ = build_protocol()
    archive = json.loads(bundle.to_archive_bytes())
    archive["proposal_freeze"]["proposer_digest"] = hashlib.sha256(
        b"different proposer"
    ).hexdigest()
    archive["chain"]["freeze"] = canonical_digest(archive["proposal_freeze"])
    archive["chain_digest"] = canonical_digest(archive["chain"])
    reseal_outer(archive)
    with pytest.raises(ArtifactTamperError, match="query was not released"):
        verify_archive_data(archive)


def test_archive_label_tamper_changes_replayed_receipt() -> None:
    bundle, _ = build_protocol()
    archive = json.loads(bundle.to_archive_bytes())
    archive["label_reveal"]["labels"][0]["positive"] = False
    archive["chain"]["labels"] = canonical_digest(archive["label_reveal"])
    archive["chain_digest"] = canonical_digest(archive["chain"])
    reseal_outer(archive)
    with pytest.raises(ArtifactTamperError, match="receipt does not reproduce"):
        verify_archive_data(archive)


def test_archive_label_order_is_canonical() -> None:
    bundle, _ = build_protocol()
    archive = json.loads(bundle.to_archive_bytes())
    archive["label_reveal"]["labels"].reverse()
    reseal_outer(archive)
    with pytest.raises(ArtifactTamperError, match="label ids must be unique and sorted"):
        verify_archive_data(archive)


def test_plain_json_round_trip_is_genuinely_cold_and_model_free() -> None:
    bundle, calls = build_protocol()
    formula_data = json.loads(canonical_json(bundle.freeze.formula.to_data()))
    cold_data = json.loads(canonical_json(bundle.cold_inputs.to_data()))
    replayed = replay_cold_payload(formula_data, cold_data)
    assert calls["count"] == 0
    assert [record.disposition for _, record in replayed] == [
        bundle.predictions.predictions[0].disposition,
        bundle.predictions.predictions[1].disposition,
    ]
    assert [record.digest() for _, record in replayed] == [
        prediction.evidence_digest
        for prediction in bundle.predictions.predictions
    ]


def test_query_release_requires_exactly_two_unlabeled_queries() -> None:
    bundle, _ = build_protocol()
    query = bundle.release.queries[0]
    with pytest.raises(ValueError, match="exactly two"):
        QueryRelease(
            bundle.support.run_id,
            bundle.freeze.digest(),
            (query,),  # type: ignore[arg-type]
            "nonce",
        )
    assert "label" not in bundle.release.to_data()
    assert all("positive" not in query.to_data() for query in bundle.release.queries)


def test_freeze_is_cryptographic_parent_of_query_and_prediction() -> None:
    bundle, _ = build_protocol()
    assert bundle.release.proposal_freeze_digest == bundle.freeze.digest()
    assert bundle.predictions.proposal_freeze_digest == bundle.freeze.digest()
    changed_freeze = replace(bundle.freeze, verifier_nonce="post-hoc-change")
    tampered = replace(bundle, freeze=changed_freeze)
    with pytest.raises(ArtifactTamperError, match="query was not released"):
        tampered.verify()


def test_prediction_commitment_is_parent_of_label_reveal() -> None:
    bundle, _ = build_protocol()
    assert (
        bundle.labels.prediction_commitment_digest == bundle.predictions.digest()
    )
    changed = replace(bundle.predictions, verifier_nonce="changed-prediction")
    with pytest.raises(ArtifactTamperError, match="labels were not revealed"):
        replace(bundle, predictions=changed).verify()


def test_tampered_cold_observation_breaks_prediction_commitment() -> None:
    bundle, _ = build_protocol()
    first_query = bundle.cold_inputs.queries[0]
    changed_atom = AtomReplayInput(
        first_query.atom_inputs[0].path,
        TruthEvidenceRecord.from_evidence(false("tampered")),
    )
    changed_query = replace(
        first_query,
        atom_inputs=(changed_atom, *first_query.atom_inputs[1:]),
    )
    changed_cold = replace(
        bundle.cold_inputs,
        queries=(changed_query, bundle.cold_inputs.queries[1]),
    )
    with pytest.raises(ArtifactTamperError, match="changed after prediction"):
        replace(bundle, cold_inputs=changed_cold).verify()


def test_cold_capture_requires_every_atom_for_both_queries() -> None:
    bundle, _ = build_protocol()
    path = atom_paths(bundle.freeze.formula)[0]
    with pytest.raises(ValueError, match="paths differ"):
        ColdReplayInputs.capture(
            freeze=bundle.freeze,
            release=bundle.release,
            atom_evidence={
                "query-a": {path: truth()},
                "query-b": {path: truth()},
            },
        )


def test_support_query_byte_overlap_is_detected() -> None:
    bundle, _ = build_protocol(query_overlap=True)
    with pytest.raises(ArtifactTamperError, match="overlap committed support"):
        bundle.verify()


def test_artifact_contract_can_coexist_with_empty_accepted_archive() -> None:
    bundle, _ = build_protocol()
    archive = ArchivePreservationContract(
        VERIFIER, hashlib.sha256(b"replay-suite").hexdigest()
    )
    assert archive.entries == ()
    assert bundle.attachment_contract.issued_by == archive.issued_by
