from __future__ import annotations

import base64
import copy
import hashlib
import json
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard.artifacts import BlobRef, SupportCommitment, SupportExample
from bongard.evidence import Disposition, Evidence, Provenance
from bongard.legs.neutral_features import (
    FEATURE_GROUP_IDS,
    extract_neutral_features,
    feature_group_catalog_digest,
    feature_space_for_group,
    project_neutral_feature_extraction,
)
from bongard.prototype_artifacts import (
    FeatureExtractionPreimage,
    PrototypeArtifactError,
    PrototypeArtifactTamperError,
    PrototypeFreezePolicy,
    PrototypePreQueryFreeze,
    PrototypeQueryArtifact,
    PrototypeSupportReplayArtifact,
    PrototypeTruthEvidence,
)
from bongard.support_prototypes import (
    PositivePrototypeFormula,
    SupportPrototypePlan,
    fit_support_prototypes,
    panel_side_assignment_digest,
)


GROUP_ID = "prototype.global_geometry"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def panel_png(index: int, *, positive: bool) -> bytes:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    offset = index % 3
    if positive:
        draw.rectangle((25 + offset, 10, 34 + offset, 52), fill="black")
    else:
        draw.rectangle((10, 25 + offset, 52, 34 + offset), fill="black")
    # A tiny interior mark gives every panel a distinct byte/vector preimage
    # without creating border clipping or an under-sized foreground.
    draw.rectangle((12 + index, 45, 13 + index, 46), fill=(index, index, index))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def extraction_preimage(payload: bytes) -> FeatureExtractionPreimage:
    full = extract_neutral_features(payload)
    projected = project_neutral_feature_extraction(full, GROUP_ID)
    assert projected.evidence.disposition is Disposition.PRESENT
    return FeatureExtractionPreimage.from_extraction(payload, projected)


def fixture() -> tuple[
    PrototypePreQueryFreeze,
    SupportCommitment,
    tuple[bytes, ...],
    tuple[bytes, ...],
]:
    positive_bytes = tuple(panel_png(index, positive=True) for index in range(6))
    negative_bytes = tuple(panel_png(index + 6, positive=False) for index in range(6))
    positive = tuple(extraction_preimage(item) for item in positive_bytes)
    negative = tuple(extraction_preimage(item) for item in negative_bytes)
    space = feature_space_for_group(GROUP_ID)
    all_spaces = {
        group_id: (feature_space_for_group(group_id), 0.01)
        for group_id in FEATURE_GROUP_IDS
    }
    policy = PrototypeFreezePolicy.create(
        feature_catalog_digest=feature_group_catalog_digest(),
        allowed_groups=all_spaces,
    )
    assignment = panel_side_assignment_digest(
        tuple(item.panel_digest for item in positive),
        tuple(item.panel_digest for item in negative),
    )
    plan = SupportPrototypePlan(space.digest(), assignment, 6)
    prototypes = fit_support_prototypes(
        plan,
        space,
        tuple(item.require_present() for item in positive),
        tuple(item.require_present() for item in negative),
        expected_plan_digest=plan.digest(),
    )
    formula = PositivePrototypeFormula(
        claim="the panel matches the positive support geometry",
        feature_space_digest=space.digest(),
        prototype_digest=prototypes.digest(),
        support_assignment_digest=assignment,
        decision_margin=0.01,
    )
    examples = tuple(
        sorted(
            (
                *(
                    SupportExample(
                        BlobRef.from_bytes(
                            f"support-positive-{index:02d}", payload, "image/png"
                        ),
                        True,
                    )
                    for index, payload in enumerate(positive_bytes)
                ),
                *(
                    SupportExample(
                        BlobRef.from_bytes(
                            f"support-negative-{index:02d}", payload, "image/png"
                        ),
                        False,
                    )
                    for index, payload in enumerate(negative_bytes)
                ),
            ),
            key=lambda item: item.panel.blob_id,
        )
    )
    support = SupportCommitment(
        run_id="prototype-artifact-test",
        issued_by="prototype-test-verifier",
        corpus_digest=digest("corpus"),
        support=examples,
        verifier_nonce="fixed-test-nonce",
    )
    freeze = PrototypePreQueryFreeze.create(
        support_commitment=support,
        policy=policy,
        selected_feature_group_id=GROUP_ID,
        feature_space=space,
        positive_support=positive,
        negative_support=negative,
        fit_plan=plan,
        prototypes=prototypes,
        positive_formula=formula,
        semantic_proposal_digest=digest("semantic proposal"),
    )
    return freeze, support, positive_bytes, negative_bytes


def json_copy(value: object) -> object:
    return json.loads(json.dumps(value))


def test_policy_precommits_catalog_space_and_margin_before_selection() -> None:
    spaces = {
        group_id: (feature_space_for_group(group_id), 0.01 + index / 100.0)
        for index, group_id in enumerate(FEATURE_GROUP_IDS)
    }
    policy = PrototypeFreezePolicy.create(
        feature_catalog_digest=feature_group_catalog_digest(),
        allowed_groups=spaces,
    )
    assert PrototypeFreezePolicy.from_data(json_copy(policy.to_data())) == policy
    selected = policy.select(GROUP_ID, spaces[GROUP_ID][0])
    assert selected.decision_margin == spaces[GROUP_ID][1]
    with pytest.raises(PrototypeArtifactTamperError, match="not precommitted"):
        policy.select("prototype.unregistered", spaces[GROUP_ID][0])


def test_prequery_freeze_round_trip_commits_all_twelve_exact_preimages() -> None:
    freeze, support, _, _ = fixture()
    data = json_copy(freeze.to_data())
    restored = PrototypePreQueryFreeze.from_data(data)
    restored.verify(support)
    assert restored == freeze
    assert len(restored.positive_support) == len(restored.negative_support) == 6
    assert all(item.feature_packet is not None for item in restored.positive_support)
    assert all(item.receipt.to_data() for item in restored.negative_support)
    assert restored.compiler_inputs().decision_margin == 0.01
    assert not ({"queries", "query", "query_id"} & set(data))
    assert "lean" not in json.dumps(data).lower()
    assert PrototypePreQueryFreeze.from_committed_data(
        json_copy(freeze.committed_data()), support_commitment=support
    ) == freeze


def test_prequery_rejects_query_injection_missing_receipt_and_digest_drift() -> None:
    freeze, _, _, _ = fixture()
    injected = json_copy(freeze.to_data())
    injected["queries"] = []
    with pytest.raises(ValueError, match="missing or unknown"):
        PrototypePreQueryFreeze.from_data(injected)

    missing = json_copy(freeze.to_data())
    del missing["positive_support"][0]["extractor_receipt"]
    with pytest.raises(ValueError, match="missing or unknown"):
        PrototypePreQueryFreeze.from_data(missing)

    panel_drift = json_copy(freeze.to_data())
    raw = base64.b64decode(panel_drift["positive_support"][0]["panel_base64"])
    changed = bytes((raw[0] ^ 1,)) + raw[1:]
    panel_drift["positive_support"][0]["panel_base64"] = base64.b64encode(
        changed
    ).decode("ascii")
    with pytest.raises(PrototypeArtifactTamperError, match="panel preimage digest"):
        PrototypePreQueryFreeze.from_data(panel_drift)

    drifted = json_copy(freeze.committed_data())
    drifted["pre_query_freeze_digest"] = digest("changed")
    with pytest.raises(PrototypeArtifactTamperError, match="digest drift"):
        PrototypePreQueryFreeze.from_committed_data(drifted)


def test_side_swap_and_support_commitment_swap_are_rejected() -> None:
    freeze, support, _, _ = fixture()
    swapped = json_copy(freeze.to_data())
    swapped["positive_support"], swapped["negative_support"] = (
        swapped["negative_support"],
        swapped["positive_support"],
    )
    with pytest.raises(PrototypeArtifactTamperError, match="assignment"):
        PrototypePreQueryFreeze.from_data(swapped)

    flipped = SupportCommitment(
        run_id=support.run_id,
        issued_by=support.issued_by,
        corpus_digest=support.corpus_digest,
        support=tuple(
            SupportExample(item.panel, not item.positive) for item in support.support
        ),
        verifier_nonce=support.verifier_nonce,
    )
    with pytest.raises(PrototypeArtifactTamperError, match="commitment digest"):
        freeze.verify(flipped)


def test_query_capture_is_held_out_and_cold_replayable() -> None:
    freeze, _, _, _ = fixture()
    query = extraction_preimage(panel_png(20, positive=True))
    artifact = PrototypeQueryArtifact.capture(
        query_id="query-0", freeze=freeze, extraction=query
    )
    restored = PrototypeQueryArtifact.from_data(
        json_copy(artifact.to_data()), freeze=freeze
    )
    assert restored == artifact
    assert artifact.margin is not None
    assert artifact.evidence.disposition in set(Disposition)


def test_query_support_overlap_and_evidence_coercion_are_rejected() -> None:
    freeze, _, _, _ = fixture()
    with pytest.raises(PrototypeArtifactTamperError, match="overlaps"):
        PrototypeQueryArtifact.capture(
            query_id="query-overlap",
            freeze=freeze,
            extraction=freeze.positive_support[0],
        )
    query = extraction_preimage(panel_png(21, positive=False))
    artifact = PrototypeQueryArtifact.capture(
        query_id="query-1", freeze=freeze, extraction=query
    )
    coerced = json_copy(artifact.to_data())
    coerced["evidence"] = True
    with pytest.raises(TypeError, match="JSON object"):
        PrototypeQueryArtifact.from_data(coerced, freeze=freeze)
    with pytest.raises(TypeError, match="four dispositions"):
        bool(artifact.evidence)


def test_query_margin_and_packet_digest_tampering_are_rejected() -> None:
    freeze, _, _, _ = fixture()
    artifact = PrototypeQueryArtifact.capture(
        query_id="query-2",
        freeze=freeze,
        extraction=extraction_preimage(panel_png(22, positive=True)),
    )
    changed_margin = json_copy(artifact.to_data())
    changed_margin["margin"]["upper"] += 0.01
    with pytest.raises(PrototypeArtifactTamperError, match="margin digest drift"):
        PrototypeQueryArtifact.from_data(changed_margin, freeze=freeze)
    changed_packet = json_copy(artifact.to_data())
    changed_packet["extraction"]["feature_packet"]["values"][0]["lower"] -= 0.001
    changed_packet["extraction"]["feature_packet"]["values"][0]["upper"] -= 0.001
    with pytest.raises(PrototypeArtifactTamperError, match="packet digest drift"):
        PrototypeQueryArtifact.from_data(changed_packet, freeze=freeze)


def test_support_replay_is_fresh_exact_and_contains_no_side_label() -> None:
    freeze, _, positive_bytes, _ = fixture()
    fresh = extraction_preimage(positive_bytes[0])
    artifact = PrototypeSupportReplayArtifact.capture(
        freeze=freeze, extraction=fresh
    )
    restored = PrototypeSupportReplayArtifact.from_data(
        json_copy(artifact.to_data()), freeze=freeze
    )
    assert restored == artifact
    assert not ({"positive", "side", "label"} & set(artifact.to_data()))
    assert artifact.margin is not None

    held_out = extraction_preimage(panel_png(23, positive=True))
    with pytest.raises(PrototypeArtifactTamperError, match="not in the frozen support"):
        PrototypeSupportReplayArtifact.capture(freeze=freeze, extraction=held_out)


def test_truth_evidence_round_trips_every_disposition_without_boolean_collapse() -> None:
    provenance = Provenance("test", "1", "four-disposition")
    cases = (
        Evidence.present(True, provenance),
        Evidence.certified_absent(provenance, "certificate"),
        Evidence.indeterminate(provenance, "not enough information"),
        Evidence.error(provenance, "ExtractorError", "failed"),
    )
    assert {item.disposition for item in cases} == set(Disposition)
    for evidence in cases:
        record = PrototypeTruthEvidence.from_evidence(evidence)
        assert PrototypeTruthEvidence.from_data(json_copy(record.to_data())) == record
        with pytest.raises(TypeError, match="four dispositions"):
            bool(record)
    coerced = PrototypeTruthEvidence.from_evidence(cases[0]).to_data()
    coerced["value"] = 1
    with pytest.raises(ValueError, match="exactly true"):
        PrototypeTruthEvidence.from_data(coerced)


def test_failed_support_extraction_cannot_be_coerced_to_a_negative() -> None:
    freeze, support, _, _ = fixture()
    failure = project_neutral_feature_extraction(
        extract_neutral_features(b"not-a-png"), GROUP_ID
    )
    failed_preimage = FeatureExtractionPreimage.from_extraction(b"not-a-png", failure)
    assert failed_preimage.feature_packet is None
    with pytest.raises(PrototypeArtifactError, match="failure is not a negative"):
        failed_preimage.require_present()
    changed = copy.copy(freeze)
    object.__setattr__(
        changed,
        "positive_support",
        tuple(sorted((failed_preimage, *freeze.positive_support[1:]), key=lambda item: item.panel_digest)),
    )
    with pytest.raises(PrototypeArtifactError, match="failure is not a negative"):
        changed.verify(support)
