from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from PIL import Image
import pytest

from bongard.admission import TypedAttachmentContract
from bongard.artifacts import (
    ColdReplayInputs,
    LabelReveal,
    PredictionCommitment,
    ProposalFreeze,
    QueryRelease,
    RunArtifactBundle,
    canonical_digest,
    canonical_json,
    verify_archive_data,
)
from bongard.benchmark import SupportGatePolicy, prepare_episode, run_episode
from bongard.corpus import ShapeBongardCorpus
from bongard.exposure import ExposureLedger
from bongard.ir import Atom
from bongard.legs import PANEL, LegRegistry
from bongard.proposer import (
    HYBRID_EPISTEMIC_STATUS,
    HeadlessCodexEpisode,
    ProposalError,
    RejectedProposalError,
    _parse_hybrid_observation,
    parse_hybrid_observation_or_error,
    parse_rule_proposal,
)
from bongard.release import OfficialReleaseDescriptor
from bongard.run_verification import (
    EXPOSURE_SCHEMA,
    OUTER_RUN_SCHEMA,
    MissingBlobPreimagesError,
    MissingSupportPreimagesError,
    RunVerificationError,
    audit_completed_run_commitments,
    audit_rejected_run_commitments,
    verify_completed_run_bytes,
    verify_completed_run_data,
    verify_rejected_run_bytes,
    verify_rejected_run_data,
)
from bongard.synthesis import compile_hybrid_proposal
import bongard.transport as T


MODEL = "gpt-5.6-sol"
EFFORT = "medium"


def _write_png(path: Path, marker: int) -> bytes:
    gray = np.full((24, 25), 255, dtype=np.uint8)
    gray[2 + marker % 18, 2:22] = 0
    gray[3 + marker % 18, 2:22] = 127
    rgb = np.repeat(gray[..., None], 3, axis=2)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")
    return path.read_bytes()


def _proposal_payload() -> dict[str, Any]:
    return {
        "positive_description": "one closed bird-like angular form",
        "panel_descriptions": {
            **{f"pos_{index}": f"bird-like angular form {index}" for index in range(6)},
            **{f"neg_{index}": f"angular comparison form {index}" for index in range(6)},
        },
        "view": "carrier_shape",
        "observable_requests": [],
        "formula_template": {"kind": "all", "atoms": ["hybrid_claim"]},
        "hybrid_claim": {
            "epistemic_status": HYBRID_EPISTEMIC_STATUS,
            "phrase": "bird-like angular form",
            "operational_definition": (
                "one central body with two lateral wing-like lobes and oblique junctions"
            ),
            "required_visual_cues": [
                {
                    "cue_id": "wing_like_lobes",
                    "positive_description": "two lateral wing-like lobes",
                },
                {
                    "cue_id": "oblique_junctions",
                    "positive_description": "boundary segments meeting at oblique angles",
                },
            ],
        },
        "confidence": "medium",
    }


def _present_payload() -> dict[str, Any]:
    return {
        "epistemic_status": HYBRID_EPISTEMIC_STATUS,
        "disposition": "present",
        "observed_cue_ids": ["wing_like_lobes", "oblique_junctions"],
        "missing_cue_ids": [],
        "missing_cue_reasons": [],
        "visibility_certificate": None,
        "reason": None,
        "error_type": None,
    }


def _nonmatch_payload() -> dict[str, Any]:
    return {
        "epistemic_status": HYBRID_EPISTEMIC_STATUS,
        "disposition": "nonmatch",
        "observed_cue_ids": [],
        "missing_cue_ids": ["wing_like_lobes"],
        "missing_cue_reasons": [
            {
                "cue_id": "wing_like_lobes",
                "finding": "the visible form has a single narrow lateral contour",
            }
        ],
        "visibility_certificate": "the complete form is visible",
        "reason": "the complete visible contour has a different lateral structure",
        "error_type": None,
    }


def _identities(paths: tuple[str, ...], names: tuple[str, ...]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "byte_count": len(Path(path).read_bytes()),
            "content_digest": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        for path, name in zip(paths, names, strict=True)
    ]


class _ReceiptFactory:
    def __init__(self) -> None:
        self.index = 0

    def make(
        self,
        prompt: str,
        paths: tuple[str, ...],
        schema: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        names: tuple[str, ...] | None = None,
    ) -> T.CodexReceipt:
        self.index += 1
        if names is None:
            names = tuple(Path(path).name for path in paths)
            input_schema = T.STRUCTURED_INPUT_DIGEST_SCHEMA
            view_digest = T.ordered_panel_view_digest(paths)
            set_digest = T.semantic_panel_set_digest(paths)
            identity_key = "ordered_panel_identities"
            view_key = "panel_view_digest"
            set_key = "panel_set_digest"
        else:
            input_schema = T.NAMED_IMAGE_INPUT_DIGEST_SCHEMA
            view_digest = T.named_image_view_digest(paths, names)
            set_digest = T.named_image_set_digest(paths, names)
            identity_key = "ordered_image_identities"
            view_key = "image_view_digest"
            set_key = "image_set_digest"
        identities = _identities(paths, names)
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        schema_digest = hashlib.sha256(canonical_json(dict(schema))).hexdigest()
        envelope = {
            "schema": input_schema,
            "task": prompt,
            identity_key: identities,
            view_key: view_digest,
            set_key: set_digest,
            "prompt_digest": prompt_digest,
            "output_schema_digest": schema_digest,
        }
        body: dict[str, Any] = {
            "schema": T.CODEX_RECEIPT_SCHEMA,
            "source": "codex-cli",
            "requested_model": MODEL,
            "reported_model": "",
            "model_identity_evidence": "explicit-cli-model-flag;jsonl-omits-model",
            "requested_reasoning_effort": EFFORT,
            "input_tokens": 10,
            "cached_input_tokens": 0,
            "output_tokens": 5,
            "reasoning_output_tokens": 1,
            "thread_id": f"00000000-0000-4000-8000-{self.index:012d}",
            "codex_cli_version": "codex-cli test",
            "codex_launcher_digest": hashlib.sha256(b"launcher").hexdigest(),
            "cloud_config_bundle_cache_binding": "absent",
            "task_digest": prompt_digest,
            "current_source_digest": "",
            "current_log_digest": "",
            "prompt_digest": prompt_digest,
            "input_digest_schema": input_schema,
            "input_digest": canonical_digest(envelope),
            "output_schema_digest": schema_digest,
            "panel_view_digest": view_digest,
            "panel_set_digest": set_digest,
            "structured_output_digest": canonical_digest(dict(payload)),
            "proposed_source_digest": "",
            "proposed_log_digest": "",
            "event_stream_digest": hashlib.sha256(
                f"stream-{self.index}".encode("ascii")
            ).hexdigest(),
            "event_types": [
                "thread.started",
                "turn.started",
                "item.completed",
                "turn.completed",
            ],
            "item_types": ["agent_message"],
            "isolation_policy": T.CODEX_ISOLATION_POLICY,
            "outcome": "success",
        }
        body["receipt_digest"] = canonical_digest(body)
        T.validate_codex_receipt(body)
        return T.CodexReceipt(
            **{
                **body,
                "event_types": tuple(body["event_types"]),
                "item_types": tuple(body["item_types"]),
            }
        )


@pytest.fixture
def completed_run(tmp_path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    corpus_root = tmp_path / "completed" / "ShapeBongard_V2"
    task_id = "bd_trapez_parallelogram_0000"
    for side, label in (("positive", "1"), ("negative", "0")):
        panel_root = corpus_root / "bd" / "images" / task_id / label
        panel_root.mkdir(parents=True)
        for index in range(7):
            _write_png(panel_root / f"{index}.png", index + (0 if side == "positive" else 7))
    split_path = corpus_root / "ShapeBongard_V2_split.json"
    split_path.write_text(
        json.dumps({"train": [task_id]}, sort_keys=True), encoding="utf-8"
    )
    corpus = ShapeBongardCorpus.discover(corpus_root)
    manifest = corpus.build_manifest()
    plan = prepare_episode(corpus, task_id, seed="run-verification", corpus_manifest=manifest)
    receipts = _ReceiptFactory()

    def proposer_transport(prompt, paths, schema, **kwargs):
        del kwargs
        payload = _proposal_payload()
        canonical_paths = tuple(paths)
        return SimpleNamespace(
            payload=payload,
            receipt=receipts.make(prompt, canonical_paths, schema, payload),
        )

    def observer_transport(prompt, paths, names, schema, **kwargs):
        del kwargs
        canonical_paths = tuple(paths)
        canonical_names = tuple(names)
        pixels = np.asarray(Image.open(canonical_paths[0]).convert("L"))
        marker = int(np.argmin(pixels.min(axis=1))) - 2
        payload = _present_payload() if marker < 7 else _nonmatch_payload()
        return SimpleNamespace(
            payload=payload,
            receipt=receipts.make(
                prompt,
                canonical_paths,
                schema,
                payload,
                names=canonical_names,
            ),
        )

    session = HeadlessCodexEpisode(
        model=MODEL,
        reasoning_effort=EFFORT,
        proposer_transport=proposer_transport,
        observer_transport=observer_transport,
    )
    result = run_episode(
        plan,
        session,
        session,
        support_gate_policy=SupportGatePolicy.empirical(),
    )
    assert result.bundle is not None
    vision = session.artifact_data()
    vision["support_gate"] = result.support_gate.to_data()
    vision["proposal_freeze"] = result.proposal_freeze.to_data()
    assert sorted(vision["observations"]) == ["query-0", "query-1"]
    archive = result.bundle.to_archive_data()
    exposure_ledger = ExposureLedger.create(manifest.digest)
    exposure_successor = exposure_ledger.record(
        phase="support_release_precommit",
        actor=MODEL,
        purpose=(
            "support_release_precommit "
            f"task={task_id} model={MODEL} plan_digest={plan.digest}"
        ),
        task_ids=(task_id,),
        source="bongard.cli.run",
        observed_at="2026-08-06T00:00:00Z",
    )
    exposure_event = exposure_successor.events[-1]
    exposure = {
        "schema": EXPOSURE_SCHEMA,
        "corpus_manifest_digest": manifest.digest,
        "task_id": task_id,
        "model": MODEL,
        "plan_digest": plan.digest,
        "ledger_before_digest": exposure_ledger.digest,
        "ledger_after_digest": exposure_successor.digest,
        "event_digest": exposure_event.digest,
        "event": exposure_event.to_dict(),
        "ledger_before_event_count": len(exposure_ledger.events),
        "ledger_after_event_count": len(exposure_successor.events),
        "ledger_input_supplied": False,
        "unseen_required": False,
        "semantic_unseen_required": False,
        "historical_seed_digest": None,
        "semantic_resolver_policy_digest": None,
        "expected_semantic_cohort": None,
        "classified_semantic_cohort": None,
        "semantic_unseen_receipt": None,
        "successor_filename": (
            exposure_successor.digest.removeprefix("sha256:") + ".exposure.json"
        ),
        "external_anchor": None,
    }
    content: dict[str, Any] = {
        "schema": OUTER_RUN_SCHEMA,
        "corpus_manifest_digest": manifest.digest,
        "split_source_digest": corpus.split.source_digest,
        "official_release": None,
        "plan": plan.to_data(),
        "episode": result.to_data(),
        "vision": vision,
        "run_archive": archive,
        "exposure": exposure,
    }
    record = {**content, "record_digest": canonical_digest(content)}
    blob_bytes = {
        source.panel.blob_id: source.path.read_bytes()
        for source in (*plan._support_sources, *plan._query_sources)
    }
    return record, blob_bytes


@pytest.fixture
def rejected_run(tmp_path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    corpus_root = tmp_path / "rejected" / "ShapeBongard_V2"
    task_id = "bd_rejected_fixture_0000"
    for side, label in (("positive", "1"), ("negative", "0")):
        panel_root = corpus_root / "bd" / "images" / task_id / label
        panel_root.mkdir(parents=True)
        for index in range(7):
            _write_png(
                panel_root / f"{index}.png",
                index + (0 if side == "positive" else 7),
            )
    split_path = corpus_root / "ShapeBongard_V2_split.json"
    split_path.write_text(
        json.dumps({"train": [task_id]}, sort_keys=True), encoding="utf-8"
    )
    corpus = ShapeBongardCorpus.discover(corpus_root)
    manifest = corpus.build_manifest()
    plan = prepare_episode(
        corpus,
        task_id,
        seed="rejected-run-verification",
        corpus_manifest=manifest,
    )
    receipts = _ReceiptFactory()

    def proposer_transport(prompt, paths, schema, **kwargs):
        del kwargs
        payload = _proposal_payload()
        payload["hybrid_claim"]["phrase"] = "a form that is not symmetric"
        canonical_paths = tuple(paths)
        return SimpleNamespace(
            payload=payload,
            receipt=receipts.make(prompt, canonical_paths, schema, payload),
        )

    session = HeadlessCodexEpisode(
        model=MODEL,
        reasoning_effort=EFFORT,
        proposer_transport=proposer_transport,
    )
    result = run_episode(
        plan,
        session,
        session,
        support_gate_policy=SupportGatePolicy.empirical(),
    )
    assert result.bundle is None
    assert result.failure is not None
    assert result.failure.error_type == RejectedProposalError.__name__
    vision = session.artifact_data()
    vision["support_gate"] = None
    vision["proposal_freeze"] = None
    assert vision["proposal"] is None
    assert vision["rejected_proposal_attempt"] is not None
    with pytest.raises(ProposalError, match="exactly one proposer call"):
        session.propose(
            SimpleNamespace(
                positive_paths=tuple(
                    source.path for source in plan._support_sources if source.positive
                ),
                negative_paths=tuple(
                    source.path for source in plan._support_sources if not source.positive
                ),
            )
        )

    exposure_ledger = ExposureLedger.create(manifest.digest)
    exposure_successor = exposure_ledger.record(
        phase="support_release_precommit",
        actor=MODEL,
        purpose=(
            "support_release_precommit "
            f"task={task_id} model={MODEL} plan_digest={plan.digest}"
        ),
        task_ids=(task_id,),
        source="bongard.cli.run",
        observed_at="2026-08-06T00:00:00Z",
    )
    exposure_event = exposure_successor.events[-1]
    exposure = {
        "schema": EXPOSURE_SCHEMA,
        "corpus_manifest_digest": manifest.digest,
        "task_id": task_id,
        "model": MODEL,
        "plan_digest": plan.digest,
        "ledger_before_digest": exposure_ledger.digest,
        "ledger_after_digest": exposure_successor.digest,
        "event_digest": exposure_event.digest,
        "event": exposure_event.to_dict(),
        "ledger_before_event_count": len(exposure_ledger.events),
        "ledger_after_event_count": len(exposure_successor.events),
        "ledger_input_supplied": False,
        "unseen_required": False,
        "semantic_unseen_required": False,
        "historical_seed_digest": None,
        "semantic_resolver_policy_digest": None,
        "expected_semantic_cohort": None,
        "classified_semantic_cohort": None,
        "semantic_unseen_receipt": None,
        "successor_filename": (
            exposure_successor.digest.removeprefix("sha256:") + ".exposure.json"
        ),
        "external_anchor": None,
    }
    content: dict[str, Any] = {
        "schema": OUTER_RUN_SCHEMA,
        "corpus_manifest_digest": manifest.digest,
        "split_source_digest": corpus.split.source_digest,
        "official_release": None,
        "plan": plan.to_data(),
        "episode": result.to_data(),
        "vision": vision,
        "run_archive": None,
        "exposure": exposure,
    }
    record = {**content, "record_digest": canonical_digest(content)}
    positive_sources = tuple(
        source for source in plan._support_sources if source.positive
    )
    negative_sources = tuple(
        source for source in plan._support_sources if not source.positive
    )
    support_bytes = {
        **{
            f"pos_{index}.png": source.path.read_bytes()
            for index, source in enumerate(positive_sources)
        },
        **{
            f"neg_{index}.png": source.path.read_bytes()
            for index, source in enumerate(negative_sources)
        },
    }
    return record, support_bytes


def _reseal_outer(record: dict[str, Any]) -> None:
    content = {key: value for key, value in record.items() if key != "record_digest"}
    record["record_digest"] = canonical_digest(content)


def _reseal_rejected_attempt(attempt: dict[str, Any]) -> None:
    content = {key: value for key, value in attempt.items() if key != "attempt_digest"}
    attempt["attempt_digest"] = "sha256:" + canonical_digest(content)


def _reseal_receipt(receipt: dict[str, Any]) -> None:
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    receipt["receipt_digest"] = canonical_digest(body)
    T.validate_codex_receipt(receipt)


def _receipt_object(receipt: Mapping[str, Any]) -> T.CodexReceipt:
    return T.CodexReceipt(
        **{
            **dict(receipt),
            "event_types": tuple(receipt["event_types"]),
            "item_types": tuple(receipt["item_types"]),
        }
    )


def _alternate_hybrid_leg(panel: object):
    """A deliberately noncanonical implementation for registry-tamper tests."""

    raise RuntimeError(f"unexpected invocation for {type(panel).__name__}")


def _reseal_with_noncanonical_compilation(
    record: Mapping[str, Any], target: str
) -> dict[str, Any]:
    """Build an internally valid chain around a noncanonical HYBRID compile."""

    changed = copy.deepcopy(record)
    archive = verify_archive_data(changed["run_archive"])
    bundle = archive.bundle
    proposal_data = changed["vision"]["proposal"]
    proposal = parse_rule_proposal(
        proposal_data["model_payload"],
        receipt=_receipt_object(proposal_data["receipt"]),
        observable_catalog={},
    )
    compiled = compile_hybrid_proposal(
        proposal,
        issued_by=changed["plan"]["verifier_id"],
    )
    assert isinstance(compiled.formula, Atom)

    registry = compiled.registry
    formula = compiled.formula
    attachment = compiled.attachment_contract
    if target == "formula":
        formula = replace(
            formula,
            claim=formula.claim + " with a substituted frozen claim",
        )
    elif target in {"registry_source", "registry_operational"}:
        contract = compiled.registry.contracts()[0]
        if target == "registry_source":
            contract = replace(contract, implementation=_alternate_hybrid_leg)
        else:
            replacement_digest = (
                "0" * 64 if contract.operational_digest != "0" * 64 else "1" * 64
            )
            contract = replace(contract, operational_digest=replacement_digest)
        registry = LegRegistry()
        reference = registry.register(contract)
        registry.freeze()
        formula = replace(
            formula,
            call=replace(formula.call, leg=reference),
        )
        attachment = TypedAttachmentContract.issue(
            issued_by=changed["plan"]["verifier_id"],
            registry=registry,
            boundary_types={"panel": PANEL},
        )
    elif target == "attachment":
        attachment = TypedAttachmentContract.issue(
            issued_by=changed["plan"]["verifier_id"],
            registry=registry,
            boundary_types={"panel": PANEL, "unused_panel": PANEL},
        )
    else:  # pragma: no cover - the parametrized test owns this helper.
        raise AssertionError(f"unknown compilation-tamper target {target!r}")

    freeze = ProposalFreeze.create(
        support=bundle.support,
        proposal_id=bundle.freeze.proposal_id,
        formula=formula,
        proposer_digest=bundle.freeze.proposer_digest,
        attachment_contract=attachment,
        registry=registry,
        support_gate_digest=bundle.freeze.support_gate_digest,
        verifier_nonce=bundle.freeze.verifier_nonce,
    )
    release = QueryRelease.create(
        freeze,
        bundle.release.queries,
        verifier_nonce=bundle.release.verifier_nonce,
    )
    atom_evidence = {
        query.query_id: {
            atom.path: atom.evidence.to_evidence() for atom in query.atom_inputs
        }
        for query in bundle.cold_inputs.queries
    }
    cold_inputs = ColdReplayInputs.capture(
        freeze=freeze,
        release=release,
        atom_evidence=atom_evidence,
    )
    predictions = PredictionCommitment.create(
        freeze=freeze,
        release=release,
        cold_inputs=cold_inputs,
        verifier_nonce=bundle.predictions.verifier_nonce,
    )
    labels = LabelReveal.create(
        predictions,
        bundle.labels.labels,
        verifier_nonce=bundle.labels.verifier_nonce,
    )
    malicious_bundle = RunArtifactBundle(
        support=bundle.support,
        attachment_contract=attachment,
        freeze=freeze,
        release=release,
        cold_inputs=cold_inputs,
        predictions=predictions,
        labels=labels,
    )
    changed["run_archive"] = malicious_bundle.to_archive_data()
    changed["vision"]["proposal_freeze"] = freeze.to_data()
    changed["episode"]["artifact_chain"] = malicious_bundle.chain_data()
    _reseal_outer(changed)

    # The adversarial fixture must pass the archive's own hash-chain and cold
    # replay checks; only verifier-owned recompilation should reject it.
    verify_archive_data(changed["run_archive"])
    return changed


def test_commitment_audit_names_missing_preimages_and_strict_verification_requires_them(
    completed_run,
) -> None:
    record, blob_bytes = completed_run
    audit = audit_completed_run_commitments(record)
    assert audit.run_id == record["episode"]["run_id"]
    assert audit.query_ids == ("query-0", "query-1")
    assert len(audit.receipt_digests) == 15
    assert len(audit.missing_blob_preimages) == 14
    assert not audit.byte_preimages_verified
    assert any("task manifest" in item for item in audit.unbound_outer_fields)

    with pytest.raises(MissingBlobPreimagesError, match="BlobRef") as caught:
        verify_completed_run_data(record)
    assert caught.value.blob_ids == audit.missing_blob_preimages

    verified = verify_completed_run_data(record, blob_bytes_by_id=blob_bytes)
    assert verified.byte_preimages_verified
    assert len(verified.verified_blob_ids) == 14
    from_bytes = verify_completed_run_bytes(
        canonical_json(record), blob_bytes_by_id=blob_bytes
    )
    assert from_bytes.to_data() == verified.to_data()


def test_rejected_proposal_is_archived_and_replayed_without_becoming_a_rule(
    rejected_run,
) -> None:
    record, support_bytes = rejected_run
    vision = record["vision"]
    assert vision["proposal"] is None
    attempt = vision["rejected_proposal_attempt"]
    assert attempt["model_payload"]["hybrid_claim"]["phrase"] == (
        "a form that is not symmetric"
    )
    assert attempt["parse_error"]["error_type"] == "ProposalError"
    assert "semantic negation" in attempt["parse_error"]["reason"]

    audit = audit_rejected_run_commitments(record)
    assert len(audit.missing_support_preimages) == 12
    assert any(
        "support_commitment_digest" in limitation
        for limitation in audit.unbound_outer_fields
    )
    with pytest.raises(MissingSupportPreimagesError):
        verify_rejected_run_data(record)

    verified = verify_rejected_run_data(
        record, support_bytes_by_name=support_bytes
    )
    assert verified.support_byte_preimages_verified
    assert len(verified.verified_support_preimages) == 12
    from_bytes = verify_rejected_run_bytes(
        canonical_json(record), support_bytes_by_name=support_bytes
    )
    assert from_bytes.to_data() == verified.to_data()


@pytest.mark.parametrize(
    "tamper",
    ("payload", "receipt", "parse_error", "support_identity", "attempt_digest", "schema"),
)
def test_rejected_proposal_audit_is_content_and_receipt_bound(
    rejected_run, tamper: str
) -> None:
    record, _support_bytes = rejected_run
    changed = copy.deepcopy(record)
    attempt = changed["vision"]["rejected_proposal_attempt"]
    if tamper == "payload":
        attempt["model_payload"]["confidence"] = "low"
    elif tamper == "receipt":
        attempt["receipt"]["structured_output_digest"] = "0" * 64
    elif tamper == "parse_error":
        attempt["parse_error"]["reason"] = "invented parser rejection"
        _reseal_rejected_attempt(attempt)
    elif tamper == "support_identity":
        attempt["support_presentation"][0]["content_digest"] = "0" * 64
        _reseal_rejected_attempt(attempt)
    elif tamper == "attempt_digest":
        attempt["attempt_digest"] = "sha256:" + "0" * 64
    else:
        attempt["proposal_schema"] = "gkm.bongard-visual-proposal.v2"
        _reseal_rejected_attempt(attempt)
    _reseal_outer(changed)

    with pytest.raises(RunVerificationError):
        audit_rejected_run_commitments(changed)


def test_rejected_artifact_cannot_be_resealed_after_payload_becomes_admissible(
    rejected_run,
) -> None:
    record, _support_bytes = rejected_run
    changed = copy.deepcopy(record)
    attempt = changed["vision"]["rejected_proposal_attempt"]
    payload = attempt["model_payload"]
    payload["hybrid_claim"]["phrase"] = "asymmetric form"
    receipt = attempt["receipt"]
    receipt["structured_output_digest"] = canonical_digest(payload)
    _reseal_receipt(receipt)
    _reseal_rejected_attempt(attempt)
    _reseal_outer(changed)

    with pytest.raises(RunVerificationError, match="accepted"):
        audit_rejected_run_commitments(changed)


def test_completed_run_cannot_carry_a_rejected_attempt(rejected_run, completed_run) -> None:
    rejected_record, _ = rejected_run
    completed_record, _ = completed_run
    changed = copy.deepcopy(completed_record)
    changed["vision"]["rejected_proposal_attempt"] = copy.deepcopy(
        rejected_record["vision"]["rejected_proposal_attempt"]
    )
    _reseal_outer(changed)
    with pytest.raises(RunVerificationError, match="completed vision"):
        audit_completed_run_commitments(changed)


@pytest.mark.parametrize(
    "tamper",
    ("order", "policy", "panel", "payload", "disposition", "gate_digest"),
)
def test_cold_verifier_rejects_support_gate_tampering(completed_run, tamper: str) -> None:
    record, _blob_bytes = completed_run
    changed = copy.deepcopy(record)
    gate = changed["vision"]["support_gate"]
    entries = gate["ordered_entries"]
    if tamper == "order":
        entries[0], entries[1] = entries[1], entries[0]
    elif tamper == "policy":
        gate["policy"]["positive_outcome"] = "anything"
    elif tamper == "panel":
        entries[0]["panel"]["sha256"] = "0" * 64
    elif tamper == "payload":
        entries[0]["observer_artifact"]["payload"]["reason"] = "invented"
    elif tamper == "disposition":
        entries[0]["evidence"]["disposition"] = "present"
    else:
        gate["gate_digest"] = "0" * 64
    _reseal_outer(changed)

    with pytest.raises(RunVerificationError, match="support|gate|observation|evidence"):
        audit_completed_run_commitments(changed)


def test_nonmatch_summary_is_bound_into_archived_gate_evidence(completed_run) -> None:
    record, _blob_bytes = completed_run
    changed = copy.deepcopy(record)
    proposal_data = changed["vision"]["proposal"]
    proposal = parse_rule_proposal(
        proposal_data["model_payload"],
        receipt=_receipt_object(proposal_data["receipt"]),
        observable_catalog={},
    )
    entry = changed["vision"]["support_gate"]["ordered_entries"][0]
    observation_data = entry["observer_artifact"]
    payload = observation_data["payload"]
    assert payload["disposition"] == "nonmatch"
    payload["reason"] = "an adversarially substituted overall summary"
    receipt = observation_data["receipt"]
    receipt["structured_output_digest"] = canonical_digest(payload)
    _reseal_receipt(receipt)
    observation = parse_hybrid_observation_or_error(
        proposal, payload, _receipt_object(receipt)
    )
    entry["observer_artifact"] = observation.to_dict()
    _reseal_outer(changed)

    with pytest.raises(RunVerificationError, match="gate evidence differs"):
        audit_completed_run_commitments(changed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("codex_launcher_digest", hashlib.sha256(b"substituted-launcher").hexdigest()),
        ("codex_cli_version", "codex-cli substituted"),
        ("cloud_config_bundle_cache_binding", "sha256:" + "0" * 64),
        ("isolation_policy", "substituted-isolation-policy"),
        ("requested_model", "gpt-5.5"),
        ("requested_reasoning_effort", "high"),
    ),
)
def test_cold_verifier_rejects_transport_identity_substitution(
    completed_run, field: str, replacement: str
) -> None:
    record, _blob_bytes = completed_run
    changed = copy.deepcopy(record)
    receipt = changed["vision"]["support_gate"]["ordered_entries"][0][
        "observer_artifact"
    ]["receipt"]
    receipt[field] = replacement
    if field == "isolation_policy":
        body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
        receipt["receipt_digest"] = canonical_digest(body)
    else:
        _reseal_receipt(receipt)
    _reseal_outer(changed)

    with pytest.raises(
        RunVerificationError, match="transport identity|invalid Codex receipt"
    ):
        audit_completed_run_commitments(changed)


@pytest.mark.parametrize("target", ("run_id", "status", "score", "artifact_chain"))
def test_task_independent_episode_fields_are_bound_to_verified_bundle(
    completed_run, target: str
) -> None:
    record, _blob_bytes = completed_run
    changed = copy.deepcopy(record)
    if target == "run_id":
        changed["episode"][target] = "run-ffffffffffffffffffffffffffffffff"
    elif target == "status":
        changed["episode"][target] = "proposal_error"
    elif target == "score":
        changed["episode"][target]["image_correct"] ^= 1
    else:
        changed["episode"][target]["support"] = "f" * 64
    _reseal_outer(changed)
    with pytest.raises(RunVerificationError):
        audit_completed_run_commitments(changed)


def test_outer_schema_and_corpus_support_binding_are_strict(completed_run) -> None:
    record, _blob_bytes = completed_run
    extra = copy.deepcopy(record)
    extra["episode"]["invented"] = True
    _reseal_outer(extra)
    with pytest.raises(RunVerificationError, match="episode fields"):
        audit_completed_run_commitments(extra)

    wrong_corpus = copy.deepcopy(record)
    wrong_corpus["corpus_manifest_digest"] = "sha256:" + "a" * 64
    wrong_corpus["exposure"]["corpus_manifest_digest"] = wrong_corpus[
        "corpus_manifest_digest"
    ]
    _reseal_outer(wrong_corpus)
    with pytest.raises(RunVerificationError, match="support.corpus_digest"):
        audit_completed_run_commitments(wrong_corpus)

    malformed_release = copy.deepcopy(record)
    malformed_release["official_release"] = {"schema": "invented"}
    _reseal_outer(malformed_release)
    with pytest.raises(RunVerificationError, match="official_release descriptor"):
        audit_completed_run_commitments(malformed_release)

    descriptor = OfficialReleaseDescriptor(
        release_id="fixture-release",
        archive_filename="fixture.zip",
        archive_sha256="sha256:" + "b" * 64,
        archive_size_bytes=1,
        split_filename="ShapeBongard_V2_split.json",
        split_sha256=record["split_source_digest"],
        split_size_bytes=1,
        upstream_repository="fixture/repository",
        upstream_commit="c" * 40,
        family_counts=(),
        primary_split_counts=(),
        regime_counts=(),
        task_ids_sha256="sha256:" + "d" * 64,
        corpus_manifest_sha256=record["corpus_manifest_digest"],
    )
    bound_release = copy.deepcopy(record)
    bound_release["official_release"] = descriptor.to_dict()
    _reseal_outer(bound_release)
    assert audit_completed_run_commitments(
        bound_release
    ).official_release_digest == descriptor.digest

    wrong_split_release = copy.deepcopy(bound_release)
    wrong_split_release["official_release"]["split"]["sha256"] = (
        "sha256:" + "e" * 64
    )
    _reseal_outer(wrong_split_release)
    with pytest.raises(RunVerificationError, match="split digest differs"):
        audit_completed_run_commitments(wrong_split_release)


def test_public_plan_is_canonical_and_bound_to_episode_and_archive(completed_run) -> None:
    record, _blob_bytes = completed_run
    assert canonical_digest(record["plan"]) == record["episode"]["plan_digest"]
    assert record["plan"]["support_commitment_digest"] == record["run_archive"][
        "chain"
    ]["support"]

    swapped_task = copy.deepcopy(record)
    swapped_task["plan"]["task_id"] = "bd_swapped_0000"
    swapped_task["episode"]["plan_digest"] = canonical_digest(swapped_task["plan"])
    swapped_task["exposure"]["plan_digest"] = swapped_task["episode"]["plan_digest"]
    _reseal_outer(swapped_task)
    with pytest.raises(RunVerificationError, match="task_id differs"):
        audit_completed_run_commitments(swapped_task)

    changed_support = copy.deepcopy(record)
    changed_support["plan"]["support_commitment_digest"] = "f" * 64
    changed_support["episode"]["plan_digest"] = canonical_digest(
        changed_support["plan"]
    )
    changed_support["exposure"]["plan_digest"] = changed_support["episode"][
        "plan_digest"
    ]
    _reseal_outer(changed_support)
    with pytest.raises(RunVerificationError, match="archive support"):
        audit_completed_run_commitments(changed_support)


def test_proposal_reparses_from_payload_receipt_and_must_match_freeze(completed_run) -> None:
    record, _blob_bytes = completed_run
    changed = copy.deepcopy(record)
    proposal_data = changed["vision"]["proposal"]
    payload = proposal_data["model_payload"]
    payload["hybrid_claim"]["phrase"] = "avian-like angular form"
    receipt = proposal_data["receipt"]
    receipt["structured_output_digest"] = canonical_digest(payload)
    _reseal_receipt(receipt)
    parsed = parse_rule_proposal(
        payload,
        receipt=_receipt_object(receipt),
        observable_catalog={},
    )
    changed["vision"]["proposal"] = parsed.to_dict()
    _reseal_outer(changed)
    with pytest.raises(RunVerificationError, match="frozen proposer_digest"):
        audit_completed_run_commitments(changed)


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("formula", "frozen formula differs"),
        ("registry_source", "frozen registry snapshot/digest differs"),
        ("registry_operational", "frozen registry snapshot/digest differs"),
        ("attachment", "frozen attachment contract differs"),
    ),
)
def test_cold_verification_recompiles_hybrid_proposal_in_verifier_owned_code(
    completed_run,
    target: str,
    message: str,
) -> None:
    record, _blob_bytes = completed_run
    changed = _reseal_with_noncanonical_compilation(record, target)

    with pytest.raises(RunVerificationError, match=message):
        audit_completed_run_commitments(changed)


@pytest.mark.parametrize("which", ("proposal", "query-0", "query-1"))
def test_every_codex_receipt_is_validated(completed_run, which: str) -> None:
    record, _blob_bytes = completed_run
    changed = copy.deepcopy(record)
    if which == "proposal":
        receipt = changed["vision"]["proposal"]["receipt"]
    else:
        receipt = changed["vision"]["observations"][which]["receipt"]
    receipt["receipt_digest"] = "0" * 64
    _reseal_outer(changed)
    with pytest.raises(RunVerificationError, match="invalid Codex receipt"):
        audit_completed_run_commitments(changed)


def test_observations_exactly_cover_release_and_bind_proposal(completed_run) -> None:
    record, _blob_bytes = completed_run
    missing = copy.deepcopy(record)
    del missing["vision"]["observations"]["query-1"]
    _reseal_outer(missing)
    with pytest.raises(RunVerificationError, match="cover exactly"):
        audit_completed_run_commitments(missing)

    invented = copy.deepcopy(record)
    invented["vision"]["observations"]["query-2"] = copy.deepcopy(
        invented["vision"]["observations"]["query-0"]
    )
    _reseal_outer(invented)
    with pytest.raises(RunVerificationError, match="cover exactly"):
        audit_completed_run_commitments(invented)

    wrong_proposal = copy.deepcopy(record)
    wrong_proposal["vision"]["observations"]["query-0"]["proposal_digest"] = (
        "sha256:" + "f" * 64
    )
    _reseal_outer(wrong_proposal)
    with pytest.raises(RunVerificationError, match="does not reproduce"):
        audit_completed_run_commitments(wrong_proposal)


def test_observer_named_image_receipt_is_tied_to_released_blob_identity(
    completed_run,
) -> None:
    record, _blob_bytes = completed_run
    changed = copy.deepcopy(record)
    receipt = changed["vision"]["observations"]["query-0"]["receipt"]
    receipt["panel_view_digest"] = "0" * 64
    _reseal_receipt(receipt)
    _reseal_outer(changed)
    with pytest.raises(RunVerificationError, match="image view differs"):
        audit_completed_run_commitments(changed)


def test_reconstructed_observation_evidence_and_provenance_must_equal_cold_atom(
    completed_run,
) -> None:
    record, _blob_bytes = completed_run
    changed = copy.deepcopy(record)
    proposal_data = changed["vision"]["proposal"]
    proposal = parse_rule_proposal(
        proposal_data["model_payload"],
        receipt=_receipt_object(proposal_data["receipt"]),
        observable_catalog={},
    )
    old_observation = changed["vision"]["observations"]["query-0"]
    payload = {
        "epistemic_status": HYBRID_EPISTEMIC_STATUS,
        "disposition": "indeterminate",
        "observed_cue_ids": [],
        "missing_cue_ids": [],
        "missing_cue_reasons": [],
        "visibility_certificate": None,
        "reason": "the boundary junctions are visually ambiguous",
        "error_type": None,
    }
    receipt = old_observation["receipt"]
    receipt["structured_output_digest"] = canonical_digest(payload)
    _reseal_receipt(receipt)
    observation = _parse_hybrid_observation(
        proposal, payload, _receipt_object(receipt)
    )
    changed["vision"]["observations"]["query-0"] = observation.to_dict()
    _reseal_outer(changed)
    with pytest.raises(RunVerificationError, match="cold atom evidence"):
        audit_completed_run_commitments(changed)


def test_wrong_blob_preimage_and_noncanonical_outer_bytes_fail(completed_run) -> None:
    record, blob_bytes = completed_run
    changed_bytes = dict(blob_bytes)
    target = "query-panel-0"
    changed_bytes[target] = changed_bytes[target] + b"tamper"
    with pytest.raises(RunVerificationError, match="differs from BlobRef"):
        verify_completed_run_data(record, blob_bytes_by_id=changed_bytes)

    pretty = json.dumps(record, sort_keys=True, indent=2).encode("utf-8")
    with pytest.raises(RunVerificationError, match="not canonical"):
        verify_completed_run_bytes(pretty, blob_bytes_by_id=blob_bytes)
