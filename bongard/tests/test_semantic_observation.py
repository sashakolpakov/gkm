from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw
import pytest

from bongard.artifacts import canonical_digest
from bongard.blind_soft_transport import BlindSoftVerifierContext
from bongard.evidence import Disposition
from bongard.semantic_observation import (
    SemanticObservationError,
    VisualSemanticObservationArtifact,
    observe_visual_semantic_panel,
)
from bongard.semantic_protocol import (
    build_prospective_soft_scorer_protocol,
    build_visual_semantic_policy,
)
from bongard.semantic_synthesis import compile_visual_semantic_proposal
from bongard.soft_predicates import SoftFamilyDevelopmentUnit, SoftScorerFamily
from bongard.transport import (
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    CodexReceipt,
    CodexStructuredResult,
    named_image_set_digest,
)
from bongard.typed_visual_proposal import (
    PANEL_DESCRIPTION_KEYS,
    parse_typed_visual_proposal,
)
from bongard.visual_predicate_catalog import DIRECT_VISUAL_ATOM_CATALOG


MODEL = "fixture-scorer"
EFFORT = "medium"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _family() -> SoftScorerFamily:
    protocol = build_prospective_soft_scorer_protocol(
        proposer_model_id="fixture-proposer",
        proposer_reasoning_effort=EFFORT,
        scorer_model_id=MODEL,
        scorer_reasoning_effort=EFFORT,
        score_bin_edges=(0.0, 0.5, 1.0),
        affirmative_boundary=0.5,
        confidence_level=0.1,
        minimum_clusters_per_bin=8,
    )
    units: list[SoftFamilyDevelopmentUnit] = []
    for score, label, prefix, bin_index in (
        (0.0, False, "low", 0),
        (1.0, True, "high", 1),
    ):
        for index in range(8):
            units.append(
                SoftFamilyDevelopmentUnit(
                    observation_id=f"{prefix}-{index:02d}",
                    task_id=f"task-{prefix}-{index:02d}",
                    panel_digest=_digest(f"panel-{prefix}-{index}"),
                    claim_digest=_digest(f"claim-{prefix}-{index}"),
                    scorer_protocol_digest=protocol.digest(),
                    proposer_call_id=f"proposer-{prefix}-{index:02d}",
                    scorer_call_id=f"scorer-{prefix}-{index:02d}",
                    dependence_cluster_id=f"cluster-{prefix}-{index:02d}",
                    score_record_digest=_digest(f"score-{prefix}-{index}"),
                    annotation_receipt_digest=_digest(
                        f"annotation-{prefix}-{index}"
                    ),
                    score=score,
                    affirmative_label=label,
                    score_bin_index=bin_index,
                )
            )
    return SoftScorerFamily.fit(
        protocol,
        tuple(sorted(units, key=lambda item: item.observation_id)),
        expected_protocol_digest=protocol.digest(),
    )


def _compiled(kind: str):
    if kind not in {"direct", "soft", "mixed"}:
        raise ValueError(kind)
    family = _family()
    direct = (
        []
        if kind == "soft"
        else [
            {
                "catalog_key": "component.count",
                "comparison": "equal",
                "arguments": {"target_count": 1},
            }
        ]
    )
    soft = (
        None
        if kind == "direct"
        else {
            "positive_description": "bird-like articulated organization",
            "cue_descriptions": [
                "one compact central body mass",
                "two lateral wing-like extensions",
            ],
        }
    )
    proposal = parse_typed_visual_proposal(
        {
            "positive_description": "one compact ink arrangement",
            "panel_descriptions": {
                name: f"literal panel {index}"
                for index, name in enumerate(PANEL_DESCRIPTION_KEYS)
            },
            "view": "carrier_shape",
            "deterministic_atoms": direct,
            "soft_claim": soft,
            "formula": {
                "kind": "all",
                "atom_indices": list(range(len(direct) + (soft is not None))),
            },
        },
        catalog=DIRECT_VISUAL_ATOM_CATALOG,
        scorer_protocol_digest=family.protocol_digest,
    )
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    return compile_visual_semantic_proposal(
        proposal,
        policy=policy,
        expected_policy_digest=policy.digest(),
        family=family,
    )


def _panel_bytes() -> bytes:
    output = BytesIO()
    image = Image.new("L", (32, 32), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 23, 23), fill=0)
    image.save(output, format="PNG")
    return output.getvalue()


def _context(precommit: str) -> BlindSoftVerifierContext:
    return BlindSoftVerifierContext(
        task_id="SECRET_TASK_bd_000001",
        panel_id="SECRET_SIDE_positive_slot_0",
        proposer_call_id="SECRET_PROPOSER_CALL",
        proposer_receipt_digest=_digest("proposer receipt"),
        scorer_call_id="scorer-call-0001",
        pre_observation_commitment_digest=precommit,
    )


def _receipt(
    prompt: str,
    paths: Sequence[str],
    names: Sequence[str],
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> CodexReceipt:
    identities = [
        {
            "name": name,
            "byte_count": len(Path(path).read_bytes()),
            "content_digest": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        for path, name in zip(paths, names, strict=True)
    ]
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = canonical_digest(dict(schema))
    view_digest = canonical_digest(identities)
    set_digest = named_image_set_digest(paths, names)
    body: dict[str, Any] = {
        "schema": CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": MODEL,
        "reported_model": "",
        "model_identity_evidence": "explicit-cli-model-flag;jsonl-omits-model",
        "requested_reasoning_effort": EFFORT,
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "output_tokens": 2,
        "reasoning_output_tokens": 1,
        "thread_id": "00000000-0000-4000-8000-000000000001",
        "codex_cli_version": "codex-cli test",
        "codex_launcher_digest": "b" * 64,
        "cloud_config_bundle_cache_binding": "absent",
        "task_digest": prompt_digest,
        "current_source_digest": "",
        "current_log_digest": "",
        "prompt_digest": prompt_digest,
        "input_digest_schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "input_digest": canonical_digest(
            {
                "schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
                "task": prompt,
                "ordered_image_identities": identities,
                "image_view_digest": view_digest,
                "image_set_digest": set_digest,
                "prompt_digest": prompt_digest,
                "output_schema_digest": schema_digest,
            }
        ),
        "output_schema_digest": schema_digest,
        "panel_view_digest": view_digest,
        "panel_set_digest": set_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": "c" * 64,
        "event_types": [
            "thread.started",
            "turn.started",
            "item.completed",
            "turn.completed",
        ],
        "item_types": ["agent_message"],
        "isolation_policy": CODEX_ISOLATION_POLICY,
        "outcome": "success",
    }
    body["receipt_digest"] = canonical_digest(body)
    return CodexReceipt(
        **{
            **body,
            "event_types": tuple(body["event_types"]),
            "item_types": tuple(body["item_types"]),
        }
    )


def test_direct_observation_uses_no_transport_and_cold_replays_pixels() -> None:
    compiled = _compiled("direct")
    precommit = _digest("outer prequery envelope")

    def forbidden_transport(*args, **kwargs):  # pragma: no cover - must not run.
        raise AssertionError("direct-only observation called soft transport")

    artifact = observe_visual_semantic_panel(
        _panel_bytes(),
        compiled,
        protocol=compiled.family.protocol,
        context=_context(precommit),
        pre_observation_commitment_digest=precommit,
        transport=forbidden_transport,
    )

    assert not artifact.transport_attempted
    assert artifact.scorer_artifact is None
    assert tuple(artifact.evidence_by_path()) == ((),)
    assert artifact.formula_evidence.disposition is Disposition.PRESENT
    measurement = artifact.to_support_gate_measurement()
    assert measurement.transport_attempted is False
    assert measurement.evidence.disposition is Disposition.PRESENT
    assert VisualSemanticObservationArtifact.from_data(
        artifact.to_data(),
        compiled=compiled,
        protocol=compiled.family.protocol,
        expected_digest=artifact.digest,
        panel_png=_panel_bytes(),
    ) == artifact


def test_mixed_observation_scores_once_neutrally_and_replays_archive() -> None:
    compiled = _compiled("mixed")
    precommit = _digest("accepted transport plus support prequery envelope")
    context = _context(precommit)
    calls = 0

    def transport(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        assert tuple(names) == ("query.png",)
        assert tuple(Path(path).name for path in paths) == ("query.png",)
        assert kwargs["model"] == MODEL
        assert kwargs["reasoning_effort"] == EFFORT
        for secret in (
            context.task_id,
            context.panel_id,
            context.proposer_call_id,
            context.proposer_receipt_digest,
            context.scorer_call_id,
            precommit,
        ):
            assert secret not in prompt
        claim = compiled.lowering_archive.soft_lowering
        assert claim is not None
        payload = {
            "cue_judgments": [
                {
                    "cue_id": cue.cue_id,
                    "judgment": "supported",
                    "witness_ids": ["panel:geometry"],
                }
                for cue in claim.claim.cues
            ]
        }
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(prompt, paths, names, schema, payload),
        )

    artifact = observe_visual_semantic_panel(
        _panel_bytes(),
        compiled,
        protocol=compiled.family.protocol,
        context=context,
        pre_observation_commitment_digest=precommit,
        transport=transport,
    )

    assert calls == 1
    assert artifact.transport_attempted
    assert artifact.scorer_artifact is not None
    assert artifact.scorer_artifact.record.score == 1.0
    assert tuple(artifact.evidence_by_path()) == ((0,), (1,))
    assert all(
        evidence.disposition is Disposition.PRESENT
        for evidence in artifact.evidence_by_path().values()
    )
    assert artifact.formula_evidence.disposition is Disposition.PRESENT
    assert artifact.to_support_gate_measurement().transport_attempted is True

    decoded = VisualSemanticObservationArtifact.from_data(
        artifact.to_data(),
        compiled=compiled,
        protocol=compiled.family.protocol,
        expected_digest=artifact.digest,
        panel_png=_panel_bytes(),
    )
    assert decoded == artifact


def test_soft_transport_failure_is_error_not_zero_or_absence() -> None:
    compiled = _compiled("soft")
    precommit = _digest("outer prequery failure test")
    calls = 0

    def transport(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("transport unavailable")

    artifact = observe_visual_semantic_panel(
        _panel_bytes(),
        compiled,
        protocol=compiled.family.protocol,
        context=_context(precommit),
        pre_observation_commitment_digest=precommit,
        transport=transport,
    )

    assert calls == 1
    assert artifact.scorer_artifact is not None
    assert artifact.scorer_artifact.record.outcome == "transport_error"
    assert artifact.scorer_artifact.record.score is None
    assert artifact.atom_evidence[0].evidence.disposition is Disposition.ERROR
    assert artifact.formula_evidence.disposition is Disposition.ERROR
    assert VisualSemanticObservationArtifact.from_data(
        artifact.to_data(),
        compiled=compiled,
        protocol=compiled.family.protocol,
        expected_digest=artifact.digest,
        panel_png=_panel_bytes(),
    ) == artifact


def test_soft_parser_failure_is_error_not_zero_or_absence() -> None:
    compiled = _compiled("soft")
    precommit = _digest("outer prequery parser failure test")

    def transport(prompt, paths, names, schema, **kwargs):
        payload: dict[str, Any] = {"cue_judgments": []}
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(prompt, paths, names, schema, payload),
        )

    artifact = observe_visual_semantic_panel(
        _panel_bytes(),
        compiled,
        protocol=compiled.family.protocol,
        context=_context(precommit),
        pre_observation_commitment_digest=precommit,
        transport=transport,
    )

    assert artifact.scorer_artifact is not None
    assert artifact.scorer_artifact.record.outcome == "parser_error"
    assert artifact.scorer_artifact.record.score is None
    assert artifact.atom_evidence[0].evidence.disposition is Disposition.ERROR
    assert artifact.formula_evidence.disposition is Disposition.ERROR


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.__setitem__("pre_observation_commitment_digest", "0" * 64),
        lambda data: data["witness_bundle"]["base_packet"].__setitem__(
            "width_pixels", 99
        ),
        lambda data: data["atom_evidence"][0].__setitem__("path", [7]),
        lambda data: data["formula_evidence"].__setitem__(
            "disposition", "certified_absent"
        ),
    ],
)
def test_archive_tampering_fails_closed(mutation) -> None:
    compiled = _compiled("direct")
    precommit = _digest("outer prequery tamper test")
    artifact = observe_visual_semantic_panel(
        _panel_bytes(),
        compiled,
        protocol=compiled.family.protocol,
        context=_context(precommit),
        pre_observation_commitment_digest=precommit,
    )
    changed = deepcopy(artifact.to_data())
    mutation(changed)
    with pytest.raises((SemanticObservationError, TypeError, ValueError)):
        VisualSemanticObservationArtifact.from_data(
            changed,
            compiled=compiled,
            protocol=compiled.family.protocol,
            expected_digest=artifact.digest,
            panel_png=_panel_bytes(),
        )


def test_explicit_precommit_must_match_verifier_context() -> None:
    compiled = _compiled("direct")
    with pytest.raises(SemanticObservationError, match="another pre-observation"):
        observe_visual_semantic_panel(
            _panel_bytes(),
            compiled,
            protocol=compiled.family.protocol,
            context=_context(_digest("context parent")),
            pre_observation_commitment_digest=_digest("different outer parent"),
        )
