from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
import pytest

from bongard.artifacts import canonical_digest, canonical_json
from bongard.soft_predicates import SoftScorerProtocol
from bongard.transport import (
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    STRUCTURED_INPUT_DIGEST_SCHEMA,
    CodexReceipt,
    CodexStructuredResult,
    semantic_panel_set_digest,
)
from bongard.typed_visual_proposal import (
    ArgumentKind,
    AtomArgument,
    RegisteredAtomCatalog,
    RegisteredAtomOption,
    RegisteredAtomSpec,
    TYPED_VISUAL_PROPOSER_GRAMMAR_ID,
    TYPED_VISUAL_PROPOSER_PROMPT_ID,
    typed_visual_proposal_grammar_digest,
    typed_visual_proposal_prompt,
    typed_visual_proposal_prompt_digest,
    typed_visual_proposal_schema,
)
from bongard.typed_visual_transport import (
    RejectedTypedVisualProposalAttempt,
    TypedVisualProposalRejected,
    TypedVisualTransportError,
    TypedVisualTransportResult,
    propose_typed_visual,
)
from bongard.visual_predicate_catalog import DIRECT_VISUAL_ATOM_CATALOG


MODEL = "gpt-test"
EFFORT = "medium"
CANONICAL_NAMES = tuple(
    [f"pos_{index}.png" for index in range(6)]
    + [f"neg_{index}.png" for index in range(6)]
)


@pytest.fixture
def catalog() -> RegisteredAtomCatalog:
    return RegisteredAtomCatalog(
        (
            RegisteredAtomSpec(
                catalog_key="component.count",
                affirmative_description=(
                    "the panel has a registered exact number of separated ink components"
                ),
                arguments=(AtomArgument("target_count", ArgumentKind.INTEGER),),
                allowed_options=(
                    RegisteredAtomOption.from_mapping(
                        "equal", {"target_count": 1}
                    ),
                    RegisteredAtomOption.from_mapping(
                        "equal", {"target_count": 2}
                    ),
                ),
            ),
        )
    )


@pytest.fixture
def protocol(catalog: RegisteredAtomCatalog) -> SoftScorerProtocol:
    return SoftScorerProtocol(
        family_id="typed-visual-transport-fixture",
        version="1",
        proposer_grammar_id=TYPED_VISUAL_PROPOSER_GRAMMAR_ID,
        proposer_grammar_digest=typed_visual_proposal_grammar_digest(catalog),
        proposer_model_id=MODEL,
        proposer_reasoning_effort=EFFORT,
        proposer_prompt_id=TYPED_VISUAL_PROPOSER_PROMPT_ID,
        proposer_prompt_digest=typed_visual_proposal_prompt_digest(catalog),
        scorer_model_id=MODEL,
        scorer_reasoning_effort=EFFORT,
        scorer_prompt_template_id="fixture-blind-scorer-template-v1",
        scorer_prompt_template_digest=hashlib.sha256(b"scorer-template").hexdigest(),
        scorer_decoder_id="fixture-ordinal-decoder-v1",
        scorer_decoder_digest=hashlib.sha256(b"scorer-decoder").hexdigest(),
        ordinal_map=(
            ("supported", 1.0),
            ("ambiguous", 0.5),
            ("unsupported", 0.0),
        ),
        aggregation="min",
        witness_extractor_id="fixture-visual-witnesses-v1",
        witness_extractor_digest=hashlib.sha256(b"witnesses").hexdigest(),
        support_gate_id="fixture-support-gate-v1",
        support_gate_digest=hashlib.sha256(b"support-gate").hexdigest(),
        score_bin_edges=(0.0, 0.5, 1.0),
        affirmative_boundary=0.5,
        confidence_level=0.8,
        minimum_clusters_per_bin=2,
    )


def _payload(description: str = "two separated ink components") -> dict[str, Any]:
    return {
        "positive_description": description,
        "panel_descriptions": {
            **{
                f"pos_{index}": f"two separated marks, presentation {index}"
                for index in range(6)
            },
            **{
                f"neg_{index}": f"one compact mark, presentation {index}"
                for index in range(6)
            },
        },
        "view": "relational",
        "deterministic_atoms": [
            {
                "catalog_key": "component.count",
                "comparison": "equal",
                "arguments": {"target_count": 2},
            }
        ],
        "soft_claim": None,
        "formula": {"kind": "all", "atom_indices": [0]},
    }


def _write_supports(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    positives: list[Path] = []
    negatives: list[Path] = []
    for side, collection, offset in (
        ("positive", positives, 0),
        ("negative", negatives, 6),
    ):
        for index in range(6):
            # These names simulate metadata that must stop at the copy boundary.
            path = (
                tmp_path
                / f"SECRET_QUERY_TASK_SOURCE_{side}_{index}_official.png"
            )
            image = Image.new("L", (20, 20), color=255)
            pixels = image.load()
            assert pixels is not None
            marker = index + offset
            for coordinate in range(marker + 1):
                pixels[coordinate % 20, coordinate // 20] = 0
            image.save(path, format="PNG")
            collection.append(path)
    return positives, negatives


def _receipt(
    prompt: str,
    paths: Sequence[str],
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    model: str = MODEL,
    effort: str = EFFORT,
) -> CodexReceipt:
    identities = [
        {
            "name": Path(path).name,
            "byte_count": len(Path(path).read_bytes()),
            "content_digest": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = canonical_digest(dict(schema))
    panel_view_digest = canonical_digest(identities)
    panel_set_digest = semantic_panel_set_digest(paths)
    input_envelope = {
        "schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "task": prompt,
        "ordered_panel_identities": identities,
        "panel_view_digest": panel_view_digest,
        "panel_set_digest": panel_set_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": schema_digest,
    }
    body: dict[str, Any] = {
        "schema": CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": model,
        "reported_model": "",
        "model_identity_evidence": "explicit-cli-model-flag;jsonl-omits-model",
        "requested_reasoning_effort": effort,
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
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "input_digest": canonical_digest(input_envelope),
        "output_schema_digest": schema_digest,
        "panel_view_digest": panel_view_digest,
        "panel_set_digest": panel_set_digest,
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


def test_one_support_only_call_is_canonical_and_byte_bound(
    tmp_path: Path,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
) -> None:
    positives, negatives = _write_supports(tmp_path)
    source_paths = {str(path.resolve()) for path in (*positives, *negatives)}
    expected_bytes = [path.read_bytes() for path in (*positives, *negatives)]
    observed_paths: tuple[str, ...] = ()
    calls = 0

    def transport(prompt, paths, schema, **kwargs):
        nonlocal calls, observed_paths
        calls += 1
        observed_paths = tuple(paths)
        assert tuple(Path(path).name for path in paths) == CANONICAL_NAMES
        assert all(Path(path).is_absolute() for path in paths)
        assert source_paths.isdisjoint(paths)
        assert [Path(path).read_bytes() for path in paths] == expected_bytes
        assert "SECRET_QUERY_TASK_SOURCE" not in prompt
        assert "SECRET_QUERY_TASK_SOURCE" not in json.dumps(kwargs)
        assert prompt == typed_visual_proposal_prompt(catalog)
        assert schema == typed_visual_proposal_schema(catalog)
        assert kwargs["model"] == MODEL
        assert kwargs["reasoning_effort"] == EFFORT
        payload = _payload()
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(prompt, paths, schema, payload),
        )

    result = propose_typed_visual(
        positives,
        negatives,
        catalog=catalog,
        protocol=protocol,
        model=MODEL,
        reasoning_effort=EFFORT,
        transport=transport,
    )

    assert calls == 1
    assert result.proposal.positive_description == "two separated ink components"
    assert result.catalog_digest == catalog.digest
    assert result.scorer_protocol_digest == protocol.digest()
    assert tuple(item.name for item in result.support_presentation) == CANONICAL_NAMES
    assert tuple(item.content_digest for item in result.support_presentation) == tuple(
        hashlib.sha256(value).hexdigest() for value in expected_bytes
    )
    archived = result.to_data()
    assert archived["result_digest"] == result.digest
    assert archived["support_presentation_digest"] == (
        result.support_presentation_digest
    )
    assert "SECRET_QUERY_TASK_SOURCE" not in json.dumps(archived)
    assert all(not Path(path).exists() for path in observed_paths)

    restored = TypedVisualTransportResult.from_data(
        archived,
        catalog=catalog,
        protocol=protocol,
        expected_digest=result.digest,
    )
    assert restored == result
    assert calls == 1

    expected_content = result._uncached_content_data()
    expected_bytes = canonical_json(expected_content)
    expected_digest = hashlib.sha256(expected_bytes).hexdigest()
    assert canonical_json(result.content_data()) == expected_bytes
    assert result.digest == expected_digest
    detached = result.content_data()
    detached["catalog_digest"] = "0" * 64
    assert canonical_json(result.content_data()) == expected_bytes
    assert result.digest == expected_digest

    object.__setattr__(result, "catalog_digest", "0" * 64)
    changed_bytes = canonical_json(result._uncached_content_data())
    assert changed_bytes != expected_bytes
    assert canonical_json(result.content_data()) == changed_bytes
    assert result.digest == hashlib.sha256(changed_bytes).hexdigest()
    with pytest.raises(
        TypedVisualTransportError,
        match="another atom catalog",
    ):
        TypedVisualTransportResult.from_data(
            result.to_data(),
            catalog=catalog,
            protocol=protocol,
        )


def test_parser_rejection_is_a_strict_attempt_not_a_false_proposal(
    tmp_path: Path,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
) -> None:
    positives, negatives = _write_supports(tmp_path)
    calls = 0

    def transport(prompt, paths, schema, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        payload = _payload("no enclosed loops")
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(prompt, paths, schema, payload),
        )

    with pytest.raises(TypedVisualProposalRejected) as rejected:
        propose_typed_visual(
            positives,
            negatives,
            catalog=catalog,
            protocol=protocol,
            model=MODEL,
            reasoning_effort=EFFORT,
            transport=transport,
        )

    assert calls == 1
    attempt = rejected.value.attempt
    data = attempt.to_data()
    assert data["attempt_digest"] == attempt.digest
    assert data["parse_error"]["error_type"] == "TypedVisualProposalError"
    assert "forbidden no" in data["parse_error"]["reason"]
    assert data["model_payload"] == _payload("no enclosed loops")
    assert "proposal" not in data
    assert "prediction" not in data
    assert "disposition" not in data

    restored = RejectedTypedVisualProposalAttempt.from_data(
        data,
        catalog=catalog,
        protocol=protocol,
        expected_digest=attempt.digest,
    )
    assert restored == attempt
    assert calls == 1


@pytest.mark.parametrize("case", ("wrong_count", "duplicate", "missing", "not_png"))
def test_support_contract_rejects_before_transport(
    tmp_path: Path,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
    case: str,
) -> None:
    positives, negatives = _write_supports(tmp_path)
    if case == "wrong_count":
        positives.pop()
    elif case == "duplicate":
        negatives[-1] = positives[0]
    elif case == "missing":
        negatives[-1] = tmp_path / "missing.png"
    else:
        negatives[-1].write_bytes(b"not a PNG")
    calls = 0

    def transport(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("transport must not run")

    with pytest.raises(TypedVisualTransportError):
        propose_typed_visual(
            positives,
            negatives,
            catalog=catalog,
            protocol=protocol,
            transport=transport,
        )
    assert calls == 0


@pytest.mark.parametrize(
    "case",
    ("wrong_model", "wrong_effort", "wrong_grammar", "wrong_prompt"),
)
def test_proposer_execution_must_match_prospective_protocol_before_call(
    tmp_path: Path,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
    case: str,
) -> None:
    positives, negatives = _write_supports(tmp_path)
    selected = protocol
    overrides: dict[str, str] = {}
    if case == "wrong_model":
        overrides["model"] = "different-model"
    elif case == "wrong_effort":
        overrides["reasoning_effort"] = "high"
    elif case == "wrong_grammar":
        selected = replace(protocol, proposer_grammar_digest="d" * 64)
    else:
        selected = replace(protocol, proposer_prompt_digest="e" * 64)
    calls = 0

    def transport(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("transport must not run")

    with pytest.raises(TypedVisualTransportError):
        propose_typed_visual(
            positives,
            negatives,
            catalog=catalog,
            protocol=selected,
            transport=transport,
            **overrides,
        )
    assert calls == 0


def test_receipt_substitution_is_rejected(
    tmp_path: Path,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
) -> None:
    positives, negatives = _write_supports(tmp_path)

    def transport(prompt, paths, schema, **kwargs):
        del kwargs
        payload = _payload()
        receipt = _receipt(prompt, paths, schema, payload)
        # It remains a CodexReceipt value, but its self-digest is now stale.
        return CodexStructuredResult(
            payload=payload,
            receipt=replace(receipt, panel_view_digest="d" * 64),
        )

    with pytest.raises(TypedVisualTransportError, match="receipt is invalid"):
        propose_typed_visual(
            positives,
            negatives,
            catalog=catalog,
            protocol=protocol,
            model=MODEL,
            reasoning_effort=EFFORT,
            transport=transport,
        )


def _accepted_archive(
    tmp_path: Path,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
) -> dict[str, Any]:
    positives, negatives = _write_supports(tmp_path)

    def transport(prompt, paths, schema, **kwargs):
        del kwargs
        payload = _payload()
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(prompt, paths, schema, payload),
        )

    return propose_typed_visual(
        positives,
        negatives,
        catalog=catalog,
        protocol=protocol,
        transport=transport,
    ).to_data()


def _rejected_archive(
    tmp_path: Path,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
) -> dict[str, Any]:
    positives, negatives = _write_supports(tmp_path)

    def transport(prompt, paths, schema, **kwargs):
        del kwargs
        payload = _payload("no enclosed loops")
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(prompt, paths, schema, payload),
        )

    with pytest.raises(TypedVisualProposalRejected) as rejected:
        propose_typed_visual(
            positives,
            negatives,
            catalog=catalog,
            protocol=protocol,
            transport=transport,
        )
    return rejected.value.attempt.to_data()


@pytest.mark.parametrize(
    "case",
    (
        "extra_field",
        "missing_field",
        "changed_digest",
        "changed_proposal",
        "changed_payload",
        "changed_receipt",
        "changed_presentation",
        "changed_presentation_digest",
    ),
)
def test_accepted_cold_decoder_rejects_archive_tampering(
    tmp_path: Path,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
    case: str,
) -> None:
    data = deepcopy(_accepted_archive(tmp_path, catalog, protocol))
    if case == "extra_field":
        data["extra"] = None
    elif case == "missing_field":
        del data["proposal"]
    elif case == "changed_digest":
        data["result_digest"] = "d" * 64
    elif case == "changed_proposal":
        data["proposal"]["positive_description"] = "one compact component"
    elif case == "changed_payload":
        data["model_payload"]["positive_description"] = "one compact component"
    elif case == "changed_receipt":
        data["receipt"]["requested_model"] = "different-model"
    elif case == "changed_presentation":
        data["support_presentation"][0]["content_digest"] = "e" * 64
    else:
        data["support_presentation_digest"] = "f" * 64

    with pytest.raises(TypedVisualTransportError):
        TypedVisualTransportResult.from_data(
            data,
            catalog=catalog,
            protocol=protocol,
        )


@pytest.mark.parametrize(
    "case",
    (
        "extra_field",
        "missing_field",
        "changed_digest",
        "changed_parse_error",
        "accepted_payload",
        "changed_receipt",
        "changed_presentation",
    ),
)
def test_rejected_cold_decoder_replays_exact_parser_failure_and_tamper_checks(
    tmp_path: Path,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
    case: str,
) -> None:
    data = deepcopy(_rejected_archive(tmp_path, catalog, protocol))
    if case == "extra_field":
        data["proposal"] = None
    elif case == "missing_field":
        del data["parse_error"]
    elif case == "changed_digest":
        data["attempt_digest"] = "d" * 64
    elif case == "changed_parse_error":
        data["parse_error"]["reason"] = "different parser rejection"
    elif case == "accepted_payload":
        data["model_payload"] = _payload()
    elif case == "changed_receipt":
        data["receipt"]["requested_reasoning_effort"] = "high"
    else:
        data["support_presentation"][0]["byte_count"] += 1

    with pytest.raises(TypedVisualTransportError):
        RejectedTypedVisualProposalAttempt.from_data(
            data,
            catalog=catalog,
            protocol=protocol,
        )


def test_cold_decoder_rejects_wrong_expected_or_frozen_protocol(
    tmp_path: Path,
    catalog: RegisteredAtomCatalog,
    protocol: SoftScorerProtocol,
) -> None:
    data = _accepted_archive(tmp_path, catalog, protocol)
    with pytest.raises(TypedVisualTransportError, match="expected digest"):
        TypedVisualTransportResult.from_data(
            data,
            catalog=catalog,
            protocol=protocol,
            expected_digest="a" * 64,
        )

    different_protocol = replace(protocol, family_id="different-family")
    with pytest.raises(TypedVisualTransportError, match="another scorer protocol"):
        TypedVisualTransportResult.from_data(
            data,
            catalog=catalog,
            protocol=different_protocol,
        )


def test_current_structured_output_schema_stays_in_api_subset() -> None:
    schema = typed_visual_proposal_schema(DIRECT_VISUAL_ATOM_CATALOG)
    assert schema.get("type") == "object"
    assert "anyOf" not in schema
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False

    forbidden = {
        "oneOf",
        "uniqueItems",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "const",
        "not",
    }
    stack: list[tuple[str, Any]] = [("$", schema)]
    while stack:
        path, node = stack.pop()
        if isinstance(node, Mapping):
            assert forbidden.isdisjoint(node), (
                f"{path} uses unsupported schema keywords "
                f"{sorted(forbidden.intersection(node))}"
            )
            if node.get("type") == "object":
                properties = node.get("properties", {})
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(properties)
            stack.extend(
                (f"{path}.{key}", item) for key, item in node.items()
            )
        elif isinstance(node, list):
            stack.extend(
                (f"{path}[{index}]", item)
                for index, item in enumerate(node)
            )
