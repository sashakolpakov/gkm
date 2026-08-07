"""Receipt-attested text-only Codex ranker for prototype-scene survivors.

The callback accepts exactly the frozen survivor identifiers and the runner's
rank-input digest.  It sends no images and no support, class, or held-out
material.  A valid turn must return one exact permutation under a strict JSON
schema and an exact zero-image transport receipt.  The resulting
``PrototypeSceneRankResponse`` is Python-authoritative; Lean is absent and
removable from identity, replay, and ranking.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import bongard.transport as _transport_module
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    validate_codex_no_tools_attestation,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.prototype_scene_headless_runner import (
    PrototypeSceneHeadlessError,
    PrototypeSceneRankResponse,
)
from bongard.transport import (
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    REASONING_EFFORTS,
    TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexProposerFailure,
    CodexReceipt,
    CodexStructuredResult,
    run_codex_text_structured,
    validate_codex_strict_output_schema,
    validate_codex_text_receipt,
)


PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID = (
    "headless.codex.prototype-scene.text-ranker.v1"
)
PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_SCHEMA = (
    "gkm.bongard-prototype-scene-codex-ranker-protocol.v1"
)
MAX_SURVIVOR_COUNT = 512
MAX_PROMPT_UTF8_BYTES = 256_000

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# The finite support language's sole conjunction joins its opaque tags with
# ``+``; keep the ranker grammar exactly compatible with that canonical ID.
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class PrototypeSceneCodexRankerError(PrototypeSceneHeadlessError):
    """A rank input, text transport, payload, receipt, or pin is invalid."""


TextStructuredTransport = Callable[..., CodexStructuredResult]


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypeSceneCodexRankerError(f"{label} must be a sha256: address")
    return value


def _require_raw_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise PrototypeSceneCodexRankerError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_ranking": False,
        "lean_defines_identity": False,
        "lean_required_for_replay": False,
        "optional_secondary_checker_detachable": True,
    }


def prototype_scene_codex_ranker_authority_data() -> dict[str, object]:
    return _authority_data()


def prototype_scene_codex_ranker_source_digest() -> str:
    return _SOURCE_SHA256


def prototype_scene_codex_ranker_transport_source_digest() -> str:
    source = getattr(_transport_module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise PrototypeSceneCodexRankerError(
            "text transport source location is unavailable"
        )
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _freeze_survivors(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PrototypeSceneCodexRankerError(
            "survivor identifiers must be a finite sequence"
        )
    survivors = tuple(values)
    if (
        not 1 <= len(survivors) <= MAX_SURVIVOR_COUNT
        or any(
            not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None
            for item in survivors
        )
        or len(set(survivors)) != len(survivors)
    ):
        raise PrototypeSceneCodexRankerError(
            "survivor identifiers must be unique bounded identifiers"
        )
    return survivors


def prototype_scene_codex_ranker_prompt(
    survivor_candidate_ids: Sequence[str], rank_input_digest: str
) -> str:
    survivors = _freeze_survivors(survivor_candidate_ids)
    rank_digest = _require_address(rank_input_digest, "rank input digest")
    rendered = "\n".join(f"- {item}" for item in survivors)
    prompt = (
        "Arrange the frozen survivor identifiers into one preference order. "
        "Return every identifier exactly once. Use only the identifier text, "
        "the frozen input order, and the supplied digest. Do not invent or "
        "request unavailable evidence.\n\n"
        f"rank_input_digest: {rank_digest}\n"
        f"frozen_survivor_ids:\n{rendered}"
    )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_UTF8_BYTES:
        raise PrototypeSceneCodexRankerError("rank prompt exceeds its byte guard")
    lowered = prompt.lower()
    for forbidden in ("pixel", "label", "query"):
        if re.search(rf"\b{forbidden}s?\b", lowered):
            raise PrototypeSceneCodexRankerError(
                "rank prompt crosses the text-only survivor boundary"
            )
    return prompt


def prototype_scene_codex_ranker_output_schema(
    survivor_candidate_ids: Sequence[str],
) -> dict[str, object]:
    survivors = _freeze_survivors(survivor_candidate_ids)
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "ordered_candidate_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(survivors)},
            }
        },
        "required": ["ordered_candidate_ids"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def prototype_scene_codex_ranker_protocol_digest() -> str:
    return "sha256:" + canonical_digest(
        {
            "schema": PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_SCHEMA,
            "protocol_id": PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID,
            "ranker_source_sha256": prototype_scene_codex_ranker_source_digest(),
            "transport_source_sha256": (
                prototype_scene_codex_ranker_transport_source_digest()
            ),
            "transport_entrypoint": "run_codex_text_structured",
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            "input_fields": ["rank_input_digest", "frozen_survivor_ids"],
            "model_visible_pixels": False,
            "model_visible_class_assignments": False,
            "model_visible_held_out_material": False,
            "output_fields": ["ordered_candidate_ids"],
            "output_rule": "exact-survivor-permutation",
            "maximum_survivor_count": MAX_SURVIVOR_COUNT,
            "maximum_prompt_utf8_bytes": MAX_PROMPT_UTF8_BYTES,
            **_authority_data(),
        }
    )


def prototype_scene_codex_ranker_model_identity_digest(
    model: str, reasoning_effort: str
) -> str:
    if not isinstance(model, str) or _MODEL.fullmatch(model) is None:
        raise PrototypeSceneCodexRankerError("ranker model is invalid")
    if reasoning_effort not in REASONING_EFFORTS:
        raise PrototypeSceneCodexRankerError(
            "ranker reasoning effort is invalid"
        )
    return "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-prototype-scene-codex-model-request.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "identity_evidence_policy": (
                "receipt-reported-model-or-explicit-cli-model-flag"
            ),
        }
    )


def prototype_scene_codex_ranker_environment_digest(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    expected_cloud_policy_cache_binding: str,
    expected_transport_source_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> str:
    model_digest = prototype_scene_codex_ranker_model_identity_digest(
        model, reasoning_effort
    )
    launcher = _require_raw_sha256(
        expected_launcher_digest, "expected launcher digest"
    )
    transport_source = _require_raw_sha256(
        expected_transport_source_digest, "expected transport source digest"
    )
    if transport_source != prototype_scene_codex_ranker_transport_source_digest():
        raise PrototypeSceneCodexRankerError(
            "text transport source differs from external commitment"
        )
    policy = expected_cloud_policy_cache_binding
    if policy != "absent":
        _require_address(policy, "expected policy-cache binding")
    if not isinstance(model_catalog_snapshot, CodexModelCatalogSnapshot):
        raise PrototypeSceneCodexRankerError(
            "exact Codex model catalog snapshot is required"
        )
    try:
        attestation = validate_codex_no_tools_attestation(
            no_tools_attestation,
            expected_launcher_digest=launcher,
            expected_model_catalog_digest=model_catalog_snapshot.raw_digest,
            expected_cloud_policy_cache_binding=policy,
        )
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise PrototypeSceneCodexRankerError(
            "Codex no-tools runtime differs from its frozen attestation"
        ) from exc
    return "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-prototype-scene-codex-ranker-environment.v1",
            "model_identity_digest": model_digest,
            "expected_launcher_digest": launcher,
            "expected_cloud_policy_cache_binding": policy,
            "model_catalog_digest": model_catalog_snapshot.raw_digest,
            "no_tools_attestation_digest": attestation.attestation_digest,
            "ranker_source_sha256": prototype_scene_codex_ranker_source_digest(),
            "transport_source_sha256": transport_source,
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            **_authority_data(),
        }
    )


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise PrototypeSceneCodexRankerError(
            "rank payload must be a JSON object"
        )
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrototypeSceneCodexRankerError(
            "rank payload is not finite canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise PrototypeSceneCodexRankerError("rank payload must be an object")
    return decoded


def _parse_payload(
    payload: Mapping[str, Any], survivors: tuple[str, ...]
) -> tuple[str, ...]:
    if set(payload) != {"ordered_candidate_ids"}:
        raise PrototypeSceneCodexRankerError(
            "rank payload fields differ from schema"
        )
    values = payload["ordered_candidate_ids"]
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise PrototypeSceneCodexRankerError(
            "ordered candidate identifiers must be a JSON list"
        )
    ordered = tuple(values)
    if (
        len(ordered) != len(survivors)
        or len(set(ordered)) != len(ordered)
        or set(ordered) != set(survivors)
    ):
        raise PrototypeSceneCodexRankerError(
            "rank payload must be the exact survivor permutation"
        )
    return ordered


def _receipt_body(value: object) -> dict[str, Any]:
    if not isinstance(value, CodexReceipt):
        raise PrototypeSceneCodexRankerError(
            "text transport returned no CodexReceipt"
        )
    return value.to_dict()


def verify_prototype_scene_codex_rank_response(
    response: PrototypeSceneRankResponse,
    *,
    survivor_candidate_ids: Sequence[str],
    rank_input_digest: str,
    expected_response_digest: str,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    expected_cloud_policy_cache_binding: str,
    expected_transport_source_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> PrototypeSceneRankResponse:
    """Cold-verify one rank response against the frozen no-tools runtime."""

    if not isinstance(response, PrototypeSceneRankResponse):
        raise TypeError("response must be PrototypeSceneRankResponse")
    survivors = _freeze_survivors(survivor_candidate_ids)
    rank_digest = _require_address(rank_input_digest, "rank input digest")
    if response.record_digest != _require_address(
        expected_response_digest, "expected rank response digest"
    ):
        raise PrototypeSceneCodexRankerError(
            "rank response differs from external commitment"
        )
    response.assert_matches(
        expected_input_digest=rank_digest,
        survivor_candidate_ids=survivors,
    )
    expected_model_identity = prototype_scene_codex_ranker_model_identity_digest(
        model, reasoning_effort
    )
    expected_environment = prototype_scene_codex_ranker_environment_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        expected_cloud_policy_cache_binding=expected_cloud_policy_cache_binding,
        expected_transport_source_digest=expected_transport_source_digest,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if (
        response.ranker_protocol_id
        != PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID
        or response.ranker_protocol_digest
        != prototype_scene_codex_ranker_protocol_digest()
        or response.model_id != model
        or response.model_identity_digest != expected_model_identity
        or response.environment_digest != expected_environment
    ):
        raise PrototypeSceneCodexRankerError(
            "rank response protocol, model, or environment differs"
        )
    receipt_envelope = response.receipt
    if not isinstance(receipt_envelope, Mapping):
        raise PrototypeSceneCodexRankerError("rank receipt envelope is invalid")
    transport_receipt = receipt_envelope.get("transport_receipt")
    if not isinstance(transport_receipt, Mapping):
        raise PrototypeSceneCodexRankerError("rank transport receipt is invalid")
    prompt = prototype_scene_codex_ranker_prompt(survivors, rank_digest)
    schema = prototype_scene_codex_ranker_output_schema(survivors)
    payload = {"ordered_candidate_ids": list(response.ordered_candidate_ids)}
    try:
        validate_codex_text_receipt(transport_receipt, prompt, schema)
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise PrototypeSceneCodexRankerError(
            "text rank receipt does not bind the frozen input"
        ) from exc
    if (
        transport_receipt["requested_model"] != model
        or transport_receipt["requested_reasoning_effort"] != reasoning_effort
        or transport_receipt["codex_launcher_digest"]
        != expected_launcher_digest
        or transport_receipt["cloud_config_bundle_cache_binding"]
        != expected_cloud_policy_cache_binding
        or transport_receipt["model_catalog_digest"]
        != model_catalog_snapshot.raw_digest
        or transport_receipt["tool_surface_attestation_digest"]
        != no_tools_attestation.attestation_digest
        or transport_receipt["structured_output_digest"]
        != canonical_digest(payload)
    ):
        raise PrototypeSceneCodexRankerError(
            "text rank receipt model, environment, or payload differs"
        )
    if PrototypeSceneRankResponse.from_data(response.to_data()) != response:
        raise PrototypeSceneCodexRankerError(
            "rank response cold round trip differs"
        )
    return response


@dataclass(frozen=True, slots=True)
class PrototypeSceneCodexRanker:
    """Configured two-argument callback for ``PrototypeSceneHeadlessRunner``."""

    model: str
    expected_launcher_digest: str
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot
    expected_cloud_policy_cache_binding: str
    expected_transport_source_digest: str
    model_catalog_snapshot: CodexModelCatalogSnapshot
    no_tools_attestation: CodexNoToolsAttestation
    reasoning_effort: str = "medium"
    minutes: int = 15
    verbose: bool = False
    executable: str = "codex"
    transport: TextStructuredTransport = field(
        default=run_codex_text_structured,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        prototype_scene_codex_ranker_model_identity_digest(
            self.model, self.reasoning_effort
        )
        _require_raw_sha256(
            self.expected_launcher_digest, "expected launcher digest"
        )
        if not isinstance(self.cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot):
            raise PrototypeSceneCodexRankerError(
                "an exact cloud policy-cache snapshot is required"
            )
        if self.expected_cloud_policy_cache_binding != (
            self.cloud_policy_cache_snapshot.binding
        ):
            raise PrototypeSceneCodexRankerError(
                "policy-cache snapshot differs from external commitment"
            )
        prototype_scene_codex_ranker_environment_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            expected_cloud_policy_cache_binding=(
                self.expected_cloud_policy_cache_binding
            ),
            expected_transport_source_digest=self.expected_transport_source_digest,
            model_catalog_snapshot=self.model_catalog_snapshot,
            no_tools_attestation=self.no_tools_attestation,
        )
        if (
            isinstance(self.minutes, bool)
            or not isinstance(self.minutes, int)
            or not 1 <= self.minutes <= 120
        ):
            raise PrototypeSceneCodexRankerError(
                "ranker timeout minutes must lie in 1..120"
            )
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be bool")
        if not isinstance(self.executable, str) or not self.executable:
            raise PrototypeSceneCodexRankerError(
                "ranker executable must be nonempty"
            )
        if not callable(self.transport):
            raise TypeError("ranker transport must be callable")

    @property
    def protocol_digest(self) -> str:
        return prototype_scene_codex_ranker_protocol_digest()

    @property
    def model_identity_digest(self) -> str:
        return prototype_scene_codex_ranker_model_identity_digest(
            self.model, self.reasoning_effort
        )

    @property
    def environment_digest(self) -> str:
        return prototype_scene_codex_ranker_environment_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            expected_cloud_policy_cache_binding=(
                self.expected_cloud_policy_cache_binding
            ),
            expected_transport_source_digest=self.expected_transport_source_digest,
            model_catalog_snapshot=self.model_catalog_snapshot,
            no_tools_attestation=self.no_tools_attestation,
        )

    def _validate_receipt(
        self,
        receipt: Mapping[str, Any],
        prompt: str,
        schema: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        try:
            validate_codex_text_receipt(receipt, prompt, schema)
        except (CodexProposerFailure, TypeError, ValueError) as exc:
            raise PrototypeSceneCodexRankerError(
                "text rank receipt does not bind the frozen input"
            ) from exc
        if (
            receipt["requested_model"] != self.model
            or receipt["requested_reasoning_effort"] != self.reasoning_effort
            or receipt["codex_launcher_digest"]
            != self.expected_launcher_digest
            or receipt["cloud_config_bundle_cache_binding"]
            != self.expected_cloud_policy_cache_binding
            or receipt["structured_output_digest"]
            != canonical_digest(dict(payload))
            or receipt["model_catalog_digest"]
            != self.model_catalog_snapshot.raw_digest
            or receipt["tool_surface_attestation_digest"]
            != self.no_tools_attestation.attestation_digest
        ):
            raise PrototypeSceneCodexRankerError(
                "text rank receipt model, environment, or payload differs"
            )

    def __call__(
        self,
        survivor_candidate_ids: tuple[str, ...],
        rank_input_digest: str,
    ) -> PrototypeSceneRankResponse:
        survivors = _freeze_survivors(survivor_candidate_ids)
        rank_digest = _require_address(rank_input_digest, "rank input digest")
        prompt = prototype_scene_codex_ranker_prompt(survivors, rank_digest)
        schema = prototype_scene_codex_ranker_output_schema(survivors)
        try:
            result = self.transport(
                prompt,
                schema,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                minutes=self.minutes,
                verbose=self.verbose,
                executable=self.executable,
                cloud_policy_cache_snapshot=self.cloud_policy_cache_snapshot,
                model_catalog_snapshot=self.model_catalog_snapshot,
                expected_launcher_digest=self.expected_launcher_digest,
                tool_surface_attestation=self.no_tools_attestation,
                expected_tool_surface_attestation_digest=(
                    self.no_tools_attestation.attestation_digest
                ),
            )
        except Exception as exc:
            raise PrototypeSceneCodexRankerError(
                "text-only rank transport failed"
            ) from exc
        if not isinstance(result, CodexStructuredResult):
            raise PrototypeSceneCodexRankerError(
                "text transport returned the wrong result type"
            )
        payload = _canonical_payload(result.payload)
        ordered = _parse_payload(payload, survivors)
        receipt = _receipt_body(result.receipt)
        self._validate_receipt(receipt, prompt, schema, payload)
        response = PrototypeSceneRankResponse.seal(
            ordered_candidate_ids=ordered,
            ranker_protocol_id=PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID,
            ranker_protocol_digest=self.protocol_digest,
            model_id=self.model,
            model_identity_digest=self.model_identity_digest,
            environment_digest=self.environment_digest,
            input_digest=rank_digest,
            receipt=receipt,
        )
        response.assert_matches(
            expected_input_digest=rank_digest,
            survivor_candidate_ids=survivors,
        )
        return response

    def verify_response(
        self,
        response: PrototypeSceneRankResponse,
        *,
        survivor_candidate_ids: Sequence[str],
        rank_input_digest: str,
        expected_response_digest: str,
    ) -> PrototypeSceneRankResponse:
        """Cold-verify one response without a transport or model call."""

        return verify_prototype_scene_codex_rank_response(
            response,
            survivor_candidate_ids=survivor_candidate_ids,
            rank_input_digest=rank_input_digest,
            expected_response_digest=expected_response_digest,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            expected_cloud_policy_cache_binding=(
                self.expected_cloud_policy_cache_binding
            ),
            expected_transport_source_digest=self.expected_transport_source_digest,
            model_catalog_snapshot=self.model_catalog_snapshot,
            no_tools_attestation=self.no_tools_attestation,
        )


__all__ = [
    "MAX_PROMPT_UTF8_BYTES",
    "MAX_SURVIVOR_COUNT",
    "PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID",
    "PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_SCHEMA",
    "PrototypeSceneCodexRanker",
    "PrototypeSceneCodexRankerError",
    "TextStructuredTransport",
    "prototype_scene_codex_ranker_authority_data",
    "prototype_scene_codex_ranker_environment_digest",
    "prototype_scene_codex_ranker_model_identity_digest",
    "prototype_scene_codex_ranker_output_schema",
    "prototype_scene_codex_ranker_prompt",
    "prototype_scene_codex_ranker_protocol_digest",
    "prototype_scene_codex_ranker_source_digest",
    "prototype_scene_codex_ranker_transport_source_digest",
    "verify_prototype_scene_codex_rank_response",
]
