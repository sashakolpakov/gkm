"""Acyclic identity construction for the visual-semantic benchmark path.

The prospective soft-scorer protocol is built before any development score or
label exists.  Development records point to that protocol, a fitted family
points to both, and the final visual-semantic policy points to the completed
family.  No earlier object contains a later digest, so construction never asks
for a hash fixed point.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from bongard.artifacts import canonical_digest
from bongard.benchmark import SupportGatePolicy
from bongard.blind_soft_transport import (
    BLIND_SOFT_DECODER_ID,
    BLIND_SOFT_PROMPT_TEMPLATE_ID,
    blind_soft_decoder_digest,
    blind_soft_prompt_template_digest,
)
from bongard.semantic_policy import VisualSemanticPolicy
from bongard.soft_predicates import (
    SoftScorerFamily,
    SoftScorerProtocol,
    blind_soft_score_output_schema,
)
from bongard.transport import validate_codex_strict_output_schema
from bongard.typed_visual_proposal import (
    MAX_DETERMINISTIC_ATOMS,
    MAX_SOFT_CUES,
    TYPED_VISUAL_PROPOSER_GRAMMAR_ID,
    TYPED_VISUAL_PROPOSER_PROMPT_ID,
    typed_visual_proposal_grammar_digest,
    typed_visual_proposal_prompt_digest,
)
import bongard.typed_visual_transport as _typed_visual_transport
from bongard.typed_visual_transport import (
    REJECTED_TYPED_VISUAL_PROPOSAL_SCHEMA_VERSION,
    TYPED_VISUAL_TRANSPORT_RESULT_SCHEMA_VERSION,
)
from bongard.visual_predicate_catalog import DIRECT_VISUAL_ATOM_CATALOG
from bongard.visual_witness_summaries import (
    visual_joint_soft_witness_interface_digest,
)
from bongard.visual_witness_bundle import (
    VISUAL_WITNESS_BUNDLE_EXTRACTOR_ID,
    VISUAL_WITNESS_BUNDLE_VERSION,
    visual_witness_bundle_catalog_digest,
    visual_witness_bundle_extractor_digest,
)
from bongard.visual_witnesses import (
    VISUAL_WITNESS_SCENARIO_IDS,
)


VISUAL_SEMANTIC_SOFT_FAMILY_ID = "visual-semantic-positive-cues"
VISUAL_SEMANTIC_SOFT_PROTOCOL_VERSION = "1"
VISUAL_SOFT_WITNESS_INTERFACE_ID = "visual-soft-witness-interface-v2"
VISUAL_SEMANTIC_PROPOSAL_PROCEDURE_ID = "typed-visual-proposal-procedure-v1"
VISUAL_SEMANTIC_PROPOSAL_PROCEDURE_SCHEMA = (
    "gkm.bongard-visual-semantic-proposal-procedure.v1"
)

_ORDINAL_MAP = (
    ("supported", 1.0),
    ("ambiguous", 0.5),
    ("unsupported", 0.0),
)


class SemanticProtocolError(ValueError):
    """A semantic protocol cannot be built from the supplied dependencies."""


class SemanticProtocolIntegrityError(SemanticProtocolError):
    """A fitted family or protocol differs from its prospective identity."""


def _proposal_transport_source_digest() -> str:
    source = getattr(_typed_visual_transport, "__file__", None)
    if not isinstance(source, str) or not source:
        raise SemanticProtocolError("typed visual transport source is unavailable")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _visual_semantic_support_gate_identity() -> tuple[str, str]:
    gate = SupportGatePolicy.visual_semantic()
    return (
        f"{gate.mode.value}@{gate.version}",
        canonical_digest(gate.to_data()),
    )


def build_prospective_soft_scorer_protocol(
    *,
    proposer_model_id: str,
    proposer_reasoning_effort: str,
    scorer_model_id: str,
    scorer_reasoning_effort: str,
    score_bin_edges: tuple[float, ...],
    affirmative_boundary: float,
    confidence_level: float,
    minimum_clusters_per_bin: int,
    family_id: str = VISUAL_SEMANTIC_SOFT_FAMILY_ID,
    version: str = VISUAL_SEMANTIC_SOFT_PROTOCOL_VERSION,
) -> SoftScorerProtocol:
    """Freeze the complete dynamic-claim scorer before development exists."""

    catalog = DIRECT_VISUAL_ATOM_CATALOG
    gate_id, gate_digest = _visual_semantic_support_gate_identity()
    protocol = SoftScorerProtocol(
        family_id=family_id,
        version=version,
        proposer_grammar_id=TYPED_VISUAL_PROPOSER_GRAMMAR_ID,
        proposer_grammar_digest=typed_visual_proposal_grammar_digest(catalog),
        proposer_model_id=proposer_model_id,
        proposer_reasoning_effort=proposer_reasoning_effort,
        proposer_prompt_id=TYPED_VISUAL_PROPOSER_PROMPT_ID,
        proposer_prompt_digest=typed_visual_proposal_prompt_digest(catalog),
        scorer_model_id=scorer_model_id,
        scorer_reasoning_effort=scorer_reasoning_effort,
        scorer_prompt_template_id=BLIND_SOFT_PROMPT_TEMPLATE_ID,
        scorer_prompt_template_digest=blind_soft_prompt_template_digest(),
        scorer_decoder_id=BLIND_SOFT_DECODER_ID,
        scorer_decoder_digest=blind_soft_decoder_digest(),
        ordinal_map=_ORDINAL_MAP,
        aggregation="min",
        witness_extractor_id=VISUAL_SOFT_WITNESS_INTERFACE_ID,
        witness_extractor_digest=visual_joint_soft_witness_interface_digest(),
        support_gate_id=gate_id,
        support_gate_digest=gate_digest,
        score_bin_edges=score_bin_edges,
        affirmative_boundary=affirmative_boundary,
        confidence_level=confidence_level,
        minimum_clusters_per_bin=minimum_clusters_per_bin,
    )
    _validate_current_protocol_dependencies(protocol)
    return protocol


def _validate_current_protocol_dependencies(protocol: SoftScorerProtocol) -> None:
    if not isinstance(protocol, SoftScorerProtocol):
        raise TypeError("prospective_protocol must be a SoftScorerProtocol")
    protocol.assert_untampered()
    # Static canary for the dynamic scorer schema.  Cue/witness values alter
    # only enums, so this catches dialect-incompatible construction before a
    # campaign selects or exposes any task.
    validate_codex_strict_output_schema(
        blind_soft_score_output_schema(("cue-00",), ("witness:00",))
    )
    catalog = DIRECT_VISUAL_ATOM_CATALOG
    gate_id, gate_digest = _visual_semantic_support_gate_identity()
    expected: dict[str, Any] = {
        "proposer_grammar_id": TYPED_VISUAL_PROPOSER_GRAMMAR_ID,
        "proposer_grammar_digest": typed_visual_proposal_grammar_digest(catalog),
        "proposer_prompt_id": TYPED_VISUAL_PROPOSER_PROMPT_ID,
        "proposer_prompt_digest": typed_visual_proposal_prompt_digest(catalog),
        "scorer_prompt_template_id": BLIND_SOFT_PROMPT_TEMPLATE_ID,
        "scorer_prompt_template_digest": blind_soft_prompt_template_digest(),
        "scorer_decoder_id": BLIND_SOFT_DECODER_ID,
        "scorer_decoder_digest": blind_soft_decoder_digest(),
        "witness_extractor_id": VISUAL_SOFT_WITNESS_INTERFACE_ID,
        "witness_extractor_digest": visual_joint_soft_witness_interface_digest(),
        "support_gate_id": gate_id,
        "support_gate_digest": gate_digest,
    }
    mismatches = tuple(
        name for name, value in expected.items() if getattr(protocol, name) != value
    )
    if mismatches:
        raise SemanticProtocolIntegrityError(
            "soft scorer protocol differs from current frozen dependencies: "
            + ", ".join(mismatches)
        )


def visual_semantic_proposal_procedure_data(
    prospective_protocol: SoftScorerProtocol,
) -> dict[str, object]:
    """Describe the proposal procedure without referring to family or policy.

    The static prompt deliberately cannot contain the protocol digest.  After
    the model turn, verifier-owned parsing injects that already-existing digest
    into any soft claim.  This forward edge is explicit here and is the reason
    the identity graph has no fixed point.
    """

    _validate_current_protocol_dependencies(prospective_protocol)
    catalog = DIRECT_VISUAL_ATOM_CATALOG
    return {
        "schema": VISUAL_SEMANTIC_PROPOSAL_PROCEDURE_SCHEMA,
        "procedure_id": VISUAL_SEMANTIC_PROPOSAL_PROCEDURE_ID,
        "direct_catalog_digest": catalog.digest,
        "typed_proposer": {
            "grammar_id": prospective_protocol.proposer_grammar_id,
            "grammar_digest": prospective_protocol.proposer_grammar_digest,
            "prompt_id": prospective_protocol.proposer_prompt_id,
            "prompt_digest": prospective_protocol.proposer_prompt_digest,
            "model_id": prospective_protocol.proposer_model_id,
            "reasoning_effort": (
                prospective_protocol.proposer_reasoning_effort
            ),
        },
        "transport": {
            "accepted_result_schema": TYPED_VISUAL_TRANSPORT_RESULT_SCHEMA_VERSION,
            "rejected_result_schema": (
                REJECTED_TYPED_VISUAL_PROPOSAL_SCHEMA_VERSION
            ),
            "source_digest": _proposal_transport_source_digest(),
            "support_presentation": "exact_canonical_pos_0_to_neg_5_png_v1",
        },
        "scorer_protocol_digest": prospective_protocol.digest(),
        "protocol_binding_stage": (
            "verifier_parser_injects_after_model_turn_before_proposal_freeze_v1"
        ),
        "family_digest_is_input": False,
        "visual_semantic_policy_digest_is_input": False,
    }


def visual_semantic_proposal_procedure_digest(
    prospective_protocol: SoftScorerProtocol,
) -> str:
    """Return the acyclic, source-bound typed proposal-procedure identity."""

    return canonical_digest(
        visual_semantic_proposal_procedure_data(prospective_protocol)
    )


def build_visual_semantic_policy(
    family: SoftScorerFamily,
    *,
    prospective_protocol: SoftScorerProtocol,
) -> VisualSemanticPolicy:
    """Bind a fitted family to its prospectively committed protocol and vision."""

    if not isinstance(family, SoftScorerFamily):
        raise TypeError("family must be a SoftScorerFamily")
    _validate_current_protocol_dependencies(prospective_protocol)
    prospective_digest = prospective_protocol.digest()
    family.assert_untampered()
    family.verify_calibration()
    if (
        family.protocol_digest != prospective_digest
        or family.protocol.to_data() != prospective_protocol.to_data()
    ):
        raise SemanticProtocolIntegrityError(
            "fitted family does not belong to the prospective scorer protocol"
        )

    return VisualSemanticPolicy(
        witness_extractor_id=VISUAL_WITNESS_BUNDLE_EXTRACTOR_ID,
        witness_extractor_version=VISUAL_WITNESS_BUNDLE_VERSION,
        witness_extractor_digest=visual_witness_bundle_extractor_digest(),
        witness_catalog_digest=visual_witness_bundle_catalog_digest(),
        scenario_ids=VISUAL_WITNESS_SCENARIO_IDS,
        direct_predicate_catalog_digest=DIRECT_VISUAL_ATOM_CATALOG.digest,
        proposal_protocol_digest=visual_semantic_proposal_procedure_digest(
            prospective_protocol
        ),
        soft_scorer_protocol_digest=prospective_digest,
        soft_scorer_family_digest=family.digest(),
        soft_family_development_manifest_digest=(
            family.development_manifest_digest
        ),
        max_direct_atoms=MAX_DETERMINISTIC_ATOMS,
        max_soft_claims=1,
        max_soft_cues=MAX_SOFT_CUES,
    )


__all__ = [
    "SemanticProtocolError",
    "SemanticProtocolIntegrityError",
    "VISUAL_SEMANTIC_PROPOSAL_PROCEDURE_ID",
    "VISUAL_SEMANTIC_PROPOSAL_PROCEDURE_SCHEMA",
    "VISUAL_SEMANTIC_SOFT_FAMILY_ID",
    "VISUAL_SEMANTIC_SOFT_PROTOCOL_VERSION",
    "VISUAL_SOFT_WITNESS_INTERFACE_ID",
    "build_prospective_soft_scorer_protocol",
    "build_visual_semantic_policy",
    "visual_semantic_proposal_procedure_data",
    "visual_semantic_proposal_procedure_digest",
]
