"""Pre-support policy identity for the typed visual-semantic pipeline.

The policy binds every verifier-owned choice that may affect a proposal or a
panel judgment before labelled support pixels are released.  It deliberately
does not name an execution backend: the closed predicate and evidence records
have one serialized meaning, with pure Python as the reference implementation
and any proof assistant limited to an optional independent checker.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from bongard.artifacts import canonical_digest


VISUAL_SEMANTIC_POLICY_SCHEMA = "bongard.visual-semantic-policy/v1"
SCENARIO_EVALUATION = "complete_direct_conjunction_inside_each_scenario_v1"
SCENARIO_CONSENSUS = "all_true_present_all_false_absent_else_indeterminate_v1"
FORMULA_GRAMMAR = "positive_conjunction_direct_composite_plus_optional_soft_v1"
REFERENCE_SEMANTICS = "closed_positive_four_disposition_ir/v1"

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class VisualSemanticPolicyError(ValueError):
    """A pre-support visual-semantic policy is malformed or inconsistent."""


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise VisualSemanticPolicyError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise VisualSemanticPolicyError(f"invalid {label} {value!r}")
    return value


def _version(value: object, label: str) -> str:
    if not isinstance(value, str) or _VERSION.fullmatch(value) is None:
        raise VisualSemanticPolicyError(f"invalid {label} {value!r}")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VisualSemanticPolicyError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class VisualSemanticPolicy:
    """Content-addressed policy fixed before the proposer sees support.

    The prospective scorer protocol, fitted family, and development-manifest
    identities cover dynamic task-local phrases without a hash cycle.  A newly
    proposed phrase is therefore an input to one already calibrated procedure,
    never an excuse to fit a fresh threshold on the same twelve support labels.
    """

    witness_extractor_id: str
    witness_extractor_version: str
    witness_extractor_digest: str
    witness_catalog_digest: str
    scenario_ids: tuple[str, ...]
    direct_predicate_catalog_digest: str
    proposal_protocol_digest: str
    soft_scorer_protocol_digest: str
    soft_scorer_family_digest: str
    soft_family_development_manifest_digest: str
    max_direct_atoms: int = 3
    max_soft_claims: int = 1
    max_soft_cues: int = 4

    def __post_init__(self) -> None:
        _identifier(self.witness_extractor_id, "witness_extractor_id")
        _version(self.witness_extractor_version, "witness_extractor_version")
        for name in (
            "witness_extractor_digest",
            "witness_catalog_digest",
            "direct_predicate_catalog_digest",
            "proposal_protocol_digest",
            "soft_scorer_protocol_digest",
            "soft_scorer_family_digest",
            "soft_family_development_manifest_digest",
        ):
            _digest(getattr(self, name), name)
        if (
            not isinstance(self.scenario_ids, tuple)
            or len(self.scenario_ids) < 2
            or any(not isinstance(item, str) for item in self.scenario_ids)
        ):
            raise VisualSemanticPolicyError(
                "scenario_ids must be an immutable tuple with at least two entries"
            )
        for item in self.scenario_ids:
            _identifier(item, "scenario id")
        if tuple(sorted(self.scenario_ids)) != self.scenario_ids:
            raise VisualSemanticPolicyError("scenario_ids must be sorted")
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise VisualSemanticPolicyError("scenario_ids must be unique")
        _positive_int(self.max_direct_atoms, "max_direct_atoms")
        _positive_int(self.max_soft_claims, "max_soft_claims")
        _positive_int(self.max_soft_cues, "max_soft_cues")
        if self.max_direct_atoms != 3:
            raise VisualSemanticPolicyError(
                "visual-semantic v1 permits exactly three direct atom slots"
            )
        if self.max_soft_claims != 1:
            raise VisualSemanticPolicyError(
                "visual-semantic v1 permits exactly one soft claim slot"
            )
        if self.max_soft_cues > 6:
            raise VisualSemanticPolicyError(
                "visual-semantic v1 permits at most six affirmative soft cues"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": VISUAL_SEMANTIC_POLICY_SCHEMA,
            "witness_extractor": {
                "id": self.witness_extractor_id,
                "version": self.witness_extractor_version,
                "artifact_digest": self.witness_extractor_digest,
                "catalog_digest": self.witness_catalog_digest,
                "scenario_ids": list(self.scenario_ids),
                "input_contract": (
                    "panel_bytes_only_no_task_candidate_side_or_role_context_v1"
                ),
            },
            "direct_predicate_catalog_digest": self.direct_predicate_catalog_digest,
            "proposal_protocol_digest": self.proposal_protocol_digest,
            "soft_scorer": {
                "protocol_digest": self.soft_scorer_protocol_digest,
                "family_digest": self.soft_scorer_family_digest,
                "development_manifest_digest": (
                    self.soft_family_development_manifest_digest
                ),
                "calibration_recomputed_from_manifest": True,
                "task_local_claim_is_input": True,
                "task_local_threshold_fitting": False,
                "model_emits_final_boolean": False,
                "model_emits_certified_absence": False,
            },
            "limits": {
                "max_direct_atoms": self.max_direct_atoms,
                "max_soft_claims": self.max_soft_claims,
                "max_soft_cues": self.max_soft_cues,
            },
            "formula_grammar": FORMULA_GRAMMAR,
            "scenario_evaluation": SCENARIO_EVALUATION,
            "scenario_consensus": SCENARIO_CONSENSUS,
            "reference_semantics": REFERENCE_SEMANTICS,
            "polarity_flip_allowed": False,
            "negation_node_available": False,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "VisualSemanticPolicy":
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "witness_extractor",
            "direct_predicate_catalog_digest",
            "proposal_protocol_digest",
            "soft_scorer",
            "limits",
            "formula_grammar",
            "scenario_evaluation",
            "scenario_consensus",
            "reference_semantics",
            "polarity_flip_allowed",
            "negation_node_available",
        }:
            raise VisualSemanticPolicyError(
                "visual-semantic policy fields differ from the closed schema"
            )
        if value["schema"] != VISUAL_SEMANTIC_POLICY_SCHEMA:
            raise VisualSemanticPolicyError("unsupported visual-semantic policy schema")
        witness = value["witness_extractor"]
        soft = value["soft_scorer"]
        limits = value["limits"]
        if not isinstance(witness, Mapping) or set(witness) != {
            "id",
            "version",
            "artifact_digest",
            "catalog_digest",
            "scenario_ids",
            "input_contract",
        }:
            raise VisualSemanticPolicyError("witness extractor policy is malformed")
        if not isinstance(soft, Mapping) or set(soft) != {
            "protocol_digest",
            "family_digest",
            "development_manifest_digest",
            "calibration_recomputed_from_manifest",
            "task_local_claim_is_input",
            "task_local_threshold_fitting",
            "model_emits_final_boolean",
            "model_emits_certified_absence",
        }:
            raise VisualSemanticPolicyError("soft scorer policy is malformed")
        if not isinstance(limits, Mapping) or set(limits) != {
            "max_direct_atoms",
            "max_soft_claims",
            "max_soft_cues",
        }:
            raise VisualSemanticPolicyError("visual-semantic limits are malformed")
        scenarios = witness["scenario_ids"]
        if not isinstance(scenarios, list) or any(
            not isinstance(item, str) for item in scenarios
        ):
            raise VisualSemanticPolicyError("scenario_ids must be a list of strings")
        fixed_values = {
            "input_contract": (
                witness["input_contract"],
                "panel_bytes_only_no_task_candidate_side_or_role_context_v1",
            ),
            "task_local_claim_is_input": (
                soft["task_local_claim_is_input"],
                True,
            ),
            "calibration_recomputed_from_manifest": (
                soft["calibration_recomputed_from_manifest"],
                True,
            ),
            "task_local_threshold_fitting": (
                soft["task_local_threshold_fitting"],
                False,
            ),
            "model_emits_final_boolean": (
                soft["model_emits_final_boolean"],
                False,
            ),
            "model_emits_certified_absence": (
                soft["model_emits_certified_absence"],
                False,
            ),
            "formula_grammar": (value["formula_grammar"], FORMULA_GRAMMAR),
            "scenario_evaluation": (
                value["scenario_evaluation"],
                SCENARIO_EVALUATION,
            ),
            "scenario_consensus": (value["scenario_consensus"], SCENARIO_CONSENSUS),
            "reference_semantics": (value["reference_semantics"], REFERENCE_SEMANTICS),
            "polarity_flip_allowed": (value["polarity_flip_allowed"], False),
            "negation_node_available": (value["negation_node_available"], False),
        }
        for name, (actual, expected) in fixed_values.items():
            if actual != expected or type(actual) is not type(expected):
                raise VisualSemanticPolicyError(f"policy changed fixed {name}")
        return cls(
            witness_extractor_id=witness["id"],
            witness_extractor_version=witness["version"],
            witness_extractor_digest=witness["artifact_digest"],
            witness_catalog_digest=witness["catalog_digest"],
            scenario_ids=tuple(scenarios),
            direct_predicate_catalog_digest=value[
                "direct_predicate_catalog_digest"
            ],
            proposal_protocol_digest=value["proposal_protocol_digest"],
            soft_scorer_protocol_digest=soft["protocol_digest"],
            soft_scorer_family_digest=soft["family_digest"],
            soft_family_development_manifest_digest=soft[
                "development_manifest_digest"
            ],
            max_direct_atoms=limits["max_direct_atoms"],
            max_soft_claims=limits["max_soft_claims"],
            max_soft_cues=limits["max_soft_cues"],
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


__all__ = [
    "FORMULA_GRAMMAR",
    "REFERENCE_SEMANTICS",
    "SCENARIO_CONSENSUS",
    "SCENARIO_EVALUATION",
    "VISUAL_SEMANTIC_POLICY_SCHEMA",
    "VisualSemanticPolicy",
    "VisualSemanticPolicyError",
]
