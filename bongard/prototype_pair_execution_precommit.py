"""Metadata-only execution precommit for one prototype-pair drill.

The precommit consumes an already authenticated cohort plan and explicit
content identities.  It does not resolve panel paths, open panel bytes, call a
model, or mutate an exposure ledger.  It freezes the 6+6 support / 1+1 query
split and seals query identities before any durable candidate can exist.

Python is the sole predicate and decision authority.  Lean may be attached as
an optional secondary checker, but removing it cannot change artifact identity,
selection, execution, replay, or the decision.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.prototype_pair_cohort import (
    OPAQUE_TAG_IDS,
    PrototypePairCohortPlan,
)
from bongard.prototype_scene_calibration import (
    PrototypeSceneTagThreshold,
    threshold_commitment,
)


PRECOMMIT_SCHEMA = "gkm.bongard-prototype-pair-execution-precommit.v3"
IDENTITIES_SCHEMA = "gkm.bongard-prototype-pair-execution-identities.v3"
ROLE_SCHEMA = "gkm.bongard-prototype-pair-execution-panel-role.v1"
ALGORITHM_ID = "bongard.prototype-pair/execution-precommit-campaign-v3"
SPLIT_RULE_ID = "per-side-sha256-rank-lowest-query-remaining-support-v1"

REQUIRED_RUNTIME_SOURCE_ROLES = frozenset(
    {
        "observer",
        "calibration",
        "predicate",
        "version-space",
        "runner",
        "runtime-adapter",
        "ranker",
        "campaign",
        "campaign-cli",
        "campaign-store",
        "transport",
        "official-panel-archive",
        "precommit",
        "canonical",
        "exposure",
        "release",
        "cohort",
        "cohorts",
        "corpus",
        "historical-exposure",
        "image-audit",
        "cluster-binomial",
        "python-authority",
        "grounded-compat",
        "package-init",
        "source-snapshot",
    }
)

PHASE_ORDER = (
    "execution_precommit_persisted",
    "exposure_successor_persisted",
    "six_prototype_pixels_released",
    "prototype_description_observed",
    "calibration_plan_frozen",
    "twenty_eight_calibration_scenes_released_and_observed",
    "calibration_family_and_predicate_library_frozen",
    "twelve_support_scenes_released_and_observed",
    "support_version_space_constructed",
    "headless_codex_candidate_ranked",
    "durable_python_candidate_frozen",
    "two_query_scenes_released_and_observed",
    "python_query_evaluation",
    "model_free_tamper_detecting_replay",
)

# Each row states the exact number of model calls in both branches.  All
# non-model phases are explicit zeroes rather than an implicit unlimited gap.
CALL_BUDGETS = (
    ("execution_precommit_persisted", "none", "always", 0, 0),
    ("exposure_successor_persisted", "none", "always", 0, 0),
    ("six_prototype_pixels_released", "none", "always", 0, 0),
    (
        "prototype_description_observed",
        "prototype_scene_observer",
        "six_prototypes_released",
        1,
        0,
    ),
    ("calibration_plan_frozen", "none", "always", 0, 0),
    (
        "twenty_eight_calibration_scenes_released_and_observed",
        "prototype_scene_observer",
        "calibration_plan_frozen",
        28,
        0,
    ),
    (
        "calibration_family_and_predicate_library_frozen",
        "none",
        "always",
        0,
        0,
    ),
    (
        "twelve_support_scenes_released_and_observed",
        "prototype_scene_observer",
        "calibration_family_certified",
        12,
        0,
    ),
    ("support_version_space_constructed", "none", "always", 0, 0),
    (
        "headless_codex_candidate_ranked",
        "headless_codex_ranker",
        "verified_survivor_set_nonempty",
        1,
        0,
    ),
    ("durable_python_candidate_frozen", "none", "always", 0, 0),
    (
        "two_query_scenes_released_and_observed",
        "prototype_scene_observer",
        "durable_python_candidate_frozen",
        2,
        0,
    ),
    ("python_query_evaluation", "none", "always", 0, 0),
    ("model_free_tamper_detecting_replay", "none", "always", 0, 0),
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")

PRECOMMIT_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class PrototypePairExecutionPrecommitError(ValueError):
    """An identity, split, authority, or replay invariant failed."""


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypePairExecutionPrecommitError(
            f"{label} must be a sha256: address"
        )
    return value


def _require_raw_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise PrototypePairExecutionPrecommitError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PrototypePairExecutionPrecommitError(
            f"{label} must be a bounded identifier"
        )
    return value


def _text(value: object, label: str, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8", errors="strict")) > maximum
    ):
        raise PrototypePairExecutionPrecommitError(f"{label} must be bounded text")
    return value


def _object(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != fields
    ):
        raise PrototypePairExecutionPrecommitError(f"{label} fields differ")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PrototypePairExecutionPrecommitError(f"{label} must be a list")
    return value


def _verify_record_digest(raw: Mapping[str, Any], label: str) -> None:
    expected = _require_address(raw["record_digest"], f"{label} record digest")
    body = {key: value for key, value in raw.items() if key != "record_digest"}
    if expected != _address(body):
        raise PrototypePairExecutionPrecommitError(f"{label} digest differs")


def _budget_data() -> list[dict[str, object]]:
    return [
        {
            "phase_id": phase_id,
            "actor": actor,
            "condition": condition,
            "calls_when_condition_true": calls_true,
            "calls_when_condition_false": calls_false,
        }
        for phase_id, actor, condition, calls_true, calls_false in CALL_BUDGETS
    ]


def execution_precommit_algorithm_digest() -> str:
    """Bind source bytes and the complete split/execution policy."""

    return _address(
        {
            "schema": "gkm.bongard-prototype-pair-execution-algorithm.v3",
            "source_sha256": PRECOMMIT_SOURCE_SHA256,
            "algorithm_id": ALGORITHM_ID,
            "precommit_schema": PRECOMMIT_SCHEMA,
            "identities_schema": IDENTITIES_SCHEMA,
            "role_schema": ROLE_SCHEMA,
            "split_rule_id": SPLIT_RULE_ID,
            "split": {"per_source_side": 7, "support": 6, "query": 1},
            "selection_inputs": [
                "execution_precommit_source_sha256",
                "execution_precommit_algorithm_digest",
                "cohort_plan_digest",
                "cohort_planner_algorithm_digest",
                "selection_seed_digest",
                "release_descriptor_digest",
                "corpus_manifest_digest",
                "opaque_side_id",
                "panel_id",
            ],
            "selection_excludes": ["pixels", "action_program_json"],
            "execution_configuration_binding": (
                "identity-root-only-and-does-not-affect-support-query-selection"
            ),
            "required_runtime_source_roles": sorted(
                REQUIRED_RUNTIME_SOURCE_ROLES
            ),
            "phase_order": list(PHASE_ORDER),
            "call_budgets": _budget_data(),
            "branch_model_call_totals": {
                "complete_candidate_and_query": 44,
                "no_verified_support_survivor": 41,
                "calibration_family_rejected": 29,
                "ranker_error": 42,
            },
            "query_invariant": (
                "query-role-identities-sealed-in-precommit-and-query-pixels-"
                "released-only-after-durable-python-candidate-freeze"
            ),
            "claim_scope": (
                "exact-unused-train-semantically-reused-targeted-engineering"
            ),
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "python_is_canonical_authority": True,
            "lean_required": False,
            "lean_removable": True,
            "lean_affects_identity_selection_execution_or_decision": False,
        }
    )


@dataclass(frozen=True, slots=True)
class PrototypePairExecutionIdentities:
    """Caller-supplied identities; this class chooses none of their values."""

    exposure_predecessor_digest: str
    execution_configuration_digest: str
    thresholds: tuple[PrototypeSceneTagThreshold, PrototypeSceneTagThreshold]
    threshold_commitment: str
    calibration_algorithm_digest: str
    observer_protocol_id: str
    observer_description_protocol_digest: str
    observer_scoring_protocol_digest: str
    observer_environment_digest: str
    observer_model_id: str
    observer_reasoning_effort: str
    observer_model_identity_digest: str
    ranker_model_id: str
    ranker_reasoning_effort: str
    ranker_model_identity_digest: str
    runner_protocol_id: str
    runner_algorithm_digest: str
    codex_cli_version: str
    codex_launcher_sha256: str
    cloud_policy_cache_binding: str
    python_runtime_id: str
    python_runtime_identity_digest: str
    runtime_source_digests: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in (
            "exposure_predecessor_digest",
            "execution_configuration_digest",
            "threshold_commitment",
            "calibration_algorithm_digest",
            "runner_algorithm_digest",
        ):
            _require_address(getattr(self, name), name)
        if self.cloud_policy_cache_binding != "absent":
            _require_address(
                self.cloud_policy_cache_binding, "cloud_policy_cache_binding"
            )
        if tuple(item.tag_id for item in self.thresholds) != OPAQUE_TAG_IDS:
            raise PrototypePairExecutionPrecommitError(
                "thresholds must cover the two opaque tags in frozen order"
            )
        if self.threshold_commitment != threshold_commitment(self.thresholds):
            raise PrototypePairExecutionPrecommitError(
                "threshold commitment differs from exact threshold records"
            )
        _identifier(self.observer_protocol_id, "observer protocol ID")
        _identifier(self.runner_protocol_id, "runner protocol ID")
        _text(self.observer_model_id, "observer model ID")
        _identifier(self.observer_reasoning_effort, "observer reasoning effort")
        _text(self.ranker_model_id, "ranker model ID")
        _identifier(self.ranker_reasoning_effort, "ranker reasoning effort")
        _text(self.codex_cli_version, "Codex CLI version")
        _text(self.python_runtime_id, "Python runtime ID")
        for name in (
            "observer_description_protocol_digest",
            "observer_scoring_protocol_digest",
            "observer_environment_digest",
            "observer_model_identity_digest",
            "ranker_model_identity_digest",
            "codex_launcher_sha256",
            "python_runtime_identity_digest",
        ):
            _require_raw_sha(getattr(self, name), name)
        if (
            not isinstance(self.runtime_source_digests, tuple)
            or any(
                not isinstance(row, tuple) or len(row) != 2
                for row in self.runtime_source_digests
            )
        ):
            raise PrototypePairExecutionPrecommitError(
                "runtime source digests must be a tuple of pairs"
            )
        roles = tuple(role for role, _digest in self.runtime_source_digests)
        if roles != tuple(sorted(set(roles))):
            raise PrototypePairExecutionPrecommitError(
                "runtime source roles must be unique and sorted"
            )
        for role, digest in self.runtime_source_digests:
            _identifier(role, "runtime source role")
            _require_raw_sha(digest, f"runtime source {role}")
        missing = REQUIRED_RUNTIME_SOURCE_ROLES - set(roles)
        if missing:
            raise PrototypePairExecutionPrecommitError(
                f"runtime source map is missing {sorted(missing)}"
            )

    @classmethod
    def create(
        cls,
        *,
        runtime_source_digests: Mapping[str, str],
        **kwargs: Any,
    ) -> "PrototypePairExecutionIdentities":
        if not isinstance(runtime_source_digests, Mapping) or any(
            not isinstance(key, str) for key in runtime_source_digests
        ):
            raise PrototypePairExecutionPrecommitError(
                "runtime source digests must be an object"
            )
        return cls(
            runtime_source_digests=tuple(sorted(runtime_source_digests.items())),
            **kwargs,
        )

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": IDENTITIES_SCHEMA,
            "exposure_predecessor_digest": self.exposure_predecessor_digest,
            "execution_configuration_digest": self.execution_configuration_digest,
            "thresholds": [item.to_data() for item in self.thresholds],
            "threshold_commitment": self.threshold_commitment,
            "calibration_algorithm_digest": self.calibration_algorithm_digest,
            "observer": {
                "protocol_id": self.observer_protocol_id,
                "description_protocol_digest": self.observer_description_protocol_digest,
                "scoring_protocol_digest": self.observer_scoring_protocol_digest,
                "environment_digest": self.observer_environment_digest,
                "model_id": self.observer_model_id,
                "reasoning_effort": self.observer_reasoning_effort,
                "model_identity_digest": self.observer_model_identity_digest,
            },
            "ranker": {
                "model_id": self.ranker_model_id,
                "reasoning_effort": self.ranker_reasoning_effort,
                "model_identity_digest": self.ranker_model_identity_digest,
            },
            "runner": {
                "protocol_id": self.runner_protocol_id,
                "algorithm_digest": self.runner_algorithm_digest,
            },
            "codex": {
                "cli_version": self.codex_cli_version,
                "launcher_sha256": self.codex_launcher_sha256,
                "cloud_policy_cache_binding": self.cloud_policy_cache_binding,
            },
            "runtime": {
                "python_runtime_id": self.python_runtime_id,
                "python_runtime_identity_digest": self.python_runtime_identity_digest,
                "source_digests": dict(self.runtime_source_digests),
            },
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypePairExecutionIdentities":
        raw = _object(
            value,
            {
                "schema",
                "exposure_predecessor_digest",
                "execution_configuration_digest",
                "thresholds",
                "threshold_commitment",
                "calibration_algorithm_digest",
                "observer",
                "ranker",
                "runner",
                "codex",
                "runtime",
                "record_digest",
            },
            "execution identities",
        )
        _verify_record_digest(raw, "execution identities")
        if raw["schema"] != IDENTITIES_SCHEMA:
            raise PrototypePairExecutionPrecommitError("identity schema differs")
        observer = _object(
            raw["observer"],
            {
                "protocol_id",
                "description_protocol_digest",
                "scoring_protocol_digest",
                "environment_digest",
                "model_id",
                "reasoning_effort",
                "model_identity_digest",
            },
            "observer identity",
        )
        ranker = _object(
            raw["ranker"],
            {"model_id", "reasoning_effort", "model_identity_digest"},
            "ranker identity",
        )
        runner = _object(
            raw["runner"], {"protocol_id", "algorithm_digest"}, "runner identity"
        )
        codex = _object(
            raw["codex"],
            {"cli_version", "launcher_sha256", "cloud_policy_cache_binding"},
            "Codex identity",
        )
        runtime = _object(
            raw["runtime"],
            {
                "python_runtime_id",
                "python_runtime_identity_digest",
                "source_digests",
            },
            "runtime identity",
        )
        sources = runtime["source_digests"]
        if not isinstance(sources, Mapping) or any(
            not isinstance(key, str) for key in sources
        ):
            raise PrototypePairExecutionPrecommitError(
                "runtime source digests must be an object"
            )
        result = cls.create(
            exposure_predecessor_digest=raw["exposure_predecessor_digest"],
            execution_configuration_digest=raw[
                "execution_configuration_digest"
            ],
            thresholds=tuple(
                PrototypeSceneTagThreshold.from_data(item)
                for item in _list(raw["thresholds"], "threshold records")
            ),
            threshold_commitment=raw["threshold_commitment"],
            calibration_algorithm_digest=raw["calibration_algorithm_digest"],
            observer_protocol_id=observer["protocol_id"],
            observer_description_protocol_digest=observer[
                "description_protocol_digest"
            ],
            observer_scoring_protocol_digest=observer["scoring_protocol_digest"],
            observer_environment_digest=observer["environment_digest"],
            observer_model_id=observer["model_id"],
            observer_reasoning_effort=observer["reasoning_effort"],
            observer_model_identity_digest=observer["model_identity_digest"],
            ranker_model_id=ranker["model_id"],
            ranker_reasoning_effort=ranker["reasoning_effort"],
            ranker_model_identity_digest=ranker["model_identity_digest"],
            runner_protocol_id=runner["protocol_id"],
            runner_algorithm_digest=runner["algorithm_digest"],
            codex_cli_version=codex["cli_version"],
            codex_launcher_sha256=codex["launcher_sha256"],
            cloud_policy_cache_binding=codex["cloud_policy_cache_binding"],
            python_runtime_id=runtime["python_runtime_id"],
            python_runtime_identity_digest=runtime[
                "python_runtime_identity_digest"
            ],
            runtime_source_digests=sources,
        )
        if result.to_data() != dict(raw):
            raise PrototypePairExecutionPrecommitError(
                "execution identities are not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class PrototypePairPanelRole:
    role_id: str
    partition: str
    opaque_side_id: str
    ordinal_within_side: int
    source_panel_id: str
    selection_rank: str

    def __post_init__(self) -> None:
        _identifier(self.role_id, "panel role ID")
        if self.partition not in {"support", "query"}:
            raise PrototypePairExecutionPrecommitError("panel partition differs")
        if self.opaque_side_id not in {"side_0", "side_1"}:
            raise PrototypePairExecutionPrecommitError("opaque side differs")
        maximum = 5 if self.partition == "support" else 0
        if (
            isinstance(self.ordinal_within_side, bool)
            or not isinstance(self.ordinal_within_side, int)
            or not 0 <= self.ordinal_within_side <= maximum
        ):
            raise PrototypePairExecutionPrecommitError("role ordinal differs")
        _text(self.source_panel_id, "source panel ID")
        _require_raw_sha(self.selection_rank, "panel selection rank")

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": ROLE_SCHEMA,
            "role_id": self.role_id,
            "partition": self.partition,
            "opaque_side_id": self.opaque_side_id,
            "ordinal_within_side": self.ordinal_within_side,
            "source_panel_id": self.source_panel_id,
            "selection_rank": self.selection_rank,
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypePairPanelRole":
        raw = _object(
            value,
            {
                "schema",
                "role_id",
                "partition",
                "opaque_side_id",
                "ordinal_within_side",
                "source_panel_id",
                "selection_rank",
                "record_digest",
            },
            "panel role",
        )
        _verify_record_digest(raw, "panel role")
        if raw["schema"] != ROLE_SCHEMA:
            raise PrototypePairExecutionPrecommitError("panel role schema differs")
        result = cls(
            role_id=raw["role_id"],
            partition=raw["partition"],
            opaque_side_id=raw["opaque_side_id"],
            ordinal_within_side=raw["ordinal_within_side"],
            source_panel_id=raw["source_panel_id"],
            selection_rank=raw["selection_rank"],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairExecutionPrecommitError("panel role is not canonical")
        return result


def _query_seal(roles: Sequence[PrototypePairPanelRole]) -> str:
    return _address(
        {
            "schema": "gkm.bongard-prototype-pair-query-role-seal.v1",
            "roles": [item.to_data() for item in roles],
        }
    )


@dataclass(frozen=True, slots=True)
class PrototypePairExecutionPrecommit:
    cohort_plan_digest: str
    cohort_planner_source_sha256: str
    cohort_planner_algorithm_digest: str
    selection_seed_digest: str
    release_descriptor_digest: str
    corpus_manifest_digest: str
    drill_task_id: str
    identities: PrototypePairExecutionIdentities
    support_roles: tuple[PrototypePairPanelRole, ...]
    query_roles: tuple[PrototypePairPanelRole, ...]
    query_role_seal_digest: str
    precommit_source_sha256: str
    precommit_algorithm_digest: str
    algorithm_id: str

    def __post_init__(self) -> None:
        for name in (
            "cohort_plan_digest",
            "cohort_planner_algorithm_digest",
            "selection_seed_digest",
            "release_descriptor_digest",
            "corpus_manifest_digest",
            "query_role_seal_digest",
            "precommit_algorithm_digest",
        ):
            _require_address(getattr(self, name), name)
        _require_raw_sha(
            self.cohort_planner_source_sha256, "cohort planner source SHA-256"
        )
        _require_raw_sha(self.precommit_source_sha256, "precommit source SHA-256")
        _identifier(self.drill_task_id, "drill task ID")
        if not isinstance(self.identities, PrototypePairExecutionIdentities):
            raise TypeError("identities must be PrototypePairExecutionIdentities")
        if (
            len(self.support_roles) != 12
            or len(self.query_roles) != 2
            or any(item.partition != "support" for item in self.support_roles)
            or any(item.partition != "query" for item in self.query_roles)
        ):
            raise PrototypePairExecutionPrecommitError("role counts differ")
        for roles, expected_count in ((self.support_roles, 6), (self.query_roles, 1)):
            if any(
                sum(item.opaque_side_id == side for item in roles) != expected_count
                for side in ("side_0", "side_1")
            ):
                raise PrototypePairExecutionPrecommitError("per-side role counts differ")
        all_roles = self.support_roles + self.query_roles
        if (
            len({item.role_id for item in all_roles}) != 14
            or len({item.source_panel_id for item in all_roles}) != 14
            or set(self.support_panel_ids) & set(self.query_panel_ids)
        ):
            raise PrototypePairExecutionPrecommitError(
                "roles must be unique, complete, and disjoint"
            )
        if self.query_role_seal_digest != _query_seal(self.query_roles):
            raise PrototypePairExecutionPrecommitError("query role seal differs")
        if (
            self.precommit_source_sha256 != PRECOMMIT_SOURCE_SHA256
            or self.precommit_algorithm_digest != execution_precommit_algorithm_digest()
            or self.algorithm_id != ALGORITHM_ID
        ):
            raise PrototypePairExecutionPrecommitError("precommit authority differs")

    @property
    def support_panel_ids(self) -> tuple[str, ...]:
        return tuple(item.source_panel_id for item in self.support_roles)

    @property
    def query_panel_ids(self) -> tuple[str, ...]:
        return tuple(item.source_panel_id for item in self.query_roles)

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": PRECOMMIT_SCHEMA,
            "algorithm_id": self.algorithm_id,
            "cohort": {
                "plan_digest": self.cohort_plan_digest,
                "planner_source_sha256": self.cohort_planner_source_sha256,
                "planner_algorithm_digest": self.cohort_planner_algorithm_digest,
                "selection_seed_digest": self.selection_seed_digest,
                "release_descriptor_digest": self.release_descriptor_digest,
                "corpus_manifest_digest": self.corpus_manifest_digest,
                "drill_task_id": self.drill_task_id,
                "split": "train",
                "exact_task_unused": True,
            },
            "identities": self.identities.to_data(),
            "roles": {
                "support": [item.to_data() for item in self.support_roles],
                "query": [item.to_data() for item in self.query_roles],
                "query_role_seal_digest": self.query_role_seal_digest,
                "model_visible_fields": ["role_id"],
                "source_panel_ids_model_visible": False,
                "source_polarity_model_visible": False,
            },
            "execution": {
                "phase_order": list(PHASE_ORDER),
                "call_budgets": _budget_data(),
                "maximum_model_calls": 44,
                "model_calls_on_complete_candidate_and_query_branch": 44,
                "model_calls_on_no_verified_support_survivor_branch": 41,
                "model_calls_on_calibration_family_rejected_branch": 29,
                "model_calls_on_ranker_error_branch": 42,
                "query_roles_sealed_in_precommit": True,
                "query_pixels_released_after_durable_candidate_freeze": True,
                "durable_candidate_freeze_phase_ordinal": PHASE_ORDER.index(
                    "durable_python_candidate_frozen"
                ),
                "query_pixel_release_phase_ordinal": PHASE_ORDER.index(
                    "two_query_scenes_released_and_observed"
                ),
                "formula_frozen_before_query_observation": True,
                "replay_is_model_free": True,
                "replay_is_tamper_detecting": True,
            },
            "claim_scope": {
                "split": "train",
                "exact_task_unused": True,
                "drill_semantics_reused": True,
                "targeted_engineering_claim": True,
                "benchmark_claim_authorized": False,
                "unseen_claim_authorized": False,
                "validation_claim_authorized": False,
                "official_test_authorized": False,
            },
            "construction_boundary": {
                "accepted_inputs": "verified-plan-and-explicit-digests-only",
                "panel_bytes_read": False,
                "panel_paths_resolved": False,
                "action_program_json_read": False,
                "model_calls_made": False,
                "exposure_ledger_read": False,
                "exposure_ledger_mutated": False,
            },
            "runtime_authority": {
                "precommit_source_sha256": self.precommit_source_sha256,
                "precommit_algorithm_digest": self.precommit_algorithm_digest,
                "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
                "python_is_canonical_authority": True,
                "lean_required": False,
                "lean_removable": True,
                "lean_defines_artifact_identity": False,
                "lean_affects_selection_or_decision": False,
                "lean_required_for_replay": False,
                "optional_secondary_checker_nondecisional": True,
            },
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypePairExecutionPrecommit":
        raw = _object(
            value,
            {
                "schema",
                "algorithm_id",
                "cohort",
                "identities",
                "roles",
                "execution",
                "claim_scope",
                "construction_boundary",
                "runtime_authority",
                "record_digest",
            },
            "execution precommit",
        )
        _verify_record_digest(raw, "execution precommit")
        if raw["schema"] != PRECOMMIT_SCHEMA:
            raise PrototypePairExecutionPrecommitError("precommit schema differs")
        cohort = _object(
            raw["cohort"],
            {
                "plan_digest",
                "planner_source_sha256",
                "planner_algorithm_digest",
                "selection_seed_digest",
                "release_descriptor_digest",
                "corpus_manifest_digest",
                "drill_task_id",
                "split",
                "exact_task_unused",
            },
            "precommit cohort",
        )
        roles = _object(
            raw["roles"],
            {
                "support",
                "query",
                "query_role_seal_digest",
                "model_visible_fields",
                "source_panel_ids_model_visible",
                "source_polarity_model_visible",
            },
            "precommit roles",
        )
        execution = _object(
            raw["execution"],
            {
                "phase_order",
                "call_budgets",
                "maximum_model_calls",
                "model_calls_on_complete_candidate_and_query_branch",
                "model_calls_on_no_verified_support_survivor_branch",
                "model_calls_on_calibration_family_rejected_branch",
                "model_calls_on_ranker_error_branch",
                "query_roles_sealed_in_precommit",
                "query_pixels_released_after_durable_candidate_freeze",
                "durable_candidate_freeze_phase_ordinal",
                "query_pixel_release_phase_ordinal",
                "formula_frozen_before_query_observation",
                "replay_is_model_free",
                "replay_is_tamper_detecting",
            },
            "precommit execution",
        )
        claim = _object(
            raw["claim_scope"],
            {
                "split",
                "exact_task_unused",
                "drill_semantics_reused",
                "targeted_engineering_claim",
                "benchmark_claim_authorized",
                "unseen_claim_authorized",
                "validation_claim_authorized",
                "official_test_authorized",
            },
            "precommit claim scope",
        )
        boundary = _object(
            raw["construction_boundary"],
            {
                "accepted_inputs",
                "panel_bytes_read",
                "panel_paths_resolved",
                "action_program_json_read",
                "model_calls_made",
                "exposure_ledger_read",
                "exposure_ledger_mutated",
            },
            "precommit boundary",
        )
        authority = _object(
            raw["runtime_authority"],
            {
                "precommit_source_sha256",
                "precommit_algorithm_digest",
                "predicate_authority_id",
                "python_is_canonical_authority",
                "lean_required",
                "lean_removable",
                "lean_defines_artifact_identity",
                "lean_affects_selection_or_decision",
                "lean_required_for_replay",
                "optional_secondary_checker_nondecisional",
            },
            "precommit runtime authority",
        )
        expected_literals = {
            "cohort": {"split": "train", "exact_task_unused": True},
            "roles": {
                "model_visible_fields": ["role_id"],
                "source_panel_ids_model_visible": False,
                "source_polarity_model_visible": False,
            },
            "execution": {
                "phase_order": list(PHASE_ORDER),
                "call_budgets": _budget_data(),
                "maximum_model_calls": 44,
                "model_calls_on_complete_candidate_and_query_branch": 44,
                "model_calls_on_no_verified_support_survivor_branch": 41,
                "model_calls_on_calibration_family_rejected_branch": 29,
                "model_calls_on_ranker_error_branch": 42,
                "query_roles_sealed_in_precommit": True,
                "query_pixels_released_after_durable_candidate_freeze": True,
                "durable_candidate_freeze_phase_ordinal": 10,
                "query_pixel_release_phase_ordinal": 11,
                "formula_frozen_before_query_observation": True,
                "replay_is_model_free": True,
                "replay_is_tamper_detecting": True,
            },
            "claim": {
                "split": "train",
                "exact_task_unused": True,
                "drill_semantics_reused": True,
                "targeted_engineering_claim": True,
                "benchmark_claim_authorized": False,
                "unseen_claim_authorized": False,
                "validation_claim_authorized": False,
                "official_test_authorized": False,
            },
            "boundary": {
                "accepted_inputs": "verified-plan-and-explicit-digests-only",
                "panel_bytes_read": False,
                "panel_paths_resolved": False,
                "action_program_json_read": False,
                "model_calls_made": False,
                "exposure_ledger_read": False,
                "exposure_ledger_mutated": False,
            },
            "authority": {
                "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
                "python_is_canonical_authority": True,
                "lean_required": False,
                "lean_removable": True,
                "lean_defines_artifact_identity": False,
                "lean_affects_selection_or_decision": False,
                "lean_required_for_replay": False,
                "optional_secondary_checker_nondecisional": True,
            },
        }
        if (
            {key: cohort[key] for key in expected_literals["cohort"]}
            != expected_literals["cohort"]
            or {key: roles[key] for key in expected_literals["roles"]}
            != expected_literals["roles"]
            or dict(execution) != expected_literals["execution"]
            or dict(claim) != expected_literals["claim"]
            or dict(boundary) != expected_literals["boundary"]
            or {
                key: authority[key]
                for key in expected_literals["authority"]
            }
            != expected_literals["authority"]
        ):
            raise PrototypePairExecutionPrecommitError(
                "execution, claim, boundary, or authority policy differs"
            )
        result = cls(
            cohort_plan_digest=cohort["plan_digest"],
            cohort_planner_source_sha256=cohort["planner_source_sha256"],
            cohort_planner_algorithm_digest=cohort["planner_algorithm_digest"],
            selection_seed_digest=cohort["selection_seed_digest"],
            release_descriptor_digest=cohort["release_descriptor_digest"],
            corpus_manifest_digest=cohort["corpus_manifest_digest"],
            drill_task_id=cohort["drill_task_id"],
            identities=PrototypePairExecutionIdentities.from_data(raw["identities"]),
            support_roles=tuple(
                PrototypePairPanelRole.from_data(item)
                for item in _list(roles["support"], "support roles")
            ),
            query_roles=tuple(
                PrototypePairPanelRole.from_data(item)
                for item in _list(roles["query"], "query roles")
            ),
            query_role_seal_digest=roles["query_role_seal_digest"],
            precommit_source_sha256=authority["precommit_source_sha256"],
            precommit_algorithm_digest=authority["precommit_algorithm_digest"],
            algorithm_id=raw["algorithm_id"],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairExecutionPrecommitError(
                "execution precommit is not canonical"
            )
        return result


def _role_rank(
    plan: PrototypePairCohortPlan, *, opaque_side_id: str, panel_id: str
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-prototype-pair-role-rank.v1",
            "algorithm_id": ALGORITHM_ID,
            "split_rule_id": SPLIT_RULE_ID,
            "execution_precommit_source_sha256": PRECOMMIT_SOURCE_SHA256,
            "execution_precommit_algorithm_digest": (
                execution_precommit_algorithm_digest()
            ),
            "cohort_plan_digest": plan.record_digest,
            "cohort_planner_algorithm_digest": plan.planner_algorithm_digest,
            "selection_seed_digest": plan.selection_seed_digest,
            "release_descriptor_digest": plan.release_descriptor_digest,
            "corpus_manifest_digest": plan.corpus_manifest_digest,
            "opaque_side_id": opaque_side_id,
            "panel_id": panel_id,
        }
    )


def _role_id(
    plan: PrototypePairCohortPlan,
    *,
    partition: str,
    opaque_side_id: str,
    ordinal: int,
    panel_id: str,
) -> str:
    digest = canonical_digest(
        {
            "schema": "gkm.bongard-prototype-pair-opaque-role-id.v1",
            "cohort_plan_digest": plan.record_digest,
            "partition": partition,
            "opaque_side_id": opaque_side_id,
            "ordinal": ordinal,
            "panel_id": panel_id,
        }
    )
    return f"panel_role_{digest[:24]}"


def _split_roles(
    plan: PrototypePairCohortPlan,
) -> tuple[tuple[PrototypePairPanelRole, ...], tuple[PrototypePairPanelRole, ...]]:
    support: list[PrototypePairPanelRole] = []
    query: list[PrototypePairPanelRole] = []
    source_sides = (
        plan.drill.positive_panel_ids,
        plan.drill.negative_panel_ids,
    )
    for side_index, panel_ids in enumerate(source_sides):
        opaque_side_id = f"side_{side_index}"
        ranked = sorted(
            (
                (_role_rank(plan, opaque_side_id=opaque_side_id, panel_id=panel_id), panel_id)
                for panel_id in panel_ids
            )
        )
        query_rank, query_panel_id = ranked[0]
        query.append(
            PrototypePairPanelRole(
                role_id=_role_id(
                    plan,
                    partition="query",
                    opaque_side_id=opaque_side_id,
                    ordinal=0,
                    panel_id=query_panel_id,
                ),
                partition="query",
                opaque_side_id=opaque_side_id,
                ordinal_within_side=0,
                source_panel_id=query_panel_id,
                selection_rank=query_rank,
            )
        )
        for ordinal, (rank, panel_id) in enumerate(ranked[1:]):
            support.append(
                PrototypePairPanelRole(
                    role_id=_role_id(
                        plan,
                        partition="support",
                        opaque_side_id=opaque_side_id,
                        ordinal=ordinal,
                        panel_id=panel_id,
                    ),
                    partition="support",
                    opaque_side_id=opaque_side_id,
                    ordinal_within_side=ordinal,
                    source_panel_id=panel_id,
                    selection_rank=rank,
                )
            )
    return tuple(support), tuple(query)


def prepare_prototype_pair_execution_precommit(
    *,
    cohort_plan: PrototypePairCohortPlan | Mapping[str, Any],
    identities: PrototypePairExecutionIdentities | Mapping[str, Any],
    expected_cohort_plan_digest: str,
    expected_identity_bundle_digest: str,
    expected_exposure_predecessor_digest: str,
) -> PrototypePairExecutionPrecommit:
    """Freeze execution using metadata values only."""

    plan = (
        cohort_plan
        if isinstance(cohort_plan, PrototypePairCohortPlan)
        else PrototypePairCohortPlan.from_data(cohort_plan)
    )
    identity = (
        identities
        if isinstance(identities, PrototypePairExecutionIdentities)
        else PrototypePairExecutionIdentities.from_data(identities)
    )
    if plan.record_digest != _require_address(
        expected_cohort_plan_digest, "expected cohort plan digest"
    ):
        raise PrototypePairExecutionPrecommitError(
            "cohort plan differs from external commitment"
        )
    if identity.record_digest != _require_address(
        expected_identity_bundle_digest, "expected identity bundle digest"
    ):
        raise PrototypePairExecutionPrecommitError(
            "execution identities differ from external commitment"
        )
    exposure_pin = _require_address(
        expected_exposure_predecessor_digest,
        "expected exposure predecessor digest",
    )
    if (
        identity.exposure_predecessor_digest != exposure_pin
        or plan.exposure_predecessor_digest != exposure_pin
    ):
        raise PrototypePairExecutionPrecommitError(
            "exposure predecessor differs across plan, identities, and pin"
        )
    support, query = _split_roles(plan)
    return PrototypePairExecutionPrecommit(
        cohort_plan_digest=plan.record_digest,
        cohort_planner_source_sha256=plan.planner_source_sha256,
        cohort_planner_algorithm_digest=plan.planner_algorithm_digest,
        selection_seed_digest=plan.selection_seed_digest,
        release_descriptor_digest=plan.release_descriptor_digest,
        corpus_manifest_digest=plan.corpus_manifest_digest,
        drill_task_id=plan.drill.task_id,
        identities=identity,
        support_roles=support,
        query_roles=query,
        query_role_seal_digest=_query_seal(query),
        precommit_source_sha256=PRECOMMIT_SOURCE_SHA256,
        precommit_algorithm_digest=execution_precommit_algorithm_digest(),
        algorithm_id=ALGORITHM_ID,
    )


def verify_prototype_pair_execution_precommit(
    precommit: PrototypePairExecutionPrecommit | Mapping[str, Any],
    *,
    cohort_plan: PrototypePairCohortPlan | Mapping[str, Any],
    identities: PrototypePairExecutionIdentities | Mapping[str, Any],
    expected_precommit_digest: str,
    expected_cohort_plan_digest: str,
    expected_identity_bundle_digest: str,
    expected_exposure_predecessor_digest: str,
) -> PrototypePairExecutionPrecommit:
    """Cold-decode and deterministically reconstruct the complete precommit."""

    archived = (
        precommit
        if isinstance(precommit, PrototypePairExecutionPrecommit)
        else PrototypePairExecutionPrecommit.from_data(precommit)
    )
    if archived.record_digest != _require_address(
        expected_precommit_digest, "expected precommit digest"
    ):
        raise PrototypePairExecutionPrecommitError(
            "execution precommit differs from external commitment"
        )
    replay = prepare_prototype_pair_execution_precommit(
        cohort_plan=cohort_plan,
        identities=identities,
        expected_cohort_plan_digest=expected_cohort_plan_digest,
        expected_identity_bundle_digest=expected_identity_bundle_digest,
        expected_exposure_predecessor_digest=expected_exposure_predecessor_digest,
    )
    if replay != archived or replay.record_digest != archived.record_digest:
        raise PrototypePairExecutionPrecommitError(
            "cold-reconstructed execution precommit differs"
        )
    return archived


__all__ = [
    "ALGORITHM_ID",
    "CALL_BUDGETS",
    "PHASE_ORDER",
    "PRECOMMIT_SOURCE_SHA256",
    "REQUIRED_RUNTIME_SOURCE_ROLES",
    "SPLIT_RULE_ID",
    "PrototypePairExecutionIdentities",
    "PrototypePairExecutionPrecommit",
    "PrototypePairExecutionPrecommitError",
    "PrototypePairPanelRole",
    "execution_precommit_algorithm_digest",
    "prepare_prototype_pair_execution_precommit",
    "verify_prototype_pair_execution_precommit",
]
