"""Sealed 12-task TRAIN campaign for prose-grounded Python predicates.

The accepted calibration directory is cold-verified before this command may
read the cohort plan, inspect the archive, create its output directory, release
panel bytes, or call a model.  The official release gate durably appends one
12-task exposure event before support bytes can be released.

For every task, twelve support panels receive one blind discovery observation.
After that batch is durable, one zero-image proposer sees the frozen prose in
both revealed support buckets and proposes an affirmative soft-tag union.  Two
independent role-blind registered-evaluation observations ground those tags,
then Python constructs a closed, conservative predicate version space.  A
typed gap makes no ranker or query calls; otherwise one zero-image ranker may
select one frozen survivor.  The exact formula is durably frozen and committed
before exactly two sealed query panels are released.  Lean is absent and
removable: Python artifacts are the sole decision and replay authority.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from threading import Lock
from typing import Any, Callable, Mapping, Protocol, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import (
    ObjectBongardBatchPlan,
    object_bongard_batch_source_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


COMMAND_ID = "bongard.scene-predicate-campaign/exact-unused-train-12-v3"
TASK_COUNT = 12
SUPPORT_PANEL_COUNT_PER_TASK = 12
DISCOVERY_CALLS_PER_TASK = 12
REGISTERED_A_CALLS_PER_TASK = 12
REGISTERED_B_CALLS_PER_TASK = 12
SUPPORT_VISUAL_CALLS_PER_TASK = 36
SEMANTIC_PROPOSER_CALLS_PER_TASK = 1
MAX_RANKER_CALLS_PER_TASK = 1
QUERY_CALLS_PER_TASK = 2
QUERY_DENOMINATOR = TASK_COUNT * QUERY_CALLS_PER_TASK
MAX_VISUAL_CALLS = TASK_COUNT * (SUPPORT_VISUAL_CALLS_PER_TASK + QUERY_CALLS_PER_TASK)
MAX_RANKER_CALLS = TASK_COUNT
MAX_SEMANTIC_PROPOSER_CALLS = TASK_COUNT

MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
DEFAULT_MINUTES = 15
DEFAULT_EXECUTABLE = "codex"
DEFAULT_EXPECTED_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)
DEFAULT_PARALLEL_WORKERS = 4
MAX_PARALLEL_WORKERS = 4
DEFAULT_CAMPAIGN_MINUTES = 480

TASK_BATCH_SCHEMA = "gkm.bongard-scene-predicate-task-visual-batch.v2"
TASK_REGISTRY_SCHEMA = "gkm.bongard-scene-predicate-task-registry-freeze.v3"
TASK_ROLE_REVEAL_SCHEMA = "gkm.bongard-scene-predicate-task-role-reveal.v1"
TASK_SEMANTIC_PREPARED_SCHEMA = (
    "gkm.bongard-scene-predicate-task-semantic-prepared.v3"
)
TASK_SEMANTIC_PROPOSAL_SCHEMA = (
    "gkm.bongard-scene-predicate-task-semantic-proposal.v3"
)
TASK_IR_SCHEMA = "gkm.bongard-scene-predicate-task-ir-freeze.v3"
TASK_RANK_INPUT_SCHEMA = "gkm.bongard-scene-predicate-task-rank-input.v3"
TASK_RANK_RESULT_SCHEMA = "gkm.bongard-scene-predicate-task-rank-result.v3"
TASK_RESULT_SCHEMA = "gkm.bongard-scene-predicate-task-result.v3"
CAMPAIGN_RUNTIME_SCHEMA = "gkm.bongard-scene-predicate-campaign-runtime.v1"
CAMPAIGN_RUNTIME_CUSTODY_SCHEMA = (
    "gkm.bongard-scene-predicate-campaign-runtime-custody.v1"
)
QUERY_RELEASE_CUSTODY_SCHEMA = (
    "gkm.bongard-scene-predicate-query-release-custody.v1"
)
CAMPAIGN_RESULT_SCHEMA = "gkm.bongard-scene-predicate-campaign-result.v3"
CAMPAIGN_REPLAY_SCHEMA = "gkm.bongard-scene-predicate-campaign-replay.v3"
RESULT_FILENAME = "campaign_result.json"
JOURNAL_DIRECTORY = "journals"

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREREGISTRATION = (
    _REPOSITORY_ROOT / "bongard/data/object_bongard_rubric_train_20260808.prereg.json"
)
DEFAULT_PLAN = (
    _REPOSITORY_ROOT / "bongard/data/object_bongard_rubric_train_20260808.plan.json"
)
DEFAULT_DESCRIPTOR = (
    _REPOSITORY_ROOT / "bongard/data/shape_bongard_v2_release_v1.json"
)
DEFAULT_ARCHIVE = _REPOSITORY_ROOT / "downloads/ShapeBongard_V2.zip"
DEFAULT_SPLIT = (
    _REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json"
)
DEFAULT_EXPOSURE_PREDECESSOR = (
    _REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full"
    / "prototype_pair_python_campaign_20260807_object_v1"
    / "objects/exposure_successor"
    / "1bcde18e387539f13c4006b4a147e61c75feacb86bb031f10a6e8ba3412fe48d.json"
)

PREREGISTRATION_FILE_SHA256 = (
    "10d52f9eec047063e1861cd7c151fa6600cf2c4ef4ad6423784cc419db0fb76e"
)
PLAN_FILE_SHA256 = "c2f07c7885a42f4125f397ddf5bf7f8827b3ef1a6c1fb77e82f08a6ab2b3d523"
PREREGISTRATION_DIGEST = (
    "sha256:b4e29960a9524f5785139a3ddf462d5ddec784d52eb0f2678cb1674820dd8107"
)
PLAN_DIGEST = "sha256:760edd40d91c67fd3c5e3b6f94119754f5368441b479f0940c2c7bd77c17b941"
EXPOSURE_PREDECESSOR_FILE_SHA256 = (
    "1bcde18e387539f13c4006b4a147e61c75feacb86bb031f10a6e8ba3412fe48d"
)
EXPOSURE_PREDECESSOR_DIGEST = (
    "sha256:73f4f6ad2cdb5413456b4298722cc26cd8de9e733e80e7b178d97b87d11fd276"
)

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")

TYPED_SEMANTIC_PROPOSAL_GAP = "typed_semantic_proposal_gap"
TYPED_LANGUAGE_GAP = "typed_language_gap"
TYPED_SELECTIVITY_GAP = "typed_selectivity_gap"
TYPED_GROUNDING_REPEATABILITY_GAP = "typed_grounding_repeatability_gap"
TYPED_TASK_GAP_STATUSES = (
    TYPED_SEMANTIC_PROPOSAL_GAP,
    TYPED_LANGUAGE_GAP,
    TYPED_SELECTIVITY_GAP,
    TYPED_GROUNDING_REPEATABILITY_GAP,
)


class ObjectBongardScenePredicateCampaignCommandError(RuntimeError):
    """Calibration, custody, budget, formula freeze, or replay failed closed."""


def object_bongard_scene_predicate_campaign_command_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "prose_grounded_closed_predicate_ir": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_selection_decision_scoring_or_replay": False,
        "lean_removal_changes_decision": False,
    }


def _automatic_release_source_bindings() -> dict[str, str]:
    from bongard.object_bongard_release_gate import (
        object_bongard_release_gate_source_digest,
    )

    return {
        "batch_source": "sha256:" + object_bongard_batch_source_digest(),
        "release_gate_source": "sha256:"
        + object_bongard_release_gate_source_digest(),
    }


def _canonical_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectBongardScenePredicateCampaignCommandError(f"{label} is not an object")
    try:
        restored = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"{label} is not canonical finite JSON"
        ) from exc
    if not isinstance(restored, dict):
        raise ObjectBongardScenePredicateCampaignCommandError(f"{label} is not an object")
    return restored


def _record(body: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    raw = _canonical_mapping(body, "record body")
    if digest_field in raw:
        raise ObjectBongardScenePredicateCampaignCommandError("record is already sealed")
    raw[digest_field] = "sha256:" + canonical_digest(raw)
    return raw


def _read_exact_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"{label} is unavailable"
        ) from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ObjectBongardScenePredicateCampaignCommandError(f"{label} identity differs")
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"{label} is not canonical JSON"
        ) from exc
    return _canonical_mapping(value, label)


def _load_exact_cohort(
    preregistration_path: Path, plan_path: Path
) -> tuple[dict[str, Any], ObjectBongardBatchPlan]:
    preregistration = _read_exact_json(
        preregistration_path, PREREGISTRATION_FILE_SHA256, "preregistration"
    )
    plan = ObjectBongardBatchPlan.from_data(
        _read_exact_json(plan_path, PLAN_FILE_SHA256, "batch plan")
    )
    body = {key: item for key, item in preregistration.items() if key != "record_digest"}
    families = tuple(task.family for task in plan.tasks)
    if (
        preregistration.get("record_digest") != PREREGISTRATION_DIGEST
        or "sha256:" + canonical_digest(body) != PREREGISTRATION_DIGEST
        or plan.record_digest != PLAN_DIGEST
        or preregistration.get("batch_plan_digest") != plan.record_digest
        or preregistration.get("query_identities_sealed_before_support_pixels") is not True
        or preregistration.get("panel_bytes_opened_before_preregistration") is not False
        or preregistration.get("official_test_authorized") is not False
        or len(plan.tasks) != TASK_COUNT
        or any(families.count(family) != 4 for family in ("bd", "ff", "hd"))
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "preregistered TRAIN cohort differs"
        )
    return preregistration, plan


class CalibrationVerifier(Protocol):
    def __call__(self, root: str | os.PathLike[str], **kwargs: object) -> object: ...


def _verify_accepted_calibration_first(
    calibration_root: str | os.PathLike[str],
    verifier: CalibrationVerifier,
) -> object:
    """The only permitted operation before cohort/archive/output access."""

    verified = verifier(calibration_root)
    if (
        getattr(verified, "accepted", False) is not True
        or getattr(verified, "status", None) != "accepted"
        or getattr(verified, "visual_fresh_call_count", None) != 36
        or getattr(verified, "semantic_proposer_fresh_call_count", None) != 1
        or getattr(verified, "ranker_fresh_call_count", None) != 1
        or not isinstance(getattr(verified, "selected_survivor_digest", None), str)
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "scene-predicate calibration is not accepted and cold-verified"
        )
    if _RAW_DIGEST.fullmatch(str(getattr(verified, "source_digest", ""))) is None:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "accepted calibration source_digest differs"
        )
    if _RAW_DIGEST.fullmatch(
        str(getattr(verified, "semantic_proposal_digest", ""))
    ) is None:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "accepted calibration semantic proposal digest differs"
        )
    for name in (
        "authorization_digest",
        "execution_precommit_digest",
        "discovery_batch_digest",
        "discovery_freeze_digest",
        "registry_digest",
        "evaluation_a_batch_digest",
        "evaluation_b_batch_digest",
        "evaluation_freeze_digest",
        "role_reveal_digest",
        "assessment_digest",
        "rank_input_freeze_digest",
        "rank_result_digest",
        "formula_freeze_digest",
        "replay_digest",
        "result_digest",
    ):
        if _ADDRESS.fullmatch(str(getattr(verified, name, ""))) is None:
            raise ObjectBongardScenePredicateCampaignCommandError(
                f"accepted calibration {name} differs"
            )
    return verified


@dataclass(frozen=True, slots=True)
class ObjectBongardScenePredicateCampaignBudget:
    discovery_calls: int = 0
    semantic_proposer_calls: int = 0
    registered_a_calls: int = 0
    registered_b_calls: int = 0
    ranker_calls: int = 0
    query_calls: int = 0

    @property
    def visual_calls(self) -> int:
        return (
            self.discovery_calls
            + self.registered_a_calls
            + self.registered_b_calls
            + self.query_calls
        )

    def validate_terminal(self, *, task_count: int, completed_tasks: int) -> None:
        if task_count != TASK_COUNT or not 0 <= completed_tasks <= task_count:
            raise ObjectBongardScenePredicateCampaignCommandError("campaign task count differs")
        if (
            self.discovery_calls != task_count * DISCOVERY_CALLS_PER_TASK
            or self.semantic_proposer_calls
            != task_count * SEMANTIC_PROPOSER_CALLS_PER_TASK
            or self.registered_a_calls != task_count * REGISTERED_A_CALLS_PER_TASK
            or self.registered_b_calls != task_count * REGISTERED_B_CALLS_PER_TASK
            or not 0 <= self.ranker_calls <= task_count
            or self.query_calls != completed_tasks * QUERY_CALLS_PER_TASK
            or self.visual_calls > MAX_VISUAL_CALLS
        ):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "campaign physical-call budget differs"
            )


class _CallBudget:
    def __init__(self, *, deadline_monotonic: float | None = None) -> None:
        self._lock = Lock()
        self._deadline_monotonic = deadline_monotonic
        self._counts = {stage: 0 for stage in (
            "discovery", "semantic_proposer", "registered_a", "registered_b",
            "ranker", "query"
        )}

    def count(self, stage: str, limit: int) -> None:
        with self._lock:
            self.assert_within_deadline()
            if stage not in self._counts or self._counts[stage] >= limit:
                raise ObjectBongardScenePredicateCampaignCommandError(
                    f"{stage} physical-call budget exhausted"
                )
            self._counts[stage] += 1

    def assert_within_deadline(self) -> None:
        if (
            self._deadline_monotonic is not None
            and time.monotonic() >= self._deadline_monotonic
        ):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "campaign wall-clock deadline exhausted"
            )

    def snapshot(self) -> ObjectBongardScenePredicateCampaignBudget:
        with self._lock:
            return ObjectBongardScenePredicateCampaignBudget(
                **{f"{key}_calls": value for key, value in self._counts.items()}
            )


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"{label} must be a raw SHA-256 digest"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"{label} must be a sha256: address"
        )
    return value


def _freeze_content(value: "ObjectBongardScenePredicateTaskFreeze") -> dict[str, Any]:
    return {
        "schema": "gkm.bongard-scene-predicate-task-formula-freeze.v1",
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_survivor_digest": value.selected_survivor_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "selected_predicate": dict(value.selected_predicate),
        "formula_frozen_before_query_release": True,
        "query_panel_ids_or_pixels_serialized": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardScenePredicateTaskFreeze:
    """Exact Python formula bytes presented to the query release gate."""

    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    version_space_digest: str
    support_version_space_digest: str
    rank_response_digest: str
    selected_survivor_digest: str
    selected_predicate_digest: str
    selected_predicate: Mapping[str, Any]
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ObjectBongardScenePredicateCampaignCommandError("freeze task ID differs")
        _address(self.task_plan_digest, "freeze task plan digest")
        _address(self.execution_precommit_digest, "freeze execution precommit digest")
        for name in (
            "version_space_digest",
            "support_version_space_digest",
            "rank_response_digest",
            "selected_survivor_digest",
            "selected_predicate_digest",
        ):
            _raw_digest(getattr(self, name), name)
        if self.support_version_space_digest != self.version_space_digest:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "freeze version-space bindings differ"
            )
        selected = _canonical_mapping(self.selected_predicate, "selected predicate")
        if canonical_digest(selected) != self.selected_predicate_digest:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "selected predicate digest differs"
            )
        if selected.get("candidate_digest") != self.selected_survivor_digest:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "selected survivor/formula binding differs"
            )
        if self.record_digest != "sha256:" + canonical_digest(_freeze_content(self)):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "task formula freeze digest differs"
            )

    @classmethod
    def seal(
        cls,
        *,
        task_id: str,
        task_plan_digest: str,
        execution_precommit_digest: str,
        version_space_digest: str,
        rank_response_digest: str,
        selected_predicate: Mapping[str, Any],
    ) -> "ObjectBongardScenePredicateTaskFreeze":
        selected = _canonical_mapping(selected_predicate, "selected predicate")
        selected_survivor = _raw_digest(
            selected.get("candidate_digest"), "selected survivor digest"
        )
        values: dict[str, Any] = {
            "task_id": task_id,
            "task_plan_digest": task_plan_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "version_space_digest": _raw_digest(
                version_space_digest, "version space digest"
            ),
            "support_version_space_digest": version_space_digest,
            "rank_response_digest": _raw_digest(
                rank_response_digest, "rank response digest"
            ),
            "selected_survivor_digest": selected_survivor,
            "selected_predicate_digest": canonical_digest(selected),
            "selected_predicate": selected,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(_freeze_content(provisional)),
        )

    def to_data(self) -> dict[str, Any]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectBongardScenePredicateTaskFreeze":
        raw = _canonical_mapping(value, "task formula freeze")
        expected = {
            "schema", "task_id", "task_plan_digest", "execution_precommit_digest",
            "version_space_digest", "support_version_space_digest",
            "rank_response_digest", "selected_survivor_digest",
            "selected_predicate_digest",
            "selected_predicate", "formula_frozen_before_query_release",
            "query_panel_ids_or_pixels_serialized", *_authority_data(), "record_digest",
        }
        if set(raw) != expected or raw["schema"] != "gkm.bongard-scene-predicate-task-formula-freeze.v1":
            raise ObjectBongardScenePredicateCampaignCommandError(
                "task formula freeze fields differ"
            )
        result = cls(
            task_id=raw["task_id"],
            task_plan_digest=raw["task_plan_digest"],
            execution_precommit_digest=raw["execution_precommit_digest"],
            version_space_digest=raw["version_space_digest"],
            support_version_space_digest=raw["support_version_space_digest"],
            rank_response_digest=raw["rank_response_digest"],
            selected_survivor_digest=raw["selected_survivor_digest"],
            selected_predicate_digest=raw["selected_predicate_digest"],
            selected_predicate=raw["selected_predicate"],
            record_digest=raw["record_digest"],
        )
        if result.to_data() != raw:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "task formula freeze is not canonical"
            )
        return result


def _commit_content(value: "ObjectBongardScenePredicateTaskCommit") -> dict[str, Any]:
    return {
        "schema": "gkm.bongard-scene-predicate-task-decision-commit.v1",
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_survivor_digest": value.selected_survivor_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "task_freeze_digest": value.task_freeze_digest,
        "exact_freeze_payload_digest": value.exact_freeze_payload_digest,
        "task_freeze_store_receipt_digest": value.task_freeze_store_receipt_digest,
        "exact_durable_formula_bytes_committed_before_query_release": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardScenePredicateTaskCommit:
    """Commitment to the exact reloaded formula-freeze payload."""

    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    version_space_digest: str
    support_version_space_digest: str
    rank_response_digest: str
    selected_survivor_digest: str
    selected_predicate_digest: str
    task_freeze_digest: str
    exact_freeze_payload_digest: str
    task_freeze_store_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ObjectBongardScenePredicateCampaignCommandError("commit task ID differs")
        for name in (
            "task_plan_digest", "execution_precommit_digest", "task_freeze_digest",
            "exact_freeze_payload_digest", "task_freeze_store_receipt_digest",
            "record_digest",
        ):
            _address(getattr(self, name), name)
        for name in (
            "version_space_digest", "support_version_space_digest",
            "rank_response_digest", "selected_predicate_digest",
            "selected_survivor_digest",
        ):
            _raw_digest(getattr(self, name), name)
        if self.version_space_digest != self.support_version_space_digest:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "commit version-space bindings differ"
            )
        if self.record_digest != "sha256:" + canonical_digest(_commit_content(self)):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "task decision commit digest differs"
            )

    @classmethod
    def seal(
        cls,
        freeze: ObjectBongardScenePredicateTaskFreeze,
        freeze_receipt: object,
    ) -> "ObjectBongardScenePredicateTaskCommit":
        payload_digest = getattr(freeze_receipt, "payload_digest", None)
        receipt_digest = getattr(freeze_receipt, "record_digest", None)
        values = {
            "task_id": freeze.task_id,
            "task_plan_digest": freeze.task_plan_digest,
            "execution_precommit_digest": freeze.execution_precommit_digest,
            "version_space_digest": freeze.version_space_digest,
            "support_version_space_digest": freeze.support_version_space_digest,
            "rank_response_digest": freeze.rank_response_digest,
            "selected_survivor_digest": freeze.selected_survivor_digest,
            "selected_predicate_digest": freeze.selected_predicate_digest,
            "task_freeze_digest": freeze.record_digest,
            "exact_freeze_payload_digest": _address(
                payload_digest, "freeze payload digest"
            ),
            "task_freeze_store_receipt_digest": _address(
                receipt_digest, "freeze receipt digest"
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest="sha256:" + canonical_digest(_commit_content(provisional)),
        )

    def to_data(self) -> dict[str, Any]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectBongardScenePredicateTaskCommit":
        raw = _canonical_mapping(value, "task decision commit")
        expected = {
            "schema", "task_id", "task_plan_digest", "execution_precommit_digest",
            "version_space_digest", "support_version_space_digest",
            "rank_response_digest", "selected_predicate_digest",
            "selected_survivor_digest",
            "task_freeze_digest", "exact_freeze_payload_digest",
            "task_freeze_store_receipt_digest",
            "exact_durable_formula_bytes_committed_before_query_release",
            *_authority_data(), "record_digest",
        }
        if (
            set(raw) != expected
            or raw["schema"]
            != "gkm.bongard-scene-predicate-task-decision-commit.v1"
            or raw["exact_durable_formula_bytes_committed_before_query_release"]
            is not True
        ):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "task decision commit fields differ"
            )
        result = cls(
            task_id=raw["task_id"],
            task_plan_digest=raw["task_plan_digest"],
            execution_precommit_digest=raw["execution_precommit_digest"],
            version_space_digest=raw["version_space_digest"],
            support_version_space_digest=raw["support_version_space_digest"],
            rank_response_digest=raw["rank_response_digest"],
            selected_survivor_digest=raw["selected_survivor_digest"],
            selected_predicate_digest=raw["selected_predicate_digest"],
            task_freeze_digest=raw["task_freeze_digest"],
            exact_freeze_payload_digest=raw["exact_freeze_payload_digest"],
            task_freeze_store_receipt_digest=raw["task_freeze_store_receipt_digest"],
            record_digest=raw["record_digest"],
        )
        if result.to_data() != raw:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "task decision commit is not canonical"
            )
        return result


def _persist_query_release_custody(
    *,
    prepared: object,
    task: object,
    side: str,
    panel_id: str,
    freeze: ObjectBongardScenePredicateTaskFreeze,
    freeze_receipt: object,
    commit: ObjectBongardScenePredicateTaskCommit,
    commit_receipt: object,
    released: object,
    release_receipt: object,
) -> object:
    store = getattr(prepared, "store", None)
    release_receipt_data = _canonical_mapping(
        getattr(release_receipt, "to_data")(), "query release store receipt"
    )
    record = _record(
        {
            "schema": QUERY_RELEASE_CUSTODY_SCHEMA,
            "command_id": COMMAND_ID,
            "task_id": freeze.task_id,
            "task_plan_digest": freeze.task_plan_digest,
            "execution_precommit_digest": freeze.execution_precommit_digest,
            "query_side": side,
            "sealed_query_panel_id": panel_id,
            "formula_freeze_digest": freeze.record_digest,
            "formula_freeze_payload_digest": _address(
                getattr(freeze_receipt, "payload_digest", None),
                "query custody freeze payload",
            ),
            "formula_freeze_receipt_digest": _address(
                getattr(freeze_receipt, "record_digest", None),
                "query custody freeze receipt",
            ),
            "decision_commit_digest": commit.record_digest,
            "decision_commit_payload_digest": _address(
                getattr(commit_receipt, "payload_digest", None),
                "query custody commit payload",
            ),
            "decision_commit_receipt_digest": _address(
                getattr(commit_receipt, "record_digest", None),
                "query custody commit receipt",
            ),
            "released_query_panel_digest": _address(
                getattr(released, "record_digest", None),
                "query custody released panel",
            ),
            "released_query_store_receipt": release_receipt_data,
            "release_gate_verified_exact_durable_freeze_and_commit": True,
            "custody_witness_persisted_before_visual_observation": True,
            **_authority_data(),
        },
        "custody_digest",
    )
    raw, receipt = _persist_record(
        store,
        object_kind="scene-query-release-custody",
        record=record,
        digest_field="custody_digest",
    )
    if (
        raw["task_id"] != getattr(task, "task_id", None)
        or getattr(release_receipt, "object_digest", None)
        != raw["released_query_panel_digest"]
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query release custody binding differs"
        )
    return receipt


@dataclass(frozen=True, slots=True)
class ObjectBongardScenePredicateQueryPhase:
    freeze: ObjectBongardScenePredicateTaskFreeze
    freeze_receipt: object
    commit: ObjectBongardScenePredicateTaskCommit
    commit_receipt: object
    query_artifacts: tuple[object, object]
    query_release_receipts: tuple[object, object]
    query_custody_receipts: tuple[object, object]


def commit_and_release_object_bongard_scene_predicate_queries(
    *,
    prepared: object,
    archive: object,
    task: object,
    freeze: ObjectBongardScenePredicateTaskFreeze,
    query_observer: Callable[[str, object], object],
    persist_freeze: Callable[..., object] | None = None,
    persist_commit: Callable[..., object] | None = None,
    release_query: Callable[..., tuple[object, object]] | None = None,
    persist_query_custody: Callable[..., object] | None = None,
) -> ObjectBongardScenePredicateQueryPhase:
    """Persist formula+commit, then release and observe exactly two queries."""

    from bongard.object_bongard_release_gate import (
        persist_object_bongard_task_commit,
        persist_object_bongard_task_freeze,
        release_object_bongard_query_panel,
    )

    freeze_writer = persist_freeze or persist_object_bongard_task_freeze
    commit_writer = persist_commit or persist_object_bongard_task_commit
    query_releaser = release_query or release_object_bongard_query_panel
    custody_writer = persist_query_custody or _persist_query_release_custody
    store = getattr(prepared, "store", None)
    precommit = getattr(prepared, "precommit", None)
    task_id = getattr(task, "task_id", None)
    task_plan_digest = getattr(task, "record_digest", None)
    precommit_digest = getattr(precommit, "record_digest", None)
    if (
        freeze.task_id != task_id
        or freeze.task_plan_digest != task_plan_digest
        or freeze.execution_precommit_digest != precommit_digest
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "formula freeze does not bind the released task"
        )
    freeze_receipt = freeze_writer(store=store, freeze=freeze)
    commit = ObjectBongardScenePredicateTaskCommit.seal(freeze, freeze_receipt)
    commit_receipt = commit_writer(store=store, commit=commit)
    query_ids = (
        getattr(task, "side_0_query_panel_id", None),
        getattr(task, "side_1_query_panel_id", None),
    )
    if any(not isinstance(item, str) or not item for item in query_ids):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "sealed query inventory differs"
        )
    artifacts: list[object] = []
    receipts: list[object] = []
    custody_receipts: list[object] = []
    for side, panel_id in zip(("side_0", "side_1"), query_ids, strict=True):
        released, receipt = query_releaser(
            prepared=prepared,
            archive=archive,
            panel_id=panel_id,
            task_freeze=freeze,
            task_commit=commit,
            task_freeze_receipt=freeze_receipt,
            task_commit_receipt=commit_receipt,
        )
        custody_receipts.append(
            custody_writer(
                prepared=prepared,
                task=task,
                side=side,
                panel_id=panel_id,
                freeze=freeze,
                freeze_receipt=freeze_receipt,
                commit=commit,
                commit_receipt=commit_receipt,
                released=released,
                release_receipt=receipt,
            )
        )
        artifacts.append(query_observer(side, released))
        receipts.append(receipt)
    if (
        len(artifacts) != QUERY_CALLS_PER_TASK
        or len(receipts) != QUERY_CALLS_PER_TASK
        or len(custody_receipts) != QUERY_CALLS_PER_TASK
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query phase did not make exactly two observations"
        )
    return ObjectBongardScenePredicateQueryPhase(
        freeze,
        freeze_receipt,
        commit,
        commit_receipt,
        (artifacts[0], artifacts[1]),
        (receipts[0], receipts[1]),
        (custody_receipts[0], custody_receipts[1]),
    )


def replay_object_bongard_scene_predicate_query_phase(
    phase: ObjectBongardScenePredicateQueryPhase,
) -> ObjectBongardScenePredicateQueryPhase:
    """Model-free canonical replay of the formula-to-query custody boundary."""

    if not isinstance(phase, ObjectBongardScenePredicateQueryPhase):
        raise TypeError("phase must be ObjectBongardScenePredicateQueryPhase")
    freeze = ObjectBongardScenePredicateTaskFreeze.from_data(phase.freeze.to_data())
    commit = ObjectBongardScenePredicateTaskCommit.from_data(phase.commit.to_data())
    if (
        commit.task_id != freeze.task_id
        or commit.task_plan_digest != freeze.task_plan_digest
        or commit.execution_precommit_digest != freeze.execution_precommit_digest
        or commit.version_space_digest != freeze.version_space_digest
        or commit.rank_response_digest != freeze.rank_response_digest
        or commit.selected_survivor_digest != freeze.selected_survivor_digest
        or commit.selected_predicate_digest != freeze.selected_predicate_digest
        or commit.task_freeze_digest != freeze.record_digest
        or commit.exact_freeze_payload_digest
        != getattr(phase.freeze_receipt, "payload_digest", None)
        or commit.task_freeze_store_receipt_digest
        != getattr(phase.freeze_receipt, "record_digest", None)
        or getattr(phase.commit_receipt, "object_digest", None) != commit.record_digest
        or len(phase.query_artifacts) != QUERY_CALLS_PER_TASK
        or len(phase.query_release_receipts) != QUERY_CALLS_PER_TASK
        or len(phase.query_custody_receipts) != QUERY_CALLS_PER_TASK
        or any(
            getattr(receipt, "object_kind", None) != "scene-query-release-custody"
            for receipt in phase.query_custody_receipts
        )
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query phase model-free replay differs"
        )
    return phase


@dataclass(frozen=True, slots=True)
class PreparedObjectBongardScenePredicateCampaign:
    output_root: Path
    calibration: object
    preregistration: Mapping[str, Any]
    plan: ObjectBongardBatchPlan
    descriptor: object
    archive: object
    release: object
    runtime_record: Mapping[str, Any]
    runtime_receipt: object
    runtime_custody_witness: Mapping[str, Any]
    runtime_custody_receipt: object


def _fresh_output_root(value: str | os.PathLike[str]) -> Path:
    root = Path(os.path.abspath(os.path.expanduser(str(value))))
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
    except (FileExistsError, OSError) as exc:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign output root must be fresh and creatable"
        ) from exc
    return root


def _archive_task_ids(archive: object) -> tuple[str, ...]:
    members = getattr(archive, "members", None)
    if not isinstance(members, tuple):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "official archive inventory differs"
        )
    tasks: set[str] = set()
    for row in members:
        if not isinstance(row, tuple) or len(row) != 3 or not isinstance(row[0], str):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "official archive member differs"
            )
        parts = row[0].split("/")
        if len(parts) == 7 and parts[0] == "ShapeBongard_V2" and parts[2] == "images":
            tasks.add(parts[3])
    if not tasks:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "official archive contains no task inventory"
        )
    return tuple(sorted(tasks))


def verify_object_bongard_scene_predicate_exposure_transition(
    *, predecessor: object, plan: object, prepared: object
) -> None:
    """Verify the one-event, all-12-tasks, persisted-before-release transition."""

    predecessor_events = getattr(predecessor, "events", None)
    successor = getattr(prepared, "successor", None)
    successor_events = getattr(successor, "events", None)
    plan_tasks = getattr(plan, "tasks", None)
    receipt = getattr(prepared, "exposure_receipt", None)
    authorization = getattr(prepared, "authorization", None)
    if (
        not isinstance(predecessor_events, tuple)
        or not isinstance(successor_events, tuple)
        or not isinstance(plan_tasks, tuple)
        or len(plan_tasks) != TASK_COUNT
        or len(successor_events) != len(predecessor_events) + 1
        or successor_events[:-1] != predecessor_events
        or getattr(successor_events[-1], "task_ids", None)
        != tuple(task.task_id for task in plan_tasks)
        or getattr(successor_events[-1], "panel_ids", None) != ()
        or getattr(receipt, "object_kind", None) != "exposure-successor"
        or getattr(receipt, "object_digest", None) != getattr(successor, "digest", None)
        or getattr(authorization, "exposure_successor_digest", None)
        != getattr(successor, "digest", None)
        or getattr(authorization, "exposure_store_receipt_digest", None)
        != getattr(receipt, "record_digest", None)
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "durable 12-task exposure transition differs"
        )


def _default_calibration_verifier(
    root: str | os.PathLike[str], **_kwargs: object
) -> object:
    from bongard.object_bongard_scene_predicate_calibration_command import (
        verify_object_bongard_scene_predicate_calibration,
    )

    return verify_object_bongard_scene_predicate_calibration(root)


def prepare_object_bongard_scene_predicate_campaign(
    output_root: str | os.PathLike[str],
    *,
    calibration_root: str | os.PathLike[str],
    calibration_verifier: CalibrationVerifier = _default_calibration_verifier,
    preregistration_path: str | os.PathLike[str] = DEFAULT_PREREGISTRATION,
    plan_path: str | os.PathLike[str] = DEFAULT_PLAN,
    descriptor_path: str | os.PathLike[str] = DEFAULT_DESCRIPTOR,
    archive_path: str | os.PathLike[str] = DEFAULT_ARCHIVE,
    split_path: str | os.PathLike[str] = DEFAULT_SPLIT,
    exposure_predecessor_path: str | os.PathLike[str] = DEFAULT_EXPOSURE_PREDECESSOR,
    exposure_observed_at: str | None = None,
    runtime_record_digest: str | None = None,
    runtime_record: Mapping[str, Any] | None = None,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    campaign_minutes: int = DEFAULT_CAMPAIGN_MINUTES,
) -> PreparedObjectBongardScenePredicateCampaign:
    """Verify calibration, bind metadata, and persist exposure before pixels."""

    # Keep this call literally before every cohort/archive/output operation.
    calibration = _verify_accepted_calibration_first(
        calibration_root, calibration_verifier
    )
    if runtime_record_digest is None or runtime_record is None:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "authenticated no-tools runtime must be durable before campaign exposure"
        )
    runtime_record_digest = _address(
        runtime_record_digest, "authenticated runtime digest"
    )
    runtime_raw = _canonical_mapping(runtime_record, "authenticated runtime")
    if runtime_raw.get("runtime_digest") != runtime_record_digest:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "authenticated runtime content/digest binding differs"
        )
    authenticated_runtime = _restore_campaign_runtime(runtime_raw)
    from bongard.transport import PINNED_CODEX_CLI_VERSION

    if runtime_raw.get("launcher_fingerprint") != {
        "version": PINNED_CODEX_CLI_VERSION,
        "launcher_digest": authenticated_runtime.expected_launcher_digest,
    }:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "authenticated runtime launcher fingerprint differs"
        )

    from bongard.corpus import SplitIndex
    from bongard.exposure import ExposureLedger
    from bongard.object_bongard_release_gate import (
        ObjectBongardReleaseStore,
        create_object_bongard_execution_precommit,
        prepare_object_bongard_release,
        verify_prepared_object_bongard_release,
    )
    from bongard.official_panel_archive import OfficialPanelArchive
    from bongard.release import OfficialReleaseDescriptor
    from bongard.object_bongard_scene_predicate_calibration_command import (
        object_bongard_scene_predicate_calibration_command_source_digest,
    )
    from bongard.object_bongard_scene_predicate_ir import (
        object_bongard_scene_predicate_ir_source_digest,
    )
    from bongard.object_bongard_turn_journal import (
        object_bongard_turn_journal_source_digest,
    )
    from bongard.object_scene_visual_frontend import (
        object_scene_visual_frontend_source_digest,
    )
    from bongard.object_scene_semantic_registry import (
        object_scene_semantic_registry_source_digest,
    )
    from bongard.prototype_scene_observer import (
        prototype_scene_transport_source_digest,
    )
    if (
        type(parallel_workers) is not int
        or not 1 <= parallel_workers <= MAX_PARALLEL_WORKERS
        or type(campaign_minutes) is not int
        or not 1 <= campaign_minutes <= 24 * 60
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign execution envelope differs"
        )
    preregistration, plan = _load_exact_cohort(
        Path(preregistration_path), Path(plan_path)
    )
    predecessor = ExposureLedger.from_dict(
        _read_exact_json(
            Path(exposure_predecessor_path),
            EXPOSURE_PREDECESSOR_FILE_SHA256,
            "exposure predecessor",
        )
    )
    if predecessor.digest != EXPOSURE_PREDECESSOR_DIGEST:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "exposure predecessor ledger digest differs"
        )
    descriptor = OfficialReleaseDescriptor.from_dict(
        _read_exact_json_unpinned(Path(descriptor_path), "official release descriptor")
    )
    archive = OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    split = SplitIndex.load(split_path)
    task_ids = _archive_task_ids(archive)
    train_task_ids = tuple(split.canonical_groups["train"])
    exact_used_task_ids = tuple(sorted(predecessor.exposed_task_ids))
    if split.source_digest != plan.split_source_digest:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "official split identity differs"
        )
    root = _fresh_output_root(output_root)
    store = ObjectBongardReleaseStore(root)
    timestamp = exposure_observed_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    calibration_result = _address(
        getattr(calibration, "result_digest", None), "calibration result digest"
    )
    bindings = {
        "campaign_command": "sha256:"
        + object_bongard_scene_predicate_campaign_command_source_digest(),
        "calibration_command": "sha256:"
        + object_bongard_scene_predicate_calibration_command_source_digest(),
        "calibration_result": calibration_result,
        "scene_visual_frontend": "sha256:"
        + object_scene_visual_frontend_source_digest(),
        "scene_semantic_registry": "sha256:"
        + object_scene_semantic_registry_source_digest(),
        "scene_predicate_ir": "sha256:"
        + object_bongard_scene_predicate_ir_source_digest(),
        "turn_journal": "sha256:" + object_bongard_turn_journal_source_digest(),
        "transport": "sha256:" + prototype_scene_transport_source_digest(),
    }
    bindings["authenticated_runtime_record"] = _address(
        runtime_record_digest, "runtime record digest"
    )
    # Persist the exact authenticated runtime and then a custody witness before
    # constructing the execution precommit.  The precommit binds the witness
    # digest, so the exposure successor cannot be replayed without this exact
    # pre-exposure receipt graph.
    runtime_raw, runtime_receipt = _persist_record(
        store,
        object_kind="scene-campaign-runtime",
        record=runtime_raw,
        digest_field="runtime_digest",
    )
    runtime_custody = _runtime_custody_record(
        runtime_record=runtime_raw,
        runtime_receipt=runtime_receipt,
        plan_digest=plan.record_digest,
        predecessor_digest=predecessor.digest,
        release_descriptor_digest=descriptor.digest,
        archive_record_digest=archive.record_digest,
    )
    runtime_custody, runtime_custody_receipt = _persist_record(
        store,
        object_kind="scene-campaign-runtime-custody",
        record=runtime_custody,
        digest_field="custody_digest",
    )
    bindings["runtime_preexposure_custody"] = runtime_custody["custody_digest"]
    bindings["runtime_preexposure_custody_receipt"] = getattr(
        runtime_custody_receipt, "record_digest"
    )
    precommit = create_object_bongard_execution_precommit(
        plan=plan,
        predecessor=predecessor,
        descriptor=descriptor,
        archive=archive,
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        exact_used_task_ids=exact_used_task_ids,
        runtime_source_bindings=bindings,
        configuration={
            "task_count": TASK_COUNT,
            "discovery_calls_per_task": DISCOVERY_CALLS_PER_TASK,
            "semantic_proposer_calls_per_task": SEMANTIC_PROPOSER_CALLS_PER_TASK,
            "registered_a_calls_per_task": REGISTERED_A_CALLS_PER_TASK,
            "registered_b_calls_per_task": REGISTERED_B_CALLS_PER_TASK,
            "ranker_calls_max_per_task": MAX_RANKER_CALLS_PER_TASK,
            "query_calls_per_task": QUERY_CALLS_PER_TASK,
            "score_denominator": QUERY_DENOMINATOR,
            "parallel_workers": parallel_workers,
            "campaign_wall_clock_minutes": campaign_minutes,
            "maximum_visual_calls": MAX_VISUAL_CALLS,
            "maximum_ranker_calls": MAX_RANKER_CALLS,
            "maximum_semantic_proposer_calls": MAX_SEMANTIC_PROPOSER_CALLS,
            "authenticated_runtime_persisted_before_exposure": True,
            "python_canonical": True,
            "lean_required": False,
        },
        exposure_observed_at=timestamp,
        exposure_actor="headless-codex-scene-predicate-campaign",
        exposure_purpose="prose-grounded-python-predicate-support-and-sealed-query",
        exposure_source=f"{COMMAND_ID}:{plan.record_digest}",
    )
    if dict(precommit.runtime_source_bindings) != {
        **bindings,
        **_automatic_release_source_bindings(),
    }:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "execution precommit automatic source bindings differ"
        )
    prepared = prepare_object_bongard_release(
        store=store,
        plan=plan,
        precommit=precommit,
        predecessor=predecessor,
    )
    verify_prepared_object_bongard_release(prepared)
    verify_object_bongard_scene_predicate_exposure_transition(
        predecessor=predecessor, plan=plan, prepared=prepared
    )
    return PreparedObjectBongardScenePredicateCampaign(
        root,
        calibration,
        preregistration,
        plan,
        descriptor,
        archive,
        prepared,
        runtime_raw,
        runtime_receipt,
        runtime_custody,
        runtime_custody_receipt,
    )


def _read_exact_json_unpinned(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"{label} is unavailable or malformed"
        ) from exc
    return _canonical_mapping(value, label)


def _persist_record(
    store: object,
    *,
    object_kind: str,
    record: Mapping[str, Any],
    digest_field: str,
) -> tuple[dict[str, Any], object]:
    raw = _canonical_mapping(record, object_kind)
    digest = _address(raw.get(digest_field), f"{object_kind} digest")
    receipt = store.persist(object_kind=object_kind, object_digest=digest, data=raw)
    restored = store.verify(receipt, expected_data=raw)
    if dict(restored) != raw:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"{object_kind} durable reload differs"
        )
    return raw, receipt


def _runtime_custody_record(
    *,
    runtime_record: Mapping[str, Any],
    runtime_receipt: object,
    plan_digest: str,
    predecessor_digest: str,
    release_descriptor_digest: str,
    archive_record_digest: str,
) -> dict[str, Any]:
    receipt_data = _canonical_mapping(
        getattr(runtime_receipt, "to_data")(), "runtime store receipt"
    )
    return _record(
        {
            "schema": CAMPAIGN_RUNTIME_CUSTODY_SCHEMA,
            "command_id": COMMAND_ID,
            "runtime_digest": _address(
                runtime_record.get("runtime_digest"), "custody runtime digest"
            ),
            "runtime_store_receipt": receipt_data,
            "batch_plan_digest": _address(plan_digest, "custody plan digest"),
            "exposure_predecessor_digest": _address(
                predecessor_digest, "custody predecessor digest"
            ),
            "release_descriptor_digest": _address(
                release_descriptor_digest, "custody descriptor digest"
            ),
            "archive_record_digest": _address(
                archive_record_digest, "custody archive digest"
            ),
            "witness_persisted_and_bound_into_precommit_before_exposure": True,
            **_authority_data(),
        },
        "custody_digest",
    )


def _runtime_record(runtime: object, fingerprint: Mapping[str, str]) -> dict[str, Any]:
    cache = getattr(runtime, "cloud_policy_cache_snapshot", None)
    catalog = getattr(runtime, "model_catalog_snapshot", None)
    attestation = getattr(runtime, "no_tools_attestation", None)
    if catalog is None or attestation is None or not hasattr(attestation, "to_dict"):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign runtime cannot be serialized"
        )
    return _record(
        {
            "schema": CAMPAIGN_RUNTIME_SCHEMA,
            "command_id": COMMAND_ID,
            "runtime_binding": dict(getattr(runtime, "binding")),
            "cloud_policy_cache_snapshot_base64": (
                None
                if cache is None or cache.data is None
                else base64.b64encode(cache.data).decode("ascii")
            ),
            "model_catalog_snapshot_base64": base64.b64encode(catalog.data).decode(
                "ascii"
            ),
            "no_tools_attestation": attestation.to_dict(),
            "launcher_fingerprint": dict(fingerprint),
            "persisted_before_support_release": True,
            **_authority_data(),
        },
        "runtime_digest",
    )


def _create_campaign_runtime(
    *,
    minutes: int,
    executable: str,
    expected_launcher_sha256: str,
    cache_snapshotter: Callable[[], object],
    catalog_snapshotter: Callable[[], object],
    launcher_fingerprinter: Callable[..., Mapping[str, str]],
    runtime_attester: Callable[..., object],
) -> tuple[object, Mapping[str, str]]:
    if (
        type(minutes) is not int
        or not 1 <= minutes <= 120
        or not isinstance(executable, str)
        or not executable
        or _RAW_DIGEST.fullmatch(expected_launcher_sha256) is None
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign runtime selectors differ"
        )
    # This is the same authenticated no-tools runtime constructor used by the
    # accepted calibration.  Campaign custody remains owned by this command.
    from bongard.object_bongard_scene_predicate_calibration_command import (
        _create_runtime,
    )

    authorization = {
        "runtime_policy": {
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "minutes": minutes,
            "verbose": False,
            "executable": executable,
            "expected_launcher_sha256": expected_launcher_sha256,
        }
    }
    return _create_runtime(
        authorization,
        cache_snapshotter=cache_snapshotter,
        catalog_snapshotter=catalog_snapshotter,
        launcher_fingerprinter=launcher_fingerprinter,
        runtime_attester=runtime_attester,
    )


@dataclass(frozen=True, slots=True)
class _TaskSupportPanel:
    ordinal: int
    blind_panel_id: str
    journal_task_id: str
    support_role: int
    released: object
    release_store_receipt: object
    inventory: object
    neutral_panel_digest: str

    @property
    def exact_png_bytes(self) -> bytes:
        return getattr(self.released, "exact_png_bytes")

    @property
    def png_sha256(self) -> str:
        return _address(
            getattr(self.released, "exact_png_digest"), "released PNG digest"
        )[7:]


def _support_commitment(
    *, ordinal: int, blind_panel_id: str, released: object, inventory: object
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-scene-predicate-neutral-support-panel.v1",
            "ordinal": ordinal,
            "blind_panel_id": blind_panel_id,
            "released_record_digest": getattr(released, "record_digest", None),
            "png_digest": getattr(released, "exact_png_digest", None),
            "proposal_inventory_digest": getattr(inventory, "inventory_digest", None),
            "support_role_serialized": False,
        }
    )


def _release_task_support_panels(
    *, prepared: PreparedObjectBongardScenePredicateCampaign, task: object, task_index: int
) -> tuple[_TaskSupportPanel, ...]:
    from bongard.object_bongard_release_gate import (
        release_object_bongard_support_panel,
    )
    from bongard.object_scene_visual_frontend import (
        extract_object_scene_proposal_inventory,
    )

    rows: list[_TaskSupportPanel] = []
    inventory = (
        (0, *tuple(getattr(task, "side_0_support_panel_ids"))),
        (1, *tuple(getattr(task, "side_1_support_panel_ids"))),
    )
    ordinal = 0
    for role_and_ids in inventory:
        role, *panel_ids = role_and_ids
        for panel_id in panel_ids:
            released, receipt = release_object_bongard_support_panel(
                prepared=prepared.release,
                archive=prepared.archive,
                panel_id=panel_id,
            )
            proposals = extract_object_scene_proposal_inventory(
                released.exact_png_bytes
            )
            blind = f"support_panel_{ordinal:02d}"
            neutral = _support_commitment(
                ordinal=ordinal,
                blind_panel_id=blind,
                released=released,
                inventory=proposals,
            )
            rows.append(
                _TaskSupportPanel(
                    ordinal,
                    blind,
                    f"{getattr(task, 'family')}_scene_{task_index:02d}_{ordinal:02d}",
                    int(role),
                    released,
                    receipt,
                    proposals,
                    neutral,
                )
            )
            ordinal += 1
    if (
        len(rows) != SUPPORT_PANEL_COUNT_PER_TASK
        or len({item.neutral_panel_digest for item in rows}) != len(rows)
        or tuple(item.support_role for item in rows) != (0,) * 6 + (1,) * 6
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "released support inventory differs"
        )
    return tuple(rows)


def _observation_context_digest(
    *, task_plan_digest: str, neutral_panel_digest: str, stage: str
) -> str:
    if stage not in ("discovery", "registered_a", "registered_b", "query_side_0", "query_side_1"):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign observation stage differs"
        )
    return "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-scene-predicate-campaign-observation-context.v1",
            "task_plan_digest": task_plan_digest,
            "neutral_panel_digest": neutral_panel_digest,
            "stage": stage,
            "support_role_visible_to_model": False,
            "candidate_or_formula_visible_to_visual_model": False,
        }
    )


def _stage_limit(stage: str) -> int:
    return {
        "discovery": TASK_COUNT * DISCOVERY_CALLS_PER_TASK,
        "semantic_proposer": MAX_SEMANTIC_PROPOSER_CALLS,
        "registered_a": TASK_COUNT * REGISTERED_A_CALLS_PER_TASK,
        "registered_b": TASK_COUNT * REGISTERED_B_CALLS_PER_TASK,
        "query": TASK_COUNT * QUERY_CALLS_PER_TASK,
        "ranker": MAX_RANKER_CALLS,
    }[stage]


def _execute_task_visual_batch(
    task_root: Path,
    *,
    stage: str,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    panels: Sequence[_TaskSupportPanel],
    registry: object | None,
    runtime: object,
    parallel_workers: int,
    transport: Callable[..., object],
    budget: _CallBudget,
) -> tuple[tuple[object, ...], dict[str, Any], object]:
    from bongard.object_bongard_scene_predicate_calibration_command import (
        _frontend_runtime_kwargs,
    )
    from bongard.object_bongard_turn_journal import (
        ObjectBongardNamedImageTurnJournalTransport,
        verify_object_bongard_turn_journal,
    )
    from bongard.object_scene_visual_frontend import (
        ObjectSceneTranscriptMode,
        observe_object_scene_transcript,
        prepare_object_scene_transcript_inputs,
    )

    if stage == "discovery":
        mode = ObjectSceneTranscriptMode.DISCOVERY
        if registry is not None:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "discovery received a soft-tag registry"
            )
    elif stage in ("registered_a", "registered_b"):
        mode = ObjectSceneTranscriptMode.REGISTERED_EVALUATION
        if registry is None:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "registered pass lacks its frozen registry"
            )
    else:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "support visual batch stage differs"
        )
    task_plan_digest = getattr(task, "record_digest")

    def one(index: int) -> tuple[object, str, str]:
        panel = panels[index]
        context = _observation_context_digest(
            task_plan_digest=task_plan_digest,
            neutral_panel_digest=panel.neutral_panel_digest,
            stage=stage,
        )
        model_inputs = prepare_object_scene_transcript_inputs(
            panel.exact_png_bytes, panel.inventory, mode, registry
        )
        relative = Path(JOURNAL_DIRECTORY) / stage / f"panel_{index:02d}"
        def counted_transport(*args: object, **kwargs: object) -> object:
            budget.count(stage, _stage_limit(stage))
            return transport(*args, **kwargs)

        journal = ObjectBongardNamedImageTurnJournalTransport(
            task_root / relative,
            authorization_digest=prepared.release.authorization.record_digest,
            execution_precommit_digest=prepared.release.precommit.record_digest,
            task_id=panel.journal_task_id,
            turn_kind=stage,
            expected_prompt=model_inputs.prompt,
            expected_images=model_inputs.presentation,
            expected_output_schema=model_inputs.output_schema,
            runtime=runtime,
            underlying_transport=counted_transport,
        )
        artifact = observe_object_scene_transcript(
            panel.exact_png_bytes,
            scene_id=panel.blind_panel_id,
            observation_context_digest=context,
            mode=mode,
            registry=registry,
            inventory=panel.inventory,
            expected_panel_sha256=panel.png_sha256,
            **_frontend_runtime_kwargs(runtime),
            transport=journal,
        )
        budget.assert_within_deadline()
        if journal.fresh_call_count != 1 or journal.reused_call_count != 0:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "support visual journal call accounting differs"
            )
        summary = verify_object_bongard_turn_journal(journal)
        return artifact, str(relative), summary.record_digest

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        outcomes = tuple(executor.map(one, range(len(panels))))
    artifacts = tuple(item[0] for item in outcomes)
    rows = [
        {
            "ordinal": panel.ordinal,
            "blind_panel_id": panel.blind_panel_id,
            "neutral_panel_digest": panel.neutral_panel_digest,
            "proposal_inventory_digest": getattr(panel.inventory, "inventory_digest"),
            "observation_context_digest": getattr(artifact, "observation_context_digest"),
            "artifact": artifact.to_data(),
            "artifact_digest": getattr(artifact, "artifact_digest"),
            "journal_directory": outcome[1],
            "journal_summary_digest": outcome[2],
        }
        for panel, artifact, outcome in zip(panels, artifacts, outcomes, strict=True)
    ]
    record = _record(
        {
            "schema": TASK_BATCH_SCHEMA,
            "command_id": COMMAND_ID,
            "task_plan_digest": task_plan_digest,
            "stage": stage,
            "rows": rows,
            "support_roles_serialized": False,
            "candidate_or_formula_serialized": False,
            "fresh_visual_call_count": len(rows),
            "reused_visual_call_count": 0,
            **_authority_data(),
        },
        "batch_digest",
    )
    raw, receipt = _persist_record(
        prepared.release.store,
        object_kind=f"scene-{stage.replace('_', '-')}-batch",
        record=record,
        digest_field="batch_digest",
    )
    restored = _restore_task_visual_batch(
        raw,
        stage=stage,
        task=task,
        panels=panels,
        registry=registry,
    )
    return restored, raw, receipt


def _restore_task_visual_batch(
    batch: Mapping[str, Any],
    *,
    stage: str,
    task: object,
    panels: Sequence[_TaskSupportPanel],
    registry: object | None,
) -> tuple[object, ...]:
    from bongard.object_scene_visual_frontend import (
        ObjectSceneTranscriptArtifact,
        ObjectSceneTranscriptMode,
        verify_object_scene_transcript_artifact,
    )

    raw = _canonical_mapping(batch, f"{stage} durable batch")
    body = {key: item for key, item in raw.items() if key != "batch_digest"}
    rows = raw.get("rows")
    expected_mode = (
        ObjectSceneTranscriptMode.DISCOVERY
        if stage == "discovery"
        else ObjectSceneTranscriptMode.REGISTERED_EVALUATION
    )
    expected_fields = {
        "schema",
        "command_id",
        "task_plan_digest",
        "stage",
        "rows",
        "support_roles_serialized",
        "candidate_or_formula_serialized",
        "fresh_visual_call_count",
        "reused_visual_call_count",
        *_authority_data(),
        "batch_digest",
    }
    expected_row_fields = {
        "ordinal",
        "blind_panel_id",
        "neutral_panel_digest",
        "proposal_inventory_digest",
        "observation_context_digest",
        "artifact",
        "artifact_digest",
        "journal_directory",
        "journal_summary_digest",
    }
    if (
        set(raw) != expected_fields
        or raw.get("schema") != TASK_BATCH_SCHEMA
        or raw.get("command_id") != COMMAND_ID
        or raw.get("task_plan_digest") != getattr(task, "record_digest")
        or raw.get("stage") != stage
        or raw.get("batch_digest") != "sha256:" + canonical_digest(body)
        or not isinstance(rows, list)
        or len(rows) != SUPPORT_PANEL_COUNT_PER_TASK
        or raw.get("support_roles_serialized") is not False
        or raw.get("candidate_or_formula_serialized") is not False
        or raw.get("fresh_visual_call_count") != SUPPORT_PANEL_COUNT_PER_TASK
        or raw.get("reused_visual_call_count") != 0
        or any(raw.get(key) != value for key, value in _authority_data().items())
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"{stage} durable batch policy differs"
        )
    result: list[object] = []
    for index, (panel, row) in enumerate(zip(panels, rows, strict=True)):
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise ObjectBongardScenePredicateCampaignCommandError(
                f"{stage} durable batch row differs"
            )
        context = _observation_context_digest(
            task_plan_digest=getattr(task, "record_digest"),
            neutral_panel_digest=panel.neutral_panel_digest,
            stage=stage,
        )
        artifact = ObjectSceneTranscriptArtifact.from_data(
            row.get("artifact"), expected_artifact_digest=row.get("artifact_digest")
        )
        verify_object_scene_transcript_artifact(
            artifact,
            panel.exact_png_bytes,
            expected_scene_id=panel.blind_panel_id,
            expected_observation_context_digest=context,
            expected_panel_sha256=panel.png_sha256,
            expected_artifact_digest=artifact.artifact_digest,
        )
        if (
            row.get("ordinal") != panel.ordinal
            or row.get("blind_panel_id") != panel.blind_panel_id
            or row.get("neutral_panel_digest") != panel.neutral_panel_digest
            or row.get("proposal_inventory_digest")
            != getattr(panel.inventory, "inventory_digest")
            or row.get("observation_context_digest") != context
            or artifact.inventory != panel.inventory
            or artifact.mode is not expected_mode
            or artifact.registry != registry
            or row.get("journal_directory")
            != str(Path(JOURNAL_DIRECTORY) / stage / f"panel_{index:02d}")
            or _ADDRESS.fullmatch(str(row.get("journal_summary_digest", ""))) is None
        ):
            raise ObjectBongardScenePredicateCampaignCommandError(
                f"{stage} durable artifact binding differs"
            )
        result.append(artifact)
    return tuple(result)


def _forbidden_named_transport(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("campaign cold replay attempted a visual model call")


def _forbidden_text_transport(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("campaign cold replay attempted a text model call")


def _cold_replay_task_visual_batch(
    task_root: Path,
    *,
    stage: str,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    panels: Sequence[_TaskSupportPanel],
    registry: object | None,
    runtime: object,
    batch: Mapping[str, Any],
) -> tuple[str, ...]:
    from bongard.object_bongard_scene_predicate_calibration_command import (
        _frontend_runtime_kwargs,
    )
    from bongard.object_bongard_turn_journal import (
        ObjectBongardNamedImageTurnJournalTransport,
        verify_object_bongard_turn_journal,
    )
    from bongard.object_scene_visual_frontend import (
        ObjectSceneTranscriptMode,
        observe_object_scene_transcript,
        prepare_object_scene_transcript_inputs,
    )

    artifacts = _restore_task_visual_batch(
        batch, stage=stage, task=task, panels=panels, registry=registry
    )
    mode = (
        ObjectSceneTranscriptMode.DISCOVERY
        if stage == "discovery"
        else ObjectSceneTranscriptMode.REGISTERED_EVALUATION
    )
    summaries: list[str] = []
    for index, (panel, artifact, row) in enumerate(
        zip(panels, artifacts, batch["rows"], strict=True)
    ):
        context = _observation_context_digest(
            task_plan_digest=getattr(task, "record_digest"),
            neutral_panel_digest=panel.neutral_panel_digest,
            stage=stage,
        )
        model_inputs = prepare_object_scene_transcript_inputs(
            panel.exact_png_bytes, panel.inventory, mode, registry
        )
        relative = Path(JOURNAL_DIRECTORY) / stage / f"panel_{index:02d}"
        journal = ObjectBongardNamedImageTurnJournalTransport(
            task_root / relative,
            authorization_digest=prepared.release.authorization.record_digest,
            execution_precommit_digest=prepared.release.precommit.record_digest,
            task_id=panel.journal_task_id,
            turn_kind=stage,
            expected_prompt=model_inputs.prompt,
            expected_images=model_inputs.presentation,
            expected_output_schema=model_inputs.output_schema,
            runtime=runtime,
            underlying_transport=_forbidden_named_transport,
        )
        replayed = observe_object_scene_transcript(
            panel.exact_png_bytes,
            scene_id=panel.blind_panel_id,
            observation_context_digest=context,
            mode=mode,
            registry=registry,
            inventory=panel.inventory,
            expected_panel_sha256=panel.png_sha256,
            **_frontend_runtime_kwargs(runtime),
            transport=journal,
        )
        summary = verify_object_bongard_turn_journal(journal)
        if (
            replayed != artifact
            or journal.fresh_call_count != 0
            or journal.reused_call_count != 1
            or summary.record_digest != row["journal_summary_digest"]
        ):
            raise ObjectBongardScenePredicateCampaignCommandError(
                f"{stage} visual journal cold replay differs"
            )
        summaries.append(summary.record_digest)
    return tuple(summaries)


def _task_role_rows(
    panels: Sequence[_TaskSupportPanel],
) -> tuple[Mapping[str, object], ...]:
    rows = tuple(
        {
            "ordinal": panel.ordinal,
            "neutral_panel_digest": panel.neutral_panel_digest,
            "historical_role": panel.support_role,
            "blind_panel_id": panel.blind_panel_id,
        }
        for panel in panels
    )
    if (
        len(rows) != SUPPORT_PANEL_COUNT_PER_TASK
        or tuple(item["historical_role"] for item in rows) != (0,) * 6 + (1,) * 6
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task support role inventory differs"
        )
    return rows


def _freeze_task_role_reveal(
    *,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    panels: Sequence[_TaskSupportPanel],
    discovery_batch: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, object], ...], dict[str, Any], object]:
    rows = _task_role_rows(panels)
    record = _record(
        {
            "schema": TASK_ROLE_REVEAL_SCHEMA,
            "command_id": COMMAND_ID,
            "task_plan_digest": getattr(task, "record_digest"),
            "discovery_batch_digest": discovery_batch["batch_digest"],
            "rows": [dict(item) for item in rows],
            "revealed_after_blind_discovery_batch_was_durable": True,
            "semantic_proposer_calls_after_reveal": SEMANTIC_PROPOSER_CALLS_PER_TASK,
            "registered_visual_calls_after_reveal": (
                REGISTERED_A_CALLS_PER_TASK + REGISTERED_B_CALLS_PER_TASK
            ),
            "registered_visual_evaluators_receive_roles": False,
            **_authority_data(),
        },
        "role_reveal_digest",
    )
    raw, receipt = _persist_record(
        prepared.release.store,
        object_kind="scene-task-role-reveal",
        record=record,
        digest_field="role_reveal_digest",
    )
    if raw["rows"] != [dict(item) for item in rows]:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task role reveal durable replay differs"
        )
    return rows, raw, receipt


def _semantic_prepared_record(
    *,
    task: object,
    discovery_batch: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
    semantic_prepared: object,
) -> dict[str, Any]:
    return _record(
        {
            "schema": TASK_SEMANTIC_PREPARED_SCHEMA,
            "command_id": COMMAND_ID,
            "task_plan_digest": getattr(task, "record_digest"),
            "discovery_batch_digest": discovery_batch["batch_digest"],
            "role_reveal_digest": role_reveal["role_reveal_digest"],
            "prepared_input": semantic_prepared.to_data(),
            "preparation_digest": semantic_prepared.preparation_digest,
            "prepared_input_persisted_before_zero_image_proposer_call": True,
            **_authority_data(),
        },
        "semantic_prepared_digest",
    )


def _freeze_task_semantic_prepared(
    *,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    discovery_batch: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
    discovery_artifacts: Sequence[object],
    role_rows: Sequence[Mapping[str, object]],
) -> tuple[object, dict[str, Any], object]:
    from bongard.object_scene_semantic_registry import (
        ObjectScenePreparedSemanticRegistryProposal,
        prepare_object_scene_semantic_registry_proposal,
    )

    semantic_prepared = prepare_object_scene_semantic_registry_proposal(
        tuple(discovery_artifacts), tuple(role_rows)
    )
    record = _semantic_prepared_record(
        task=task,
        discovery_batch=discovery_batch,
        role_reveal=role_reveal,
        semantic_prepared=semantic_prepared,
    )
    raw, receipt = _persist_record(
        prepared.release.store,
        object_kind="scene-task-semantic-prepared",
        record=record,
        digest_field="semantic_prepared_digest",
    )
    restored = ObjectScenePreparedSemanticRegistryProposal.from_data(
        raw["prepared_input"]
    )
    expected = prepare_object_scene_semantic_registry_proposal(
        tuple(discovery_artifacts), tuple(role_rows)
    )
    if restored != semantic_prepared or expected != semantic_prepared:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task semantic prepared input durable replay differs"
        )
    return restored, raw, receipt


def _semantic_proposal_result_record(
    *,
    task: object,
    semantic_prepared_record: Mapping[str, Any],
    semantic_proposal: object,
    registry: object,
    payload: Mapping[str, Any],
    transport_receipt: object,
    journal_directory: str,
    journal_summary_digest: str,
) -> dict[str, Any]:
    status = getattr(semantic_proposal, "status", None)
    if (
        status not in ("proposed", "typed_proposal_gap")
        or getattr(semantic_proposal, "preparation_digest", None)
        != semantic_prepared_record["preparation_digest"]
        or getattr(semantic_proposal, "registry_digest", None)
        != getattr(registry, "registry_digest", None)
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task semantic proposal status differs"
        )
    return _record(
        {
            "schema": TASK_SEMANTIC_PROPOSAL_SCHEMA,
            "command_id": COMMAND_ID,
            "task_plan_digest": getattr(task, "record_digest"),
            "semantic_prepared_digest": semantic_prepared_record[
                "semantic_prepared_digest"
            ],
            "semantic_proposal": semantic_proposal.to_data(),
            "semantic_proposal_digest": semantic_proposal.proposal_digest,
            "semantic_proposal_status": status,
            "semantic_proposal_valid": status == "proposed",
            "semantic_registry": registry.to_data(),
            "semantic_registry_digest": registry.registry_digest,
            "proposer_payload": _canonical_mapping(
                payload, "task semantic proposer payload"
            ),
            "proposer_receipt": transport_receipt.to_dict(),
            "proposer_receipt_digest": transport_receipt.receipt_digest,
            "proposer_journal_directory": journal_directory,
            "proposer_journal_summary_digest": journal_summary_digest,
            "proposer_fresh_call_count": 1,
            "proposer_reused_call_count": 0,
            "quarantined_concept_count": len(
                semantic_proposal.dropped_concepts
            ),
            "quarantined_concept_digests": [
                item.drop_digest for item in semantic_proposal.dropped_concepts
            ],
            "invalid_optional_rows_do_not_discard_valid_concepts_when_each_orientation_retains_one": True,
            "orientation_coverage_gap_suppresses_otherwise_valid_concepts_from_registry": True,
            "structural_or_zero_orientation_payload_becomes_zero_tag_typed_gap": True,
            **_authority_data(),
        },
        "semantic_proposal_result_digest",
    )


def _semantic_payload_gap_code(semantic_prepared: object) -> str:
    usable_by_role = {
        role: sum(
            item["usable"] is True and item["historical_role"] == role
            for item in getattr(semantic_prepared, "alias_bindings")
        )
        for role in (0, 1)
    }
    return (
        "insufficient_discovery_evidence"
        if any(count < 2 for count in usable_by_role.values())
        else "payload_rejected"
    )


def _restore_task_semantic_proposal(
    record: Mapping[str, Any],
    *,
    task: object,
    semantic_prepared_record: Mapping[str, Any],
    semantic_prepared: object,
    discovery_artifacts: Sequence[object],
    role_rows: Sequence[Mapping[str, object]],
) -> tuple[object, object]:
    from bongard.object_bongard_turn_journal import _receipt_from_data
    from bongard.object_scene_semantic_registry import (
        ObjectSceneSemanticRegistryPayloadError,
        ObjectSceneSemanticRegistryProposal,
        build_object_scene_semantic_registry_gap,
        build_object_scene_semantic_registry_proposal,
        verify_object_scene_semantic_registry_proposal,
    )
    from bongard.object_scene_visual_frontend import ObjectSceneSoftTagRegistry

    raw = _canonical_mapping(record, "task semantic proposal result")
    _validate_self_sealed_record(
        raw,
        schema=TASK_SEMANTIC_PROPOSAL_SCHEMA,
        digest_field="semantic_proposal_result_digest",
        label="task semantic proposal result",
    )
    persisted_proposal = ObjectSceneSemanticRegistryProposal.from_data(
        raw["semantic_proposal"]
    )
    persisted_registry = ObjectSceneSoftTagRegistry.from_data(
        raw["semantic_registry"]
    )
    payload = _canonical_mapping(raw["proposer_payload"], "semantic proposer payload")
    if raw.get("semantic_proposal_status") == "proposed":
        proposal, registry = build_object_scene_semantic_registry_proposal(
            semantic_prepared, payload
        )
    elif raw.get("semantic_proposal_status") == "typed_proposal_gap":
        try:
            build_object_scene_semantic_registry_proposal(semantic_prepared, payload)
        except ObjectSceneSemanticRegistryPayloadError:
            pass
        else:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "task semantic proposal gap payload is valid"
            )
        expected_gap_code = _semantic_payload_gap_code(semantic_prepared)
        if persisted_proposal.gap_code != expected_gap_code:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "task semantic proposal gap code differs"
            )
        proposal, registry = build_object_scene_semantic_registry_gap(
            semantic_prepared, expected_gap_code, payload
        )
    else:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task semantic proposal result status differs"
        )
    transport_receipt = _receipt_from_data(raw["proposer_receipt"])
    expected = _semantic_proposal_result_record(
        task=task,
        semantic_prepared_record=semantic_prepared_record,
        semantic_proposal=proposal,
        registry=registry,
        payload=payload,
        transport_receipt=transport_receipt,
        journal_directory=raw["proposer_journal_directory"],
        journal_summary_digest=raw["proposer_journal_summary_digest"],
    )
    if (
        proposal != persisted_proposal
        or registry != persisted_registry
        or raw != expected
        or (proposal.status == "typed_proposal_gap" and tuple(registry.tags))
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task semantic proposal differs on reconstruction"
        )
    verify_object_scene_semantic_registry_proposal(
        proposal,
        registry,
        tuple(discovery_artifacts),
        tuple(role_rows),
    )
    return proposal, registry


def _execute_task_semantic_proposal(
    task_root: Path,
    *,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    task_index: int,
    runtime: object,
    semantic_prepared_record: Mapping[str, Any],
    semantic_prepared: object,
    discovery_artifacts: Sequence[object],
    role_rows: Sequence[Mapping[str, object]],
    text_transport: Callable[..., object],
    budget: _CallBudget,
) -> tuple[object, object, dict[str, Any], object]:
    from bongard.object_bongard_scene_predicate_calibration_command import (
        _journal_runtime_kwargs,
    )
    from bongard.object_bongard_turn_journal import (
        ObjectBongardTextTurnJournalTransport,
        verify_object_bongard_turn_journal,
    )
    from bongard.object_scene_semantic_registry import (
        ObjectSceneSemanticRegistryPayloadError,
        build_object_scene_semantic_registry_gap,
        build_object_scene_semantic_registry_proposal,
    )

    relative = Path(JOURNAL_DIRECTORY) / "semantic_registry_proposer"

    def counted_transport(*args: object, **kwargs: object) -> object:
        budget.count("semantic_proposer", _stage_limit("semantic_proposer"))
        return text_transport(*args, **kwargs)

    journal = ObjectBongardTextTurnJournalTransport(
        task_root / relative,
        authorization_digest=prepared.release.authorization.record_digest,
        execution_precommit_digest=prepared.release.precommit.record_digest,
        task_id=f"{getattr(task, 'family')}_scene_semantic_{task_index:02d}",
        turn_kind="semantic_registry_proposal",
        expected_prompt=semantic_prepared.prompt,
        expected_output_schema=semantic_prepared.output_schema,
        runtime=runtime,
        underlying_transport=counted_transport,
    )
    result = journal(
        semantic_prepared.prompt,
        semantic_prepared.output_schema,
        **_journal_runtime_kwargs(runtime),
    )
    budget.assert_within_deadline()
    payload = _canonical_mapping(result.payload, "task semantic proposer payload")
    try:
        proposal, registry = build_object_scene_semantic_registry_proposal(
            semantic_prepared, payload
        )
    except ObjectSceneSemanticRegistryPayloadError:
        proposal, registry = build_object_scene_semantic_registry_gap(
            semantic_prepared, _semantic_payload_gap_code(semantic_prepared), payload
        )
    if journal.fresh_call_count != 1 or journal.reused_call_count != 0:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task semantic proposer call accounting differs"
        )
    summary = verify_object_bongard_turn_journal(journal)
    record = _semantic_proposal_result_record(
        task=task,
        semantic_prepared_record=semantic_prepared_record,
        semantic_proposal=proposal,
        registry=registry,
        payload=payload,
        transport_receipt=result.receipt,
        journal_directory=str(relative),
        journal_summary_digest=summary.record_digest,
    )
    raw, receipt = _persist_record(
        prepared.release.store,
        object_kind="scene-task-semantic-proposal",
        record=record,
        digest_field="semantic_proposal_result_digest",
    )
    restored_proposal, restored_registry = _restore_task_semantic_proposal(
        raw,
        task=task,
        semantic_prepared_record=semantic_prepared_record,
        semantic_prepared=semantic_prepared,
        discovery_artifacts=discovery_artifacts,
        role_rows=role_rows,
    )
    if restored_proposal != proposal or restored_registry != registry:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task semantic proposal durable reload differs"
        )
    return proposal, registry, raw, receipt


def _cold_replay_task_semantic_proposal(
    task_root: Path,
    *,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    task_index: int,
    runtime: object,
    semantic_prepared_record: Mapping[str, Any],
    semantic_prepared: object,
    semantic_proposal_record: Mapping[str, Any],
    discovery_artifacts: Sequence[object],
    role_rows: Sequence[Mapping[str, object]],
) -> tuple[object, object, str]:
    from bongard.object_bongard_scene_predicate_calibration_command import (
        _journal_runtime_kwargs,
    )
    from bongard.object_bongard_turn_journal import (
        ObjectBongardTextTurnJournalTransport,
        verify_object_bongard_turn_journal,
    )

    relative = Path(JOURNAL_DIRECTORY) / "semantic_registry_proposer"
    journal = ObjectBongardTextTurnJournalTransport(
        task_root / relative,
        authorization_digest=prepared.release.authorization.record_digest,
        execution_precommit_digest=prepared.release.precommit.record_digest,
        task_id=f"{getattr(task, 'family')}_scene_semantic_{task_index:02d}",
        turn_kind="semantic_registry_proposal",
        expected_prompt=semantic_prepared.prompt,
        expected_output_schema=semantic_prepared.output_schema,
        runtime=runtime,
        underlying_transport=_forbidden_text_transport,
    )
    replayed = journal(
        semantic_prepared.prompt,
        semantic_prepared.output_schema,
        **_journal_runtime_kwargs(runtime),
    )
    summary = verify_object_bongard_turn_journal(journal)
    proposal, registry = _restore_task_semantic_proposal(
        semantic_proposal_record,
        task=task,
        semantic_prepared_record=semantic_prepared_record,
        semantic_prepared=semantic_prepared,
        discovery_artifacts=discovery_artifacts,
        role_rows=role_rows,
    )
    if (
        _canonical_mapping(replayed.payload, "replayed task semantic payload")
        != semantic_proposal_record["proposer_payload"]
        or replayed.receipt.to_dict()
        != semantic_proposal_record["proposer_receipt"]
        or summary.record_digest
        != semantic_proposal_record["proposer_journal_summary_digest"]
        or journal.fresh_call_count != 0
        or journal.reused_call_count != 1
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task semantic proposer cold replay differs"
        )
    return proposal, registry, summary.record_digest


def _freeze_task_registry(
    *,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    discovery_batch: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
    semantic_prepared_record: Mapping[str, Any],
    semantic_proposal_record: Mapping[str, Any],
    semantic_proposal: object,
    registry: object,
    discovery_artifacts: Sequence[object],
    role_rows: Sequence[Mapping[str, object]],
) -> tuple[object, dict[str, Any], object]:
    from bongard.object_scene_semantic_registry import (
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
        verify_object_scene_semantic_registry_proposal,
    )
    from bongard.object_scene_visual_frontend import ObjectSceneSoftTagRegistry

    verify_object_scene_semantic_registry_proposal(
        semantic_proposal,
        registry,
        tuple(discovery_artifacts),
        tuple(role_rows),
    )
    record = _record(
        {
            "schema": TASK_REGISTRY_SCHEMA,
            "command_id": COMMAND_ID,
            "task_plan_digest": getattr(task, "record_digest"),
            "discovery_batch_digest": discovery_batch["batch_digest"],
            "role_reveal_digest": role_reveal["role_reveal_digest"],
            "semantic_prepared_digest": semantic_prepared_record[
                "semantic_prepared_digest"
            ],
            "semantic_proposal_result_digest": semantic_proposal_record[
                "semantic_proposal_result_digest"
            ],
            "semantic_proposal_digest": getattr(
                semantic_proposal, "proposal_digest"
            ),
            "semantic_proposal_status": getattr(semantic_proposal, "status"),
            "registry": registry.to_data(),
            "registry_digest": registry.registry_digest,
            "registry_derivation_mode": (
                ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
            ),
            "orientation_membership_discarded_before_registered_visual_calls": True,
            "registered_visual_evaluators_receive_roles": False,
            "persisted_and_reloaded_before_registered_pass_a": True,
            **_authority_data(),
        },
        "registry_freeze_digest",
    )
    raw, receipt = _persist_record(
        prepared.release.store,
        object_kind="scene-task-registry",
        record=record,
        digest_field="registry_freeze_digest",
    )
    restored = ObjectSceneSoftTagRegistry.from_data(raw["registry"])
    verify_object_scene_semantic_registry_proposal(
        semantic_proposal,
        restored,
        tuple(discovery_artifacts),
        tuple(role_rows),
    )
    if restored != registry:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task soft-tag registry durable replay differs"
        )
    return restored, raw, receipt


def _registered_envelopes_match(
    artifacts_a: Sequence[object], artifacts_b: Sequence[object]
) -> bool:
    if len(artifacts_a) != SUPPORT_PANEL_COUNT_PER_TASK or len(artifacts_b) != len(
        artifacts_a
    ):
        return False
    fields = (
        "panel_digest",
        "inventory_digest",
        "registry_digest",
        "preparation_digest",
        "prompt_digest",
        "output_schema_digest",
        "presentation",
    )
    return all(
        all(getattr(first, name) == getattr(second, name) for name in fields)
        and first.observation_context_digest != second.observation_context_digest
        for first, second in zip(artifacts_a, artifacts_b, strict=True)
    )


def _freeze_task_ir(
    *,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    registry: object,
    semantic_proposal: object,
    semantic_proposal_record: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
    role_rows: Sequence[Mapping[str, object]],
    discovery_artifacts: Sequence[object],
    registered_a_artifacts: Sequence[object],
    registered_b_artifacts: Sequence[object],
    discovery_batch: Mapping[str, Any],
    registered_a_batch: Mapping[str, Any],
    registered_b_batch: Mapping[str, Any],
) -> tuple[object, dict[str, Any], object]:
    from bongard.object_bongard_scene_predicate_ir import (
        SCENE_CALIBRATION_BUNDLE_SCHEMA,
        ScenePredicateCalibrationBundle,
        build_object_bongard_scene_predicate_calibration_bundle,
        cold_replay_object_bongard_scene_predicate_calibration_bundle,
    )
    from bongard.object_scene_semantic_registry import (
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
    )

    if not _registered_envelopes_match(
        registered_a_artifacts, registered_b_artifacts
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task registered pass A/B model-visible envelopes differ"
        )
    bundle = build_object_bongard_scene_predicate_calibration_bundle(
        registry,
        tuple(discovery_artifacts),
        tuple(registered_a_artifacts),
        tuple(registered_b_artifacts),
        tuple(role_rows),
        semantic_registry_proposal=semantic_proposal,
    )
    data = bundle.to_data()
    if (
        data.get("schema") != SCENE_CALIBRATION_BUNDLE_SCHEMA
        or bundle.registry_derivation_mode
        != ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        or bundle.registry_derivation_digest
        != getattr(semantic_proposal, "proposal_digest", None)
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task IR does not bind the current role-aware semantic registry"
        )
    decoded = ScenePredicateCalibrationBundle.from_data(data)
    replayed = cold_replay_object_bongard_scene_predicate_calibration_bundle(
        decoded,
        registry,
        semantic_registry_proposal=semantic_proposal,
        discovery_artifacts=tuple(discovery_artifacts),
        role_rows=tuple(role_rows),
    )
    if replayed != decoded:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task Python IR cold replay differs"
        )
    record = _record(
        {
            "schema": TASK_IR_SCHEMA,
            "command_id": COMMAND_ID,
            "task_plan_digest": getattr(task, "record_digest"),
            "discovery_batch_digest": discovery_batch["batch_digest"],
            "registered_a_batch_digest": registered_a_batch["batch_digest"],
            "registered_b_batch_digest": registered_b_batch["batch_digest"],
            "role_reveal_digest": role_reveal["role_reveal_digest"],
            "semantic_proposal_result_digest": semantic_proposal_record[
                "semantic_proposal_result_digest"
            ],
            "semantic_proposal_digest": getattr(
                semantic_proposal, "proposal_digest"
            ),
            "semantic_proposal_status": getattr(semantic_proposal, "status"),
            "role_rows": [dict(item) for item in role_rows],
            "roles_revealed_after_discovery_before_semantic_proposer": True,
            "registered_visual_passes_were_role_blind": True,
            "bundle": data,
            "bundle_digest": bundle.bundle_digest,
            "model_calls_during_python_build_or_replay": 0,
            **_authority_data(),
        },
        "ir_freeze_digest",
    )
    encoded_size = len(canonical_json(record)) + 1
    if encoded_size > 64 * 1024 * 1024:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task IR freeze exceeds the explicit 64 MiB durable-store envelope"
        )
    raw, receipt = _persist_record(
        prepared.release.store,
        object_kind="scene-task-ir",
        record=record,
        digest_field="ir_freeze_digest",
    )
    durable_bundle = ScenePredicateCalibrationBundle.from_data(raw["bundle"])
    if durable_bundle != bundle:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task IR durable reload differs"
        )
    return durable_bundle, raw, receipt


def _ranker_output_schema(survivors: Sequence[str]) -> dict[str, object]:
    from bongard.transport import validate_codex_strict_output_schema

    values = tuple(survivors)
    if (
        not 1 <= len(values) <= 64
        or len(set(values)) != len(values)
        or any(_RAW_DIGEST.fullmatch(item) is None for item in values)
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task ranker slate differs"
        )
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "selected_survivor_digest": {"type": "string", "enum": list(values)}
        },
        "required": ["selected_survivor_digest"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _digest_free_task_ranker_value(value: object) -> object:
    """Project frozen IR rows into the deterministic model-visible view."""

    if isinstance(value, Mapping):
        return {
            key: _digest_free_task_ranker_value(item)
            for key, item in value.items()
            if key == "candidate_digest" or not key.endswith("_digest")
        }
    if isinstance(value, (list, tuple)):
        return [_digest_free_task_ranker_value(item) for item in value]
    return value


def _digest_free_task_ranker_row(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _canonical_mapping(value, "task ranker row")
    projected = _digest_free_task_ranker_value(row)
    if not isinstance(projected, dict):  # pragma: no cover - structural guard
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task ranker slate projection differs"
        )
    candidate_digest = projected.get("candidate_digest")
    if (
        not isinstance(candidate_digest, str)
        or _RAW_DIGEST.fullmatch(candidate_digest) is None
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task ranker slate projection lost candidate identity"
        )
    return projected


def _task_bundle_field(bundle: object, field: str) -> object:
    if isinstance(bundle, Mapping):
        return bundle.get(field)
    return getattr(bundle, field, None)


def _task_typed_gap_status(*, semantic_valid: bool, bundle: object) -> str | None:
    """Name the first failed evidence gate, matching calibration semantics."""

    if type(semantic_valid) is not bool:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task semantic proposal decision is not Boolean"
        )
    if not semantic_valid:
        return TYPED_SEMANTIC_PROPOSAL_GAP
    survivors = _task_bundle_field(bundle, "complete_survivor_digests")
    if not isinstance(survivors, (list, tuple)):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task survivor inventory differs"
        )
    if survivors:
        return None
    for field, status in (
        ("coverage_gate", TYPED_LANGUAGE_GAP),
        ("selectivity_gate", TYPED_SELECTIVITY_GAP),
        ("repeatability_gate", TYPED_GROUNDING_REPEATABILITY_GAP),
    ):
        gate = _task_bundle_field(bundle, field)
        passed = gate.get("passed") if isinstance(gate, Mapping) else getattr(
            gate, "passed", None
        )
        if type(passed) is not bool:
            raise ObjectBongardScenePredicateCampaignCommandError(
                f"task {field} decision differs"
            )
        if not passed:
            return status
    raise ObjectBongardScenePredicateCampaignCommandError(
        "empty task survivor space has no failed evidence gate"
    )


def _ranker_prompt(rows: Sequence[Mapping[str, Any]]) -> str:
    slate = tuple(_canonical_mapping(item, "task ranker row") for item in rows)
    _ranker_output_schema(tuple(item["candidate_digest"] for item in slate))
    return (
        "Choose exactly one already-verified frozen Python predicate. Compare "
        "only the displayed affirmative formula meaning, orientation, "
        "complexity, and complete repeat-tested support result. Do not create, "
        "edit, combine, negate, or repair a predicate. Return only "
        "selected_survivor_digest.\n\nFrozen survivor slate:\n"
        + canonical_json(list(slate)).decode("utf-8")
    )


def _assert_task_ranker_privacy(
    rows: Sequence[Mapping[str, Any]], *, task: object, prompt: str
) -> None:
    def visit(value: object, key: str | None = None) -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and (
            _RAW_DIGEST.fullmatch(value) is not None
            or _ADDRESS.fullmatch(value) is not None
        ):
            if key != "candidate_digest" or _RAW_DIGEST.fullmatch(value) is None:
                raise ObjectBongardScenePredicateCampaignCommandError(
                    "ranker slate leaks a non-candidate lineage digest"
                )

    for row in rows:
        visit(row)
    hidden = {
        getattr(task, "task_id", None),
        getattr(task, "record_digest", None),
        *tuple(getattr(task, "side_0_support_panel_ids", ())),
        *tuple(getattr(task, "side_1_support_panel_ids", ())),
        getattr(task, "side_0_query_panel_id", None),
        getattr(task, "side_1_query_panel_id", None),
    }
    if any(isinstance(item, str) and item and item in prompt for item in hidden):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "ranker prompt leaks task or panel identity"
        )
    if len(prompt.encode("utf-8")) > 256_000:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "ranker prompt exceeds its bounded semantic envelope"
        )


def _rank_task_bundle(
    task_root: Path,
    *,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    task_index: int,
    bundle: object,
    ir_record: Mapping[str, Any],
    semantic_proposal_record: Mapping[str, Any],
    runtime: object,
    text_transport: Callable[..., object],
    budget: _CallBudget,
) -> tuple[dict[str, Any], object, dict[str, Any], object, str | None]:
    from bongard.object_bongard_scene_predicate_calibration_command import (
        _journal_runtime_kwargs,
    )
    from bongard.object_bongard_turn_journal import (
        ObjectBongardTextTurnJournalTransport,
        verify_object_bongard_turn_journal,
    )

    survivors = tuple(getattr(bundle, "complete_survivor_digests"))
    semantic_valid = semantic_proposal_record.get("semantic_proposal_valid") is True
    gap_status = _task_typed_gap_status(
        semantic_valid=semantic_valid,
        bundle=bundle,
    )
    if semantic_valid:
        slate = tuple(
            _digest_free_task_ranker_row(item)
            for item in getattr(bundle, "ranker_slate")
        )
        omitted = tuple(dict(item) for item in getattr(bundle, "omitted_survivors"))
    else:
        slate = ()
        omitted = tuple(
            {
                "candidate_digest": item,
                "reason": "mandatory_semantic_proposal_gap",
            }
            for item in survivors
        )
    rank_input_record = _record(
        {
            "schema": TASK_RANK_INPUT_SCHEMA,
            "command_id": COMMAND_ID,
            "task_plan_digest": getattr(task, "record_digest"),
            "ir_freeze_digest": ir_record["ir_freeze_digest"],
            "bundle_digest": getattr(bundle, "bundle_digest"),
            "semantic_proposal_result_digest": semantic_proposal_record[
                "semantic_proposal_result_digest"
            ],
            "semantic_proposal_digest": semantic_proposal_record[
                "semantic_proposal_digest"
            ],
            "semantic_proposal_status": semantic_proposal_record[
                "semantic_proposal_status"
            ],
            "semantic_proposal_valid": semantic_valid,
            "typed_gap_status": gap_status,
            "complete_survivor_digests": list(survivors),
            "ranker_slate": list(slate),
            "omitted_survivors": list(omitted),
            "ranker_input_frozen_before_call": True,
            **_authority_data(),
        },
        "rank_input_digest",
    )
    rank_input, rank_input_receipt = _persist_record(
        prepared.release.store,
        object_kind="scene-task-rank-input",
        record=rank_input_record,
        digest_field="rank_input_digest",
    )
    if gap_status is not None:
        if slate:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "typed gap carries a ranker slate"
            )
        result_record = _record(
            {
                "schema": TASK_RANK_RESULT_SCHEMA,
                "command_id": COMMAND_ID,
                "task_plan_digest": getattr(task, "record_digest"),
                "rank_input_digest": rank_input["rank_input_digest"],
                "status": gap_status,
                "ranker_called": False,
                "ranker_fresh_call_count": 0,
                "selected_survivor_digest": None,
                "ranker_payload": None,
                "ranker_journal_directory": None,
                **_authority_data(),
            },
            "rank_result_digest",
        )
        result, receipt = _persist_record(
            prepared.release.store,
            object_kind="scene-task-rank-result",
            record=result_record,
            digest_field="rank_result_digest",
        )
        return rank_input, rank_input_receipt, result, receipt, None

    slate_digests = tuple(item["candidate_digest"] for item in slate)
    if set(slate_digests) | {item["candidate_digest"] for item in omitted} != set(
        survivors
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "ranker input does not account for every survivor"
        )
    prompt = _ranker_prompt(slate)
    _assert_task_ranker_privacy(slate, task=task, prompt=prompt)
    schema = _ranker_output_schema(slate_digests)
    relative = Path(JOURNAL_DIRECTORY) / "ranker"
    def counted_transport(*args: object, **kwargs: object) -> object:
        budget.count("ranker", _stage_limit("ranker"))
        return text_transport(*args, **kwargs)

    journal = ObjectBongardTextTurnJournalTransport(
        task_root / relative,
        authorization_digest=prepared.release.authorization.record_digest,
        execution_precommit_digest=prepared.release.precommit.record_digest,
        task_id=f"{getattr(task, 'family')}_scene_rank_{task_index:02d}",
        turn_kind="survivor_rank",
        expected_prompt=prompt,
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=counted_transport,
    )
    transport_result = journal(prompt, schema, **_journal_runtime_kwargs(runtime))
    budget.assert_within_deadline()
    payload = _canonical_mapping(transport_result.payload, "task ranker payload")
    selected = payload.get("selected_survivor_digest")
    if set(payload) != {"selected_survivor_digest"} or selected not in slate_digests:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task ranker escaped the frozen survivor slate"
        )
    if journal.fresh_call_count != 1 or journal.reused_call_count != 0:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task ranker journal call accounting differs"
        )
    summary = verify_object_bongard_turn_journal(journal)
    result_record = _record(
        {
            "schema": TASK_RANK_RESULT_SCHEMA,
            "command_id": COMMAND_ID,
            "task_plan_digest": getattr(task, "record_digest"),
            "rank_input_digest": rank_input["rank_input_digest"],
            "status": "selected_frozen_survivor",
            "ranker_called": True,
            "ranker_fresh_call_count": 1,
            "selected_survivor_digest": selected,
            "ranker_payload": payload,
            "ranker_journal_directory": str(relative),
            "ranker_journal_summary_digest": summary.record_digest,
            **_authority_data(),
        },
        "rank_result_digest",
    )
    result, receipt = _persist_record(
        prepared.release.store,
        object_kind="scene-task-rank-result",
        record=result_record,
        digest_field="rank_result_digest",
    )
    return rank_input, rank_input_receipt, result, receipt, str(selected)


def _cold_replay_task_ranker(
    task_root: Path,
    *,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    task_index: int,
    runtime: object,
    rank_input: Mapping[str, Any],
    rank_result: Mapping[str, Any],
) -> str | None:
    from bongard.object_bongard_scene_predicate_calibration_command import (
        _journal_runtime_kwargs,
    )
    from bongard.object_bongard_turn_journal import (
        ObjectBongardTextTurnJournalTransport,
        verify_object_bongard_turn_journal,
    )

    slate = tuple(rank_input["ranker_slate"])
    if not slate:
        semantic_valid = rank_input.get("semantic_proposal_valid") is True
        expected_status = rank_input.get("typed_gap_status")
        if (
            expected_status not in TYPED_TASK_GAP_STATUSES
            or (
                not semantic_valid
                and expected_status != TYPED_SEMANTIC_PROPOSAL_GAP
            )
            or (
                semantic_valid
                and expected_status
                not in (
                    TYPED_LANGUAGE_GAP,
                    TYPED_SELECTIVITY_GAP,
                    TYPED_GROUNDING_REPEATABILITY_GAP,
                )
            )
            or (semantic_valid and rank_input["complete_survivor_digests"])
            or rank_result.get("status") != expected_status
            or rank_result["ranker_called"] is not False
            or rank_result["selected_survivor_digest"] is not None
            or (task_root / JOURNAL_DIRECTORY / "ranker").exists()
        ):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "typed gap forged a ranker journal"
            )
        return None
    if rank_input.get("typed_gap_status") is not None:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "nonempty cold-replay ranker slate carries a typed gap"
        )
    prompt = _ranker_prompt(slate)
    _assert_task_ranker_privacy(slate, task=task, prompt=prompt)
    schema = _ranker_output_schema(
        tuple(item["candidate_digest"] for item in slate)
    )
    relative = Path(JOURNAL_DIRECTORY) / "ranker"
    journal = ObjectBongardTextTurnJournalTransport(
        task_root / relative,
        authorization_digest=prepared.release.authorization.record_digest,
        execution_precommit_digest=prepared.release.precommit.record_digest,
        task_id=f"{getattr(task, 'family')}_scene_rank_{task_index:02d}",
        turn_kind="survivor_rank",
        expected_prompt=prompt,
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=_forbidden_text_transport,
    )
    replayed = journal(prompt, schema, **_journal_runtime_kwargs(runtime))
    payload = _canonical_mapping(replayed.payload, "cold-replayed task ranker")
    summary = verify_object_bongard_turn_journal(journal)
    selected = payload.get("selected_survivor_digest")
    if (
        payload != rank_result["ranker_payload"]
        or selected != rank_result["selected_survivor_digest"]
        or summary.record_digest != rank_result["ranker_journal_summary_digest"]
        or journal.fresh_call_count != 0
        or journal.reused_call_count != 1
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task ranker journal cold replay differs"
        )
    return str(selected)


def _observe_task_query(
    task_root: Path,
    *,
    side: str,
    released: object,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    task_index: int,
    registry: object,
    runtime: object,
    transport: Callable[..., object],
    budget: _CallBudget,
) -> object:
    from bongard.object_bongard_scene_predicate_calibration_command import (
        _frontend_runtime_kwargs,
    )
    from bongard.object_bongard_turn_journal import (
        ObjectBongardNamedImageTurnJournalTransport,
        verify_object_bongard_turn_journal,
    )
    from bongard.object_scene_visual_frontend import (
        ObjectSceneTranscriptMode,
        extract_object_scene_proposal_inventory,
        observe_object_scene_transcript,
        prepare_object_scene_transcript_inputs,
    )

    if side not in ("side_0", "side_1"):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query side differs"
        )
    png = getattr(released, "exact_png_bytes")
    inventory = extract_object_scene_proposal_inventory(png)
    blind = f"query_panel_{int(side[-1]):02d}"
    neutral = canonical_digest(
        {
            "schema": "gkm.bongard-scene-predicate-neutral-query-panel.v1",
            "blind_panel_id": blind,
            "released_record_digest": getattr(released, "record_digest"),
            "png_digest": getattr(released, "exact_png_digest"),
            "proposal_inventory_digest": inventory.inventory_digest,
            "sealed_query_side_serialized_to_model": False,
        }
    )
    stage = f"query_{side}"
    context = _observation_context_digest(
        task_plan_digest=getattr(task, "record_digest"),
        neutral_panel_digest=neutral,
        stage=stage,
    )
    mode = ObjectSceneTranscriptMode.REGISTERED_EVALUATION
    model_inputs = prepare_object_scene_transcript_inputs(
        png, inventory, mode, registry
    )
    relative = Path(JOURNAL_DIRECTORY) / "query" / side

    def counted_transport(*args: object, **kwargs: object) -> object:
        budget.count("query", _stage_limit("query"))
        return transport(*args, **kwargs)

    journal = ObjectBongardNamedImageTurnJournalTransport(
        task_root / relative,
        authorization_digest=prepared.release.authorization.record_digest,
        execution_precommit_digest=prepared.release.precommit.record_digest,
        task_id=f"{getattr(task, 'family')}_scene_query_{task_index:02d}",
        turn_kind=stage,
        expected_prompt=model_inputs.prompt,
        expected_images=model_inputs.presentation,
        expected_output_schema=model_inputs.output_schema,
        runtime=runtime,
        underlying_transport=counted_transport,
    )
    artifact = observe_object_scene_transcript(
        png,
        scene_id=blind,
        observation_context_digest=context,
        mode=mode,
        registry=registry,
        inventory=inventory,
        expected_panel_sha256=_address(
            getattr(released, "exact_png_digest"), "query PNG digest"
        )[7:],
        **_frontend_runtime_kwargs(runtime),
        transport=journal,
    )
    budget.assert_within_deadline()
    if journal.fresh_call_count != 1 or journal.reused_call_count != 0:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query visual journal call accounting differs"
        )
    verify_object_bongard_turn_journal(journal)
    return artifact


def _cold_replay_task_query(
    task_root: Path,
    *,
    side: str,
    released: object,
    expected_artifact: object,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    task_index: int,
    registry: object,
    runtime: object,
) -> str:
    from bongard.object_bongard_scene_predicate_calibration_command import (
        _frontend_runtime_kwargs,
    )
    from bongard.object_bongard_turn_journal import (
        ObjectBongardNamedImageTurnJournalTransport,
        verify_object_bongard_turn_journal,
    )
    from bongard.object_scene_visual_frontend import (
        ObjectSceneTranscriptMode,
        extract_object_scene_proposal_inventory,
        observe_object_scene_transcript,
        prepare_object_scene_transcript_inputs,
    )

    png = getattr(released, "exact_png_bytes")
    inventory = extract_object_scene_proposal_inventory(png)
    blind = f"query_panel_{int(side[-1]):02d}"
    neutral = canonical_digest(
        {
            "schema": "gkm.bongard-scene-predicate-neutral-query-panel.v1",
            "blind_panel_id": blind,
            "released_record_digest": getattr(released, "record_digest"),
            "png_digest": getattr(released, "exact_png_digest"),
            "proposal_inventory_digest": inventory.inventory_digest,
            "sealed_query_side_serialized_to_model": False,
        }
    )
    stage = f"query_{side}"
    context = _observation_context_digest(
        task_plan_digest=getattr(task, "record_digest"),
        neutral_panel_digest=neutral,
        stage=stage,
    )
    mode = ObjectSceneTranscriptMode.REGISTERED_EVALUATION
    model_inputs = prepare_object_scene_transcript_inputs(
        png, inventory, mode, registry
    )
    relative = Path(JOURNAL_DIRECTORY) / "query" / side
    journal = ObjectBongardNamedImageTurnJournalTransport(
        task_root / relative,
        authorization_digest=prepared.release.authorization.record_digest,
        execution_precommit_digest=prepared.release.precommit.record_digest,
        task_id=f"{getattr(task, 'family')}_scene_query_{task_index:02d}",
        turn_kind=stage,
        expected_prompt=model_inputs.prompt,
        expected_images=model_inputs.presentation,
        expected_output_schema=model_inputs.output_schema,
        runtime=runtime,
        underlying_transport=_forbidden_named_transport,
    )
    replayed = observe_object_scene_transcript(
        png,
        scene_id=blind,
        observation_context_digest=context,
        mode=mode,
        registry=registry,
        inventory=inventory,
        expected_panel_sha256=_address(
            getattr(released, "exact_png_digest"), "query PNG digest"
        )[7:],
        **_frontend_runtime_kwargs(runtime),
        transport=journal,
    )
    summary = verify_object_bongard_turn_journal(journal)
    if (
        replayed != expected_artifact
        or journal.fresh_call_count != 0
        or journal.reused_call_count != 1
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query visual journal cold replay differs"
        )
    return summary.record_digest


def _query_score_rows(
    *, bundle: object, selected_candidate_data: Mapping[str, Any], artifacts: Sequence[object]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from bongard.evidence import Disposition
    from bongard.object_bongard_scene_predicate_ir import (
        SceneOrientation,
        ScenePredicateCandidate,
        ScenePredicateLanguage,
        adapt_object_scene_registered_single,
        evaluate_object_scene_candidate,
    )

    version = getattr(bundle, "version_space")
    language = ScenePredicateLanguage.from_data(version["language"])
    candidate = ScenePredicateCandidate.from_data(
        selected_candidate_data, language=language
    )
    if len(artifacts) != QUERY_CALLS_PER_TASK:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query artifact inventory differs"
        )
    expected = (
        (Disposition.PRESENT, Disposition.CERTIFIED_ABSENT)
        if candidate.orientation is SceneOrientation.GROUP0_POSITIVE
        else (Disposition.CERTIFIED_ABSENT, Disposition.PRESENT)
    )
    rows: list[dict[str, Any]] = []
    for index, (artifact, wanted) in enumerate(zip(artifacts, expected, strict=True)):
        panel = adapt_object_scene_registered_single(
            f"query_panel_{index:02d}", artifact
        )
        actual = evaluate_object_scene_candidate(candidate, language, panel)
        rows.append(
            {
                "side": f"side_{index}",
                "query_artifact_digest": getattr(artifact, "artifact_digest"),
                "query_observation_digest": panel.observation_digest,
                "expected_disposition": wanted.value,
                "actual_disposition": actual.value,
                "correct": actual is wanted,
                "indeterminate_or_error_scores_incorrect": actual
                in (Disposition.INDETERMINATE, Disposition.ERROR),
            }
        )
    return rows[0], rows[1]


def _typed_gap_score_rows() -> tuple[dict[str, Any], dict[str, Any]]:
    return tuple(  # type: ignore[return-value]
        {
            "side": f"side_{index}",
            "query_artifact_digest": None,
            "query_observation_digest": None,
            "expected_disposition": None,
            "actual_disposition": "typed_gap_no_query",
            "correct": False,
            "indeterminate_or_error_scores_incorrect": True,
        }
        for index in range(QUERY_CALLS_PER_TASK)
    )


@dataclass(frozen=True, slots=True)
class _TaskOutcome:
    result: Mapping[str, Any]
    result_receipt: object
    scored_correct: int
    queried: bool


def _budget_delta(
    before: ObjectBongardScenePredicateCampaignBudget,
    after: ObjectBongardScenePredicateCampaignBudget,
) -> dict[str, int]:
    return {
        name: getattr(after, name) - getattr(before, name)
        for name in (
            "discovery_calls",
            "semantic_proposer_calls",
            "registered_a_calls",
            "registered_b_calls",
            "ranker_calls",
            "query_calls",
        )
    }


def _execute_task(
    *,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    task_index: int,
    runtime: object,
    named_image_transport: Callable[..., object],
    text_transport: Callable[..., object],
    parallel_workers: int,
    budget: _CallBudget,
) -> _TaskOutcome:
    from bongard.object_bongard_scene_predicate_ir import (
        cold_replay_object_bongard_scene_predicate_calibration_bundle,
    )

    task_root = prepared.output_root / "tasks" / f"task_{task_index:02d}"
    task_root.mkdir(parents=True, exist_ok=False)
    before = budget.snapshot()
    panels = _release_task_support_panels(
        prepared=prepared, task=task, task_index=task_index
    )
    discovery, discovery_batch, discovery_receipt = _execute_task_visual_batch(
        task_root,
        stage="discovery",
        prepared=prepared,
        task=task,
        panels=panels,
        registry=None,
        runtime=runtime,
        parallel_workers=parallel_workers,
        transport=named_image_transport,
        budget=budget,
    )
    role_rows, role_reveal, role_reveal_receipt = _freeze_task_role_reveal(
        prepared=prepared,
        task=task,
        panels=panels,
        discovery_batch=discovery_batch,
    )
    semantic_prepared, semantic_prepared_record, semantic_prepared_receipt = (
        _freeze_task_semantic_prepared(
            prepared=prepared,
            task=task,
            discovery_batch=discovery_batch,
            role_reveal=role_reveal,
            discovery_artifacts=discovery,
            role_rows=role_rows,
        )
    )
    semantic_proposal, semantic_registry, semantic_proposal_record, semantic_proposal_receipt = (
        _execute_task_semantic_proposal(
            task_root,
            prepared=prepared,
            task=task,
            task_index=task_index,
            runtime=runtime,
            semantic_prepared_record=semantic_prepared_record,
            semantic_prepared=semantic_prepared,
            discovery_artifacts=discovery,
            role_rows=role_rows,
            text_transport=text_transport,
            budget=budget,
        )
    )
    registry, registry_record, registry_receipt = _freeze_task_registry(
        prepared=prepared,
        task=task,
        discovery_batch=discovery_batch,
        role_reveal=role_reveal,
        semantic_prepared_record=semantic_prepared_record,
        semantic_proposal_record=semantic_proposal_record,
        semantic_proposal=semantic_proposal,
        registry=semantic_registry,
        discovery_artifacts=discovery,
        role_rows=role_rows,
    )
    pass_a, pass_a_batch, pass_a_receipt = _execute_task_visual_batch(
        task_root,
        stage="registered_a",
        prepared=prepared,
        task=task,
        panels=panels,
        registry=registry,
        runtime=runtime,
        parallel_workers=parallel_workers,
        transport=named_image_transport,
        budget=budget,
    )
    pass_b, pass_b_batch, pass_b_receipt = _execute_task_visual_batch(
        task_root,
        stage="registered_b",
        prepared=prepared,
        task=task,
        panels=panels,
        registry=registry,
        runtime=runtime,
        parallel_workers=parallel_workers,
        transport=named_image_transport,
        budget=budget,
    )
    bundle, ir_record, ir_receipt = _freeze_task_ir(
        prepared=prepared,
        task=task,
        registry=registry,
        semantic_proposal=semantic_proposal,
        semantic_proposal_record=semantic_proposal_record,
        role_reveal=role_reveal,
        role_rows=role_rows,
        discovery_artifacts=discovery,
        registered_a_artifacts=pass_a,
        registered_b_artifacts=pass_b,
        discovery_batch=discovery_batch,
        registered_a_batch=pass_a_batch,
        registered_b_batch=pass_b_batch,
    )
    rank_input, rank_input_receipt, rank_result, rank_result_receipt, selected = (
        _rank_task_bundle(
            task_root,
            prepared=prepared,
            task=task,
            task_index=task_index,
            bundle=bundle,
            ir_record=ir_record,
            semantic_proposal_record=semantic_proposal_record,
            runtime=runtime,
            text_transport=text_transport,
            budget=budget,
        )
    )

    candidate_by_digest = {
        item.candidate_digest: item.to_data() for item in bundle.candidates
    }
    query_phase: ObjectBongardScenePredicateQueryPhase | None = None
    query_released: list[object] = []
    query_journal_summaries: tuple[str, ...] = ()
    query_batch: dict[str, Any] | None = None
    query_batch_receipt: object | None = None
    score_rows: tuple[dict[str, Any], dict[str, Any]]
    semantic_valid = semantic_proposal_record["semantic_proposal_valid"] is True
    expected_gap_status = _task_typed_gap_status(
        semantic_valid=semantic_valid,
        bundle=bundle,
    )
    if selected is None:
        if expected_gap_status is None:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "nonempty task version space was converted to a gap"
            )
        if rank_result.get("status") != expected_gap_status:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "task rank result differs from its failed evidence gate"
            )
        if not semantic_valid and tuple(getattr(registry, "tags")):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "invalid semantic proposal escaped its zero-tag typed gap"
            )
        score_rows = _typed_gap_score_rows()
    else:
        if not semantic_valid or expected_gap_status is not None:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "invalid semantic proposal reached ranker or query release"
            )
        selected_data = candidate_by_digest.get(selected)
        if selected_data is None:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "ranker selection is absent from the complete Python inventory"
            )
        rank_payload_digest = _address(
            getattr(rank_result_receipt, "payload_digest"),
            "persisted rank result payload digest",
        )[7:]
        freeze = ObjectBongardScenePredicateTaskFreeze.seal(
            task_id=getattr(task, "task_id"),
            task_plan_digest=getattr(task, "record_digest"),
            execution_precommit_digest=prepared.release.precommit.record_digest,
            version_space_digest=bundle.bundle_digest,
            rank_response_digest=rank_payload_digest,
            selected_predicate=selected_data,
        )

        def query_observer(side: str, released: object) -> object:
            query_released.append(released)
            return _observe_task_query(
                task_root,
                side=side,
                released=released,
                prepared=prepared,
                task=task,
                task_index=task_index,
                registry=registry,
                runtime=runtime,
                transport=named_image_transport,
                budget=budget,
            )

        query_phase = commit_and_release_object_bongard_scene_predicate_queries(
            prepared=prepared.release,
            archive=prepared.archive,
            task=task,
            freeze=freeze,
            query_observer=query_observer,
        )
        replay_object_bongard_scene_predicate_query_phase(query_phase)
        score_rows = _query_score_rows(
            bundle=bundle,
            selected_candidate_data=selected_data,
            artifacts=query_phase.query_artifacts,
        )
        if len(query_released) != QUERY_CALLS_PER_TASK:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "query release inventory differs"
            )
        query_journal_summaries = tuple(
            _cold_replay_task_query(
                task_root,
                side=side,
                released=released,
                expected_artifact=artifact,
                prepared=prepared,
                task=task,
                task_index=task_index,
                registry=registry,
                runtime=runtime,
            )
            for side, released, artifact in zip(
                ("side_0", "side_1"),
                query_released,
                query_phase.query_artifacts,
                strict=True,
            )
        )
        query_batch_record = _record(
            {
                "schema": TASK_BATCH_SCHEMA,
                "command_id": COMMAND_ID,
                "task_plan_digest": getattr(task, "record_digest"),
                "stage": "query",
                "formula_freeze": query_phase.freeze.to_data(),
                "formula_freeze_receipt": query_phase.freeze_receipt.to_data(),
                "decision_commit": query_phase.commit.to_data(),
                "decision_commit_receipt": query_phase.commit_receipt.to_data(),
                "artifacts": [item.to_data() for item in query_phase.query_artifacts],
                "release_receipts": [
                    item.to_data() for item in query_phase.query_release_receipts
                ],
                "custody_witness_receipts": [
                    item.to_data() for item in query_phase.query_custody_receipts
                ],
                "journal_summary_digests": list(query_journal_summaries),
                "score_rows": [dict(item) for item in score_rows],
                "fresh_visual_call_count": QUERY_CALLS_PER_TASK,
                "formula_frozen_and_committed_before_query_release": True,
                **_authority_data(),
            },
            "batch_digest",
        )
        query_batch, query_batch_receipt = _persist_record(
            prepared.release.store,
            object_kind="scene-task-query-batch",
            record=query_batch_record,
            digest_field="batch_digest",
        )

    replayed_semantic_proposal, replayed_semantic_registry, semantic_journal_summary = (
        _cold_replay_task_semantic_proposal(
            task_root,
            prepared=prepared,
            task=task,
            task_index=task_index,
            runtime=runtime,
            semantic_prepared_record=semantic_prepared_record,
            semantic_prepared=semantic_prepared,
            semantic_proposal_record=semantic_proposal_record,
            discovery_artifacts=discovery,
            role_rows=role_rows,
        )
    )
    if (
        replayed_semantic_proposal != semantic_proposal
        or replayed_semantic_registry != registry
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task semantic proposal changed during terminal replay"
        )
    replayed_bundle = cold_replay_object_bongard_scene_predicate_calibration_bundle(
        bundle,
        registry,
        semantic_registry_proposal=semantic_proposal,
        discovery_artifacts=discovery,
        role_rows=role_rows,
    )
    if replayed_bundle != bundle:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task version space changed during terminal replay"
        )
    support_journal_summaries = (
        *_cold_replay_task_visual_batch(
            task_root,
            stage="discovery",
            prepared=prepared,
            task=task,
            panels=panels,
            registry=None,
            runtime=runtime,
            batch=discovery_batch,
        ),
        *_cold_replay_task_visual_batch(
            task_root,
            stage="registered_a",
            prepared=prepared,
            task=task,
            panels=panels,
            registry=registry,
            runtime=runtime,
            batch=pass_a_batch,
        ),
        *_cold_replay_task_visual_batch(
            task_root,
            stage="registered_b",
            prepared=prepared,
            task=task,
            panels=panels,
            registry=registry,
            runtime=runtime,
            batch=pass_b_batch,
        ),
    )
    replayed_rank_selection = _cold_replay_task_ranker(
        task_root,
        prepared=prepared,
        task=task,
        task_index=task_index,
        runtime=runtime,
        rank_input=rank_input,
        rank_result=rank_result,
    )
    if replayed_rank_selection != selected:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "terminal rank selection replay differs"
        )
    after = budget.snapshot()
    delta = _budget_delta(before, after)
    queried = selected is not None
    expected_delta = {
        "discovery_calls": DISCOVERY_CALLS_PER_TASK,
        "semantic_proposer_calls": SEMANTIC_PROPOSER_CALLS_PER_TASK,
        "registered_a_calls": REGISTERED_A_CALLS_PER_TASK,
        "registered_b_calls": REGISTERED_B_CALLS_PER_TASK,
        "ranker_calls": int(queried),
        "query_calls": QUERY_CALLS_PER_TASK if queried else 0,
    }
    if delta != expected_delta:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task physical-call budget differs"
        )
    dependencies = {
        "discovery_batch": discovery_receipt.to_data(),
        "role_reveal": role_reveal_receipt.to_data(),
        "semantic_prepared": semantic_prepared_receipt.to_data(),
        "semantic_proposal": semantic_proposal_receipt.to_data(),
        "registry_freeze": registry_receipt.to_data(),
        "registered_a_batch": pass_a_receipt.to_data(),
        "registered_b_batch": pass_b_receipt.to_data(),
        "ir_freeze": ir_receipt.to_data(),
        "rank_input": rank_input_receipt.to_data(),
        "rank_result": rank_result_receipt.to_data(),
        "query_batch": (
            None if query_batch_receipt is None else query_batch_receipt.to_data()
        ),
    }
    task_record = _record(
        {
            "schema": TASK_RESULT_SCHEMA,
            "command_id": COMMAND_ID,
            "task_ordinal": task_index,
            "task_id": getattr(task, "task_id"),
            "task_plan_digest": getattr(task, "record_digest"),
            "execution_precommit_digest": prepared.release.precommit.record_digest,
            "support_release_receipts": [
                item.release_store_receipt.to_data() for item in panels
            ],
            "dependencies": dependencies,
            "status": (
                "evaluated"
                if queried
                else expected_gap_status
            ),
            "semantic_proposal_digest": semantic_proposal.proposal_digest,
            "semantic_proposal_status": semantic_proposal.status,
            "semantic_proposal_valid": semantic_valid,
            "selected_survivor_digest": selected,
            "bundle_digest": bundle.bundle_digest,
            "rank_result_digest": rank_result["rank_result_digest"],
            "task_formula_freeze_digest": (
                None if query_phase is None else query_phase.freeze.record_digest
            ),
            "task_decision_commit_digest": (
                None if query_phase is None else query_phase.commit.record_digest
            ),
            "score_rows": [dict(item) for item in score_rows],
            "correct_count": sum(bool(item["correct"]) for item in score_rows),
            "score_denominator_contribution": QUERY_CALLS_PER_TASK,
            "physical_call_delta": delta,
            "support_pixels_released_only_through_official_gate": True,
            "query_pixels_released_only_after_exact_formula_commit": queried,
            "typed_gap_makes_no_ranker_or_query_calls": not queried,
            "terminal_python_ir_cold_replayed": True,
            "support_journal_summary_digests": list(support_journal_summaries),
            "semantic_proposer_journal_summary_digest": semantic_journal_summary,
            "query_journal_summary_digests": list(query_journal_summaries),
            "semantic_proposer_journal_cold_replayed": True,
            "ranker_journal_cold_replayed_if_called": queried,
            "all_task_journals_cold_replayed_without_model_calls": True,
            **_authority_data(),
        },
        "task_result_digest",
    )
    result, receipt = _persist_record(
        prepared.release.store,
        object_kind="scene-task-result",
        record=task_record,
        digest_field="task_result_digest",
    )
    return _TaskOutcome(
        result,
        receipt,
        int(result["correct_count"]),
        queried,
    )


@dataclass(frozen=True, slots=True)
class VerifiedObjectBongardScenePredicateCampaign:
    output_root: Path
    result_digest: str
    replay_digest: str
    exposure_successor_digest: str
    task_result_digests: tuple[str, ...]
    correct_count: int
    denominator: int
    typed_gap_count: int
    evaluated_task_count: int
    visual_fresh_call_count: int
    semantic_proposer_fresh_call_count: int
    ranker_fresh_call_count: int


def _write_campaign_result(root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    from bongard import object_bongard_rubric_nomination_command as _durable

    raw = _canonical_mapping(value, "campaign result")
    _durable._write_once(root / RESULT_FILENAME, raw, "scene-predicate campaign result")
    restored = _durable._read_record(
        root / RESULT_FILENAME, "scene-predicate campaign result"
    )
    if restored != raw:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign result durable reload differs"
        )
    return restored


def _verified_campaign(root: Path, result: Mapping[str, Any]) -> VerifiedObjectBongardScenePredicateCampaign:
    return VerifiedObjectBongardScenePredicateCampaign(
        root,
        _address(result["result_digest"], "campaign result digest"),
        _address(result["replay"]["replay_digest"], "campaign replay digest"),
        _address(
            result["exposure_successor_digest"], "campaign exposure successor"
        ),
        tuple(result["task_result_digests"]),
        int(result["score"]["correct_count"]),
        int(result["score"]["denominator"]),
        int(result["typed_gap_count"]),
        int(result["evaluated_task_count"]),
        int(result["physical_calls"]["visual_calls"]),
        int(result["physical_calls"]["semantic_proposer_calls"]),
        int(result["physical_calls"]["ranker_calls"]),
    )


def run_object_bongard_scene_predicate_campaign_command(
    output_root: str | os.PathLike[str],
    *,
    calibration_root: str | os.PathLike[str],
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    campaign_minutes: int = DEFAULT_CAMPAIGN_MINUTES,
    minutes: int = DEFAULT_MINUTES,
    executable: str = DEFAULT_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_EXPECTED_LAUNCHER_SHA256,
    calibration_verifier: CalibrationVerifier = _default_calibration_verifier,
    named_image_transport: Callable[..., object] | None = None,
    text_transport: Callable[..., object] | None = None,
    cache_snapshotter: Callable[[], object] | None = None,
    catalog_snapshotter: Callable[[], object] | None = None,
    launcher_fingerprinter: Callable[..., Mapping[str, str]] | None = None,
    runtime_attester: Callable[..., object] | None = None,
    preregistration_path: str | os.PathLike[str] = DEFAULT_PREREGISTRATION,
    plan_path: str | os.PathLike[str] = DEFAULT_PLAN,
    descriptor_path: str | os.PathLike[str] = DEFAULT_DESCRIPTOR,
    archive_path: str | os.PathLike[str] = DEFAULT_ARCHIVE,
    split_path: str | os.PathLike[str] = DEFAULT_SPLIT,
    exposure_predecessor_path: str | os.PathLike[str] = DEFAULT_EXPOSURE_PREDECESSOR,
    exposure_observed_at: str | None = None,
) -> VerifiedObjectBongardScenePredicateCampaign:
    """Run the exact 12-task release-gated TRAIN campaign."""

    if (
        type(parallel_workers) is not int
        or not 1 <= parallel_workers <= MAX_PARALLEL_WORKERS
        or type(campaign_minutes) is not int
        or not 1 <= campaign_minutes <= 24 * 60
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign parallel worker count differs"
        )
    from bongard.codex_no_tools_preflight import attest_codex_no_tools
    from bongard.transport import (
        codex_cli_authenticated_fingerprint,
        run_codex_named_images_structured,
        run_codex_text_structured,
        snapshot_cloud_policy_cache,
        snapshot_pinned_model_catalog,
    )

    # Calibration is the only dependency touched first. Runtime attestation is
    # completed before the cohort exposure successor or output root exists.
    calibration = _verify_accepted_calibration_first(
        calibration_root, calibration_verifier
    )
    runtime, fingerprint = _create_campaign_runtime(
        minutes=minutes,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
        cache_snapshotter=cache_snapshotter or snapshot_cloud_policy_cache,
        catalog_snapshotter=catalog_snapshotter or snapshot_pinned_model_catalog,
        launcher_fingerprinter=(
            launcher_fingerprinter or codex_cli_authenticated_fingerprint
        ),
        runtime_attester=runtime_attester or attest_codex_no_tools,
    )
    runtime_record = _runtime_record(runtime, fingerprint)

    def already_verified(_root: object, **_kwargs: object) -> object:
        return calibration

    prepared = prepare_object_bongard_scene_predicate_campaign(
        output_root,
        calibration_root=calibration_root,
        calibration_verifier=already_verified,
        preregistration_path=preregistration_path,
        plan_path=plan_path,
        descriptor_path=descriptor_path,
        archive_path=archive_path,
        split_path=split_path,
        exposure_predecessor_path=exposure_predecessor_path,
        exposure_observed_at=exposure_observed_at,
        runtime_record_digest=runtime_record["runtime_digest"],
        runtime_record=runtime_record,
        parallel_workers=parallel_workers,
        campaign_minutes=campaign_minutes,
    )
    runtime_record = dict(prepared.runtime_record)
    runtime_receipt = prepared.runtime_receipt
    bindings = dict(prepared.release.precommit.runtime_source_bindings)
    if (
        bindings.get("authenticated_runtime_record")
        != runtime_record["runtime_digest"]
        or bindings.get("runtime_preexposure_custody")
        != prepared.runtime_custody_witness["custody_digest"]
        or bindings.get("runtime_preexposure_custody_receipt")
        != getattr(prepared.runtime_custody_receipt, "record_digest", None)
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "execution precommit does not bind the authenticated runtime custody"
        )

    budget = _CallBudget(
        deadline_monotonic=time.monotonic() + campaign_minutes * 60
    )
    visual_transport = named_image_transport or run_codex_named_images_structured
    rank_transport = text_transport or run_codex_text_structured
    outcomes = tuple(
        _execute_task(
            prepared=prepared,
            task=task,
            task_index=index,
            runtime=runtime,
            named_image_transport=visual_transport,
            text_transport=rank_transport,
            parallel_workers=parallel_workers,
            budget=budget,
        )
        for index, task in enumerate(prepared.plan.tasks)
    )
    queried_count = sum(item.queried for item in outcomes)
    final_budget = budget.snapshot()
    final_budget.validate_terminal(
        task_count=TASK_COUNT, completed_tasks=queried_count
    )
    if final_budget.ranker_calls != queried_count:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign ranker/query task accounting differs"
        )
    correct = sum(item.scored_correct for item in outcomes)
    replay = _record(
        {
            "schema": CAMPAIGN_REPLAY_SCHEMA,
            "command_id": COMMAND_ID,
            "execution_precommit_digest": prepared.release.precommit.record_digest,
            "exposure_successor_digest": prepared.release.successor.digest,
            "task_result_digests": [
                item.result["task_result_digest"] for item in outcomes
            ],
            "task_python_version_spaces_cold_replayed": TASK_COUNT,
            "query_formula_evaluations_recomputed": queried_count * 2,
            "support_visual_journals_cold_replayed": (
                TASK_COUNT * SUPPORT_VISUAL_CALLS_PER_TASK
            ),
            "semantic_proposer_journals_cold_replayed": TASK_COUNT,
            "semantic_registry_proposals_cold_replayed": TASK_COUNT,
            "query_visual_journals_cold_replayed": queried_count * 2,
            "ranker_journals_cold_replayed": queried_count,
            "model_calls_during_replay": 0,
            "query_pixels_created_during_replay": 0,
            **_authority_data(),
        },
        "replay_digest",
    )
    result = _record(
        {
            "schema": CAMPAIGN_RESULT_SCHEMA,
            "command_id": COMMAND_ID,
            "campaign_command_source_sha256": (
                object_bongard_scene_predicate_campaign_command_source_digest()
            ),
            "calibration_result_digest": getattr(calibration, "result_digest"),
            "batch_plan_digest": prepared.plan.record_digest,
            "execution_precommit_digest": prepared.release.precommit.record_digest,
            "exposure_predecessor_digest": prepared.release.predecessor.digest,
            "exposure_successor_digest": prepared.release.successor.digest,
            "runtime_record_receipt": runtime_receipt.to_data(),
            "runtime_custody_witness_receipt": (
                prepared.runtime_custody_receipt.to_data()
            ),
            "release_receipts": {
                "plan": prepared.release.plan_receipt.to_data(),
                "precommit": prepared.release.precommit_receipt.to_data(),
                "exposure_successor": prepared.release.exposure_receipt.to_data(),
                "authorization": prepared.release.authorization_receipt.to_data(),
            },
            "task_result_receipts": [
                item.result_receipt.to_data() for item in outcomes
            ],
            "task_result_digests": [
                item.result["task_result_digest"] for item in outcomes
            ],
            "evaluated_task_count": queried_count,
            "typed_gap_count": TASK_COUNT - queried_count,
            "score": {
                "correct_count": correct,
                "denominator": QUERY_DENOMINATOR,
                "accuracy": correct / QUERY_DENOMINATOR,
                "typed_gap_query_items_scored_incorrect_without_release": (
                    (TASK_COUNT - queried_count) * QUERY_CALLS_PER_TASK
                ),
            },
            "physical_calls": {
                "discovery_calls": final_budget.discovery_calls,
                "semantic_proposer_calls": final_budget.semantic_proposer_calls,
                "registered_a_calls": final_budget.registered_a_calls,
                "registered_b_calls": final_budget.registered_b_calls,
                "ranker_calls": final_budget.ranker_calls,
                "query_calls": final_budget.query_calls,
                "visual_calls": final_budget.visual_calls,
            },
            "execution_envelope": {
                "parallel_workers": parallel_workers,
                "campaign_wall_clock_minutes": campaign_minutes,
                "per_turn_timeout_minutes": minutes,
                "deadline_checked_before_and_after_every_physical_turn": True,
                "maximum_deadline_overrun_is_one_per_turn_timeout": True,
            },
            "replay": replay,
            "exact_denominator_includes_typed_gaps": True,
            "all_support_and_query_pixels_release_gated": True,
            **_authority_data(),
        },
        "result_digest",
    )
    result = _write_campaign_result(prepared.output_root, result)
    # Launch is successful only after a fresh reconstruction from disk replays
    # the complete release, runtime, visual journals, rankers, predicates, and
    # query scores with every physical transport forbidden.
    return verify_object_bongard_scene_predicate_campaign(
        prepared.output_root,
        calibration_root=calibration_root,
        calibration_verifier=already_verified,
        preregistration_path=preregistration_path,
        plan_path=plan_path,
        descriptor_path=descriptor_path,
        archive_path=archive_path,
        exposure_predecessor_path=exposure_predecessor_path,
    )


def _existing_campaign_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign root cannot be a symlink"
        )
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign root is unavailable"
        ) from exc
    if not root.is_dir():
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign root is not a directory"
        )
    return root


def _load_stored_record(
    store: object,
    receipt_data: object,
    label: str,
    *,
    expected_object_kind: str | None = None,
) -> tuple[dict[str, Any], object]:
    from bongard.object_bongard_release_gate import ObjectBongardWriteOnceReceipt

    receipt = ObjectBongardWriteOnceReceipt.from_data(
        _canonical_mapping(receipt_data, f"{label} receipt")
    )
    if expected_object_kind is not None and receipt.object_kind != expected_object_kind:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"stored {label} object kind differs"
        )
    path = getattr(store, "root") / receipt.relative_path
    try:
        payload = path.read_bytes()
        decoded = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"stored {label} is unavailable or malformed"
        ) from exc
    raw = _canonical_mapping(decoded, label)
    if canonical_json(raw) + b"\n" != payload:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"stored {label} is not canonical"
        )
    restored = store.verify(receipt, expected_data=raw)
    top_level_identities = {
        raw.get(name)
        for name in (
            "record_digest",
            "ledger_digest",
            "runtime_digest",
            "custody_digest",
            "batch_digest",
            "registry_freeze_digest",
            "ir_freeze_digest",
            "rank_input_digest",
            "rank_result_digest",
            "task_result_digest",
        )
        if isinstance(raw.get(name), str)
    }
    if dict(restored) != raw or receipt.object_digest not in top_level_identities:
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"stored {label} receipt replay or object identity differs"
        )
    return raw, receipt


def _validate_self_sealed_record(
    raw: Mapping[str, Any],
    *,
    schema: str,
    digest_field: str,
    label: str,
) -> None:
    body = {key: item for key, item in raw.items() if key != digest_field}
    if (
        raw.get("schema") != schema
        or raw.get(digest_field) != "sha256:" + canonical_digest(body)
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            f"{label} self-seal differs"
        )


def _restore_campaign_runtime(record: Mapping[str, Any]) -> object:
    from bongard.codex_no_tools_preflight import CodexNoToolsAttestation
    from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
    from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
    from bongard.transport import CloudPolicyCacheSnapshot, CodexModelCatalogSnapshot

    raw = _canonical_mapping(record, "campaign runtime")
    _validate_self_sealed_record(
        raw,
        schema=CAMPAIGN_RUNTIME_SCHEMA,
        digest_field="runtime_digest",
        label="campaign runtime",
    )
    expected_fields = {
        "schema",
        "command_id",
        "runtime_binding",
        "cloud_policy_cache_snapshot_base64",
        "model_catalog_snapshot_base64",
        "no_tools_attestation",
        "launcher_fingerprint",
        "persisted_before_support_release",
        *_authority_data(),
        "runtime_digest",
    }
    if (
        set(raw) != expected_fields
        or raw.get("command_id") != COMMAND_ID
        or raw.get("persisted_before_support_release") is not True
        or any(raw.get(key) != item for key, item in _authority_data().items())
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign runtime policy differs"
        )

    def decode(value: object, label: str) -> bytes | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ObjectBongardScenePredicateCampaignCommandError(
                f"{label} snapshot differs"
            )
        try:
            result = base64.b64decode(value.encode("ascii"), validate=True)
        except (UnicodeError, ValueError) as exc:
            raise ObjectBongardScenePredicateCampaignCommandError(
                f"{label} snapshot differs"
            ) from exc
        if base64.b64encode(result).decode("ascii") != value:
            raise ObjectBongardScenePredicateCampaignCommandError(
                f"{label} snapshot is not canonical"
            )
        return result

    binding = _canonical_mapping(raw.get("runtime_binding"), "runtime binding")
    catalog_bytes = decode(raw.get("model_catalog_snapshot_base64"), "model catalog")
    if catalog_bytes is None:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign model catalog is absent"
        )
    cache_bytes = decode(
        raw.get("cloud_policy_cache_snapshot_base64"), "policy cache"
    )
    cache_present = binding.get("cloud_policy_cache_snapshot_present")
    if type(cache_present) is not bool:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign policy-cache presence binding differs"
        )
    runtime = ObjectBongardTurnRuntime(
        model=binding["model"],
        reasoning_effort=binding["reasoning_effort"],
        minutes=binding["minutes"],
        verbose=binding["verbose"],
        executable=binding["executable"],
        cloud_policy_cache_snapshot=(
            CloudPolicyCacheSnapshot(cache_bytes) if cache_present else None
        ),
        model_catalog_snapshot=CodexModelCatalogSnapshot(catalog_bytes),
        expected_launcher_digest=binding["expected_launcher_digest"],
        no_tools_attestation=CodexNoToolsAttestation.from_mapping(
            raw["no_tools_attestation"]
        ),
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    if runtime.binding != binding:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign runtime binding differs on reconstruction"
        )
    return runtime


def _reachable_store_object_paths(store: object, root_value: object) -> set[str]:
    """Follow every typed write-once receipt reachable from the result graph."""

    from bongard.object_bongard_release_gate import ObjectBongardWriteOnceReceipt

    found: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            try:
                receipt = ObjectBongardWriteOnceReceipt.from_data(value)
            except Exception:
                for child in value.values():
                    visit(child)
                return
            if receipt.relative_path in found:
                return
            found.add(receipt.relative_path)
            raw, _ = _load_stored_record(
                store, value, f"reachable object {receipt.object_kind}"
            )
            visit(raw)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(root_value)
    return found


def _validate_task_result_record(value: object) -> dict[str, Any]:
    raw = _canonical_mapping(value, "task result")
    expected_fields = {
        "schema",
        "command_id",
        "task_ordinal",
        "task_id",
        "task_plan_digest",
        "execution_precommit_digest",
        "support_release_receipts",
        "dependencies",
        "status",
        "semantic_proposal_digest",
        "semantic_proposal_status",
        "semantic_proposal_valid",
        "selected_survivor_digest",
        "bundle_digest",
        "rank_result_digest",
        "task_formula_freeze_digest",
        "task_decision_commit_digest",
        "score_rows",
        "correct_count",
        "score_denominator_contribution",
        "physical_call_delta",
        "support_pixels_released_only_through_official_gate",
        "query_pixels_released_only_after_exact_formula_commit",
        "typed_gap_makes_no_ranker_or_query_calls",
        "terminal_python_ir_cold_replayed",
        "support_journal_summary_digests",
        "semantic_proposer_journal_summary_digest",
        "query_journal_summary_digests",
        "semantic_proposer_journal_cold_replayed",
        "ranker_journal_cold_replayed_if_called",
        "all_task_journals_cold_replayed_without_model_calls",
        *_authority_data(),
        "task_result_digest",
    }
    body = {key: item for key, item in raw.items() if key != "task_result_digest"}
    rows = raw.get("score_rows")
    delta = raw.get("physical_call_delta")
    dependencies = raw.get("dependencies")
    support_summaries = raw.get("support_journal_summary_digests")
    query_summaries = raw.get("query_journal_summary_digests")
    selected = raw.get("selected_survivor_digest")
    queried = selected is not None
    semantic_valid = raw.get("semantic_proposal_valid") is True
    status = raw.get("status")
    status_matches_semantic_state = (
        status == "evaluated"
        if queried
        else (
            status == TYPED_SEMANTIC_PROPOSAL_GAP
            if not semantic_valid
            else status in (
                TYPED_LANGUAGE_GAP,
                TYPED_SELECTIVITY_GAP,
                TYPED_GROUNDING_REPEATABILITY_GAP,
            )
        )
    )
    expected_delta = {
        "discovery_calls": DISCOVERY_CALLS_PER_TASK,
        "semantic_proposer_calls": SEMANTIC_PROPOSER_CALLS_PER_TASK,
        "registered_a_calls": REGISTERED_A_CALLS_PER_TASK,
        "registered_b_calls": REGISTERED_B_CALLS_PER_TASK,
        "ranker_calls": int(queried),
        "query_calls": QUERY_CALLS_PER_TASK if queried else 0,
    }
    expected_score_row_fields = {
        "side",
        "query_artifact_digest",
        "query_observation_digest",
        "expected_disposition",
        "actual_disposition",
        "correct",
        "indeterminate_or_error_scores_incorrect",
    }
    if (
        set(raw) != expected_fields
        or raw.get("schema") != TASK_RESULT_SCHEMA
        or raw.get("command_id") != COMMAND_ID
        or raw.get("task_result_digest") != "sha256:" + canonical_digest(body)
        or type(raw.get("task_ordinal")) is not int
        or not 0 <= raw["task_ordinal"] < TASK_COUNT
        or not isinstance(rows, list)
        or len(rows) != QUERY_CALLS_PER_TASK
        or any(not isinstance(item, Mapping) for item in rows)
        or any(set(item) != expected_score_row_fields for item in rows)
        or [item.get("side") for item in rows] != ["side_0", "side_1"]
        or any(type(item.get("correct")) is not bool for item in rows)
        or any(
            type(item.get("indeterminate_or_error_scores_incorrect")) is not bool
            for item in rows
        )
        or type(raw.get("correct_count")) is not int
        or raw.get("correct_count") != sum(item["correct"] for item in rows)
        or type(raw.get("score_denominator_contribution")) is not int
        or raw["score_denominator_contribution"] != QUERY_CALLS_PER_TASK
        or type(raw.get("semantic_proposal_valid")) is not bool
        or raw.get("semantic_proposal_status")
        != ("proposed" if semantic_valid else "typed_proposal_gap")
        or (queried and not semantic_valid)
        or _RAW_DIGEST.fullmatch(str(raw.get("semantic_proposal_digest"))) is None
        or not status_matches_semantic_state
        or not isinstance(delta, Mapping)
        or set(delta) != set(expected_delta)
        or any(type(delta.get(key)) is not int for key in expected_delta)
        or dict(delta) != expected_delta
        or not isinstance(dependencies, Mapping)
        or not isinstance(support_summaries, list)
        or len(support_summaries) != SUPPORT_VISUAL_CALLS_PER_TASK
        or any(_ADDRESS.fullmatch(str(item)) is None for item in support_summaries)
        or not isinstance(query_summaries, list)
        or len(query_summaries) != (QUERY_CALLS_PER_TASK if queried else 0)
        or any(_ADDRESS.fullmatch(str(item)) is None for item in query_summaries)
        or set(dependencies)
        != {
            "discovery_batch",
            "role_reveal",
            "semantic_prepared",
            "semantic_proposal",
            "registry_freeze",
            "registered_a_batch",
            "registered_b_batch",
            "ir_freeze",
            "rank_input",
            "rank_result",
            "query_batch",
        }
        or (dependencies["query_batch"] is None) is queried
        or (raw.get("task_formula_freeze_digest") is None) is queried
        or (raw.get("task_decision_commit_digest") is None) is queried
        or raw.get("typed_gap_makes_no_ranker_or_query_calls") is queried
        or raw.get("ranker_journal_cold_replayed_if_called") is not queried
        or raw.get("semantic_proposer_journal_cold_replayed") is not True
        or _ADDRESS.fullmatch(
            str(raw.get("semantic_proposer_journal_summary_digest"))
        )
        is None
        or raw.get("all_task_journals_cold_replayed_without_model_calls") is not True
        or raw.get("support_pixels_released_only_through_official_gate") is not True
        or raw.get("query_pixels_released_only_after_exact_formula_commit")
        is not queried
        or raw.get("terminal_python_ir_cold_replayed") is not True
        or any(raw.get(key) != item for key, item in _authority_data().items())
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task result policy or budget differs"
        )
    if queried:
        _raw_digest(selected, "task selected survivor")
        _address(raw["task_formula_freeze_digest"], "task formula freeze")
        _address(raw["task_decision_commit_digest"], "task decision commit")
    elif rows != [dict(item) for item in _typed_gap_score_rows()]:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "typed gap score rows differ from the canonical no-query rows"
        )
    return raw


def _verify_task_from_store(
    *,
    prepared: PreparedObjectBongardScenePredicateCampaign,
    task: object,
    task_index: int,
    runtime: object,
    task_result: Mapping[str, Any],
) -> tuple[int, bool]:
    from bongard.object_bongard_scene_predicate_ir import (
        SCENE_CALIBRATION_BUNDLE_SCHEMA,
        ScenePredicateCalibrationBundle,
        ScenePredicateCandidate,
        ScenePredicateLanguage,
        build_object_bongard_scene_predicate_calibration_bundle,
        cold_replay_object_bongard_scene_predicate_calibration_bundle,
    )
    from bongard.object_scene_semantic_registry import (
        ObjectScenePreparedSemanticRegistryProposal,
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
        prepare_object_scene_semantic_registry_proposal,
        verify_object_scene_semantic_registry_proposal,
    )
    from bongard.object_scene_visual_frontend import (
        ObjectSceneSoftTagRegistry,
        ObjectSceneTranscriptArtifact,
        extract_object_scene_proposal_inventory,
    )
    from bongard.official_panel_archive import ReleasedOfficialPanel

    store = prepared.release.store
    if (
        task_result["task_ordinal"] != task_index
        or task_result["task_id"] != getattr(task, "task_id")
        or task_result["task_plan_digest"] != getattr(task, "record_digest")
        or task_result["execution_precommit_digest"]
        != prepared.release.precommit.record_digest
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task result does not bind the exact preregistered task"
        )

    expected_support_ids = (
        *tuple(getattr(task, "side_0_support_panel_ids")),
        *tuple(getattr(task, "side_1_support_panel_ids")),
    )
    support_receipts = task_result.get("support_release_receipts")
    if not isinstance(support_receipts, list) or len(support_receipts) != len(
        expected_support_ids
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task support release inventory differs"
        )
    panel_rows: list[_TaskSupportPanel] = []
    for ordinal, (receipt_data, expected_panel_id) in enumerate(
        zip(support_receipts, expected_support_ids, strict=True)
    ):
        released_raw, released_store_receipt = _load_stored_record(
            store,
            receipt_data,
            f"released support panel {task_index}:{ordinal}",
            expected_object_kind="released-support-panel",
        )
        released = ReleasedOfficialPanel.from_data(released_raw)
        released.cold_verify(
            prepared.archive,
            expected_execution_precommit_digest=(
                prepared.release.precommit.record_digest
            ),
            expected_exposure_successor_digest=prepared.release.successor.digest,
        )
        if (
            released.panel_id != expected_panel_id
            or released_store_receipt.object_digest != released.record_digest
        ):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "released support panel differs from the exact task plan"
            )
        inventory = extract_object_scene_proposal_inventory(
            released.exact_png_bytes
        )
        blind = f"support_panel_{ordinal:02d}"
        panel_rows.append(
            _TaskSupportPanel(
                ordinal,
                blind,
                f"{getattr(task, 'family')}_scene_{task_index:02d}_{ordinal:02d}",
                0 if ordinal < 6 else 1,
                released,
                released_store_receipt,
                inventory,
                _support_commitment(
                    ordinal=ordinal,
                    blind_panel_id=blind,
                    released=released,
                    inventory=inventory,
                ),
            )
        )
    panels = tuple(panel_rows)
    task_root = prepared.output_root / "tasks" / f"task_{task_index:02d}"

    dependencies = task_result["dependencies"]
    dependency_specs = (
        ("discovery_batch", "discovery batch", "scene-discovery-batch"),
        ("role_reveal", "role reveal", "scene-task-role-reveal"),
        (
            "semantic_prepared",
            "semantic prepared input",
            "scene-task-semantic-prepared",
        ),
        (
            "semantic_proposal",
            "semantic proposal",
            "scene-task-semantic-proposal",
        ),
        ("registry_freeze", "registry freeze", "scene-task-registry"),
        ("registered_a_batch", "registered A batch", "scene-registered-a-batch"),
        ("registered_b_batch", "registered B batch", "scene-registered-b-batch"),
        ("ir_freeze", "IR freeze", "scene-task-ir"),
        ("rank_input", "rank input", "scene-task-rank-input"),
        ("rank_result", "rank result", "scene-task-rank-result"),
    )
    loaded: dict[str, tuple[dict[str, Any], object]] = {}
    for key, label, kind in dependency_specs:
        loaded[key] = _load_stored_record(
            store,
            dependencies[key],
            label,
            expected_object_kind=kind,
        )
    discovery = loaded["discovery_batch"][0]
    role_reveal = loaded["role_reveal"][0]
    semantic_prepared_record = loaded["semantic_prepared"][0]
    semantic_proposal_record = loaded["semantic_proposal"][0]
    registry_record = loaded["registry_freeze"][0]
    pass_a = loaded["registered_a_batch"][0]
    pass_b = loaded["registered_b_batch"][0]
    ir_record = loaded["ir_freeze"][0]
    rank_input = loaded["rank_input"][0]
    rank_result, rank_result_receipt = loaded["rank_result"]
    for batch, stage in (
        (discovery, "discovery"),
        (pass_a, "registered_a"),
        (pass_b, "registered_b"),
    ):
        _validate_self_sealed_record(
            batch,
            schema=TASK_BATCH_SCHEMA,
            digest_field="batch_digest",
            label=f"{stage} batch",
        )
    for raw, schema, field, label in (
        (role_reveal, TASK_ROLE_REVEAL_SCHEMA, "role_reveal_digest", "role reveal"),
        (
            semantic_prepared_record,
            TASK_SEMANTIC_PREPARED_SCHEMA,
            "semantic_prepared_digest",
            "semantic prepared input",
        ),
        (
            semantic_proposal_record,
            TASK_SEMANTIC_PROPOSAL_SCHEMA,
            "semantic_proposal_result_digest",
            "semantic proposal",
        ),
        (registry_record, TASK_REGISTRY_SCHEMA, "registry_freeze_digest", "registry freeze"),
        (ir_record, TASK_IR_SCHEMA, "ir_freeze_digest", "IR freeze"),
        (rank_input, TASK_RANK_INPUT_SCHEMA, "rank_input_digest", "rank input"),
        (rank_result, TASK_RANK_RESULT_SCHEMA, "rank_result_digest", "rank result"),
    ):
        _validate_self_sealed_record(
            raw, schema=schema, digest_field=field, label=label
        )
    role_reveal_fields = {
        "schema",
        "command_id",
        "task_plan_digest",
        "discovery_batch_digest",
        "rows",
        "revealed_after_blind_discovery_batch_was_durable",
        "semantic_proposer_calls_after_reveal",
        "registered_visual_calls_after_reveal",
        "registered_visual_evaluators_receive_roles",
        *_authority_data(),
        "role_reveal_digest",
    }
    semantic_prepared_fields = {
        "schema",
        "command_id",
        "task_plan_digest",
        "discovery_batch_digest",
        "role_reveal_digest",
        "prepared_input",
        "preparation_digest",
        "prepared_input_persisted_before_zero_image_proposer_call",
        *_authority_data(),
        "semantic_prepared_digest",
    }
    semantic_proposal_fields = {
        "schema",
        "command_id",
        "task_plan_digest",
        "semantic_prepared_digest",
        "semantic_proposal",
        "semantic_proposal_digest",
        "semantic_proposal_status",
        "semantic_proposal_valid",
        "semantic_registry",
        "semantic_registry_digest",
        "proposer_payload",
        "proposer_receipt",
        "proposer_receipt_digest",
        "proposer_journal_directory",
        "proposer_journal_summary_digest",
        "proposer_fresh_call_count",
        "proposer_reused_call_count",
        "quarantined_concept_count",
        "quarantined_concept_digests",
        "invalid_optional_rows_do_not_discard_valid_concepts_when_each_orientation_retains_one",
        "orientation_coverage_gap_suppresses_otherwise_valid_concepts_from_registry",
        "structural_or_zero_orientation_payload_becomes_zero_tag_typed_gap",
        *_authority_data(),
        "semantic_proposal_result_digest",
    }
    registry_fields = {
        "schema",
        "command_id",
        "task_plan_digest",
        "discovery_batch_digest",
        "role_reveal_digest",
        "semantic_prepared_digest",
        "semantic_proposal_result_digest",
        "semantic_proposal_digest",
        "semantic_proposal_status",
        "registry",
        "registry_digest",
        "registry_derivation_mode",
        "orientation_membership_discarded_before_registered_visual_calls",
        "registered_visual_evaluators_receive_roles",
        "persisted_and_reloaded_before_registered_pass_a",
        *_authority_data(),
        "registry_freeze_digest",
    }
    ir_fields = {
        "schema",
        "command_id",
        "task_plan_digest",
        "discovery_batch_digest",
        "registered_a_batch_digest",
        "registered_b_batch_digest",
        "role_reveal_digest",
        "semantic_proposal_result_digest",
        "semantic_proposal_digest",
        "semantic_proposal_status",
        "role_rows",
        "roles_revealed_after_discovery_before_semantic_proposer",
        "registered_visual_passes_were_role_blind",
        "bundle",
        "bundle_digest",
        "model_calls_during_python_build_or_replay",
        *_authority_data(),
        "ir_freeze_digest",
    }
    rank_input_fields = {
        "schema",
        "command_id",
        "task_plan_digest",
        "ir_freeze_digest",
        "bundle_digest",
        "semantic_proposal_result_digest",
        "semantic_proposal_digest",
        "semantic_proposal_status",
        "semantic_proposal_valid",
        "typed_gap_status",
        "complete_survivor_digests",
        "ranker_slate",
        "omitted_survivors",
        "ranker_input_frozen_before_call",
        *_authority_data(),
        "rank_input_digest",
    }
    gap_rank_result_fields = {
        "schema",
        "command_id",
        "task_plan_digest",
        "rank_input_digest",
        "status",
        "ranker_called",
        "ranker_fresh_call_count",
        "selected_survivor_digest",
        "ranker_payload",
        "ranker_journal_directory",
        *_authority_data(),
        "rank_result_digest",
    }
    selected_rank_result_fields = gap_rank_result_fields | {
        "ranker_journal_summary_digest"
    }
    nested_semantic_proposal = semantic_proposal_record.get("semantic_proposal")
    nested_dropped_concepts = (
        nested_semantic_proposal.get("dropped_concepts")
        if isinstance(nested_semantic_proposal, Mapping)
        else None
    )
    nested_drop_digests = (
        [
            item.get("drop_digest") if isinstance(item, Mapping) else None
            for item in nested_dropped_concepts
        ]
        if isinstance(nested_dropped_concepts, list)
        else None
    )
    if (
        set(role_reveal) != role_reveal_fields
        or set(semantic_prepared_record) != semantic_prepared_fields
        or set(semantic_proposal_record) != semantic_proposal_fields
        or set(registry_record) != registry_fields
        or set(ir_record) != ir_fields
        or set(rank_input) != rank_input_fields
        or set(rank_result)
        != (
            selected_rank_result_fields
            if rank_result.get("ranker_called") is True
            else gap_rank_result_fields
        )
        or role_reveal.get("revealed_after_blind_discovery_batch_was_durable")
        is not True
        or role_reveal.get("semantic_proposer_calls_after_reveal")
        != SEMANTIC_PROPOSER_CALLS_PER_TASK
        or role_reveal.get("registered_visual_calls_after_reveal")
        != REGISTERED_A_CALLS_PER_TASK + REGISTERED_B_CALLS_PER_TASK
        or role_reveal.get("registered_visual_evaluators_receive_roles") is not False
        or semantic_prepared_record.get(
            "prepared_input_persisted_before_zero_image_proposer_call"
        )
        is not True
        or semantic_proposal_record.get("proposer_fresh_call_count") != 1
        or semantic_proposal_record.get("proposer_reused_call_count") != 0
        or semantic_proposal_record.get(
            "invalid_optional_rows_do_not_discard_valid_concepts_when_each_orientation_retains_one"
        )
        is not True
        or semantic_proposal_record.get(
            "orientation_coverage_gap_suppresses_otherwise_valid_concepts_from_registry"
        )
        is not True
        or semantic_proposal_record.get(
            "structural_or_zero_orientation_payload_becomes_zero_tag_typed_gap"
        )
        is not True
        or semantic_proposal_record.get("quarantined_concept_count")
        != (
            len(nested_dropped_concepts)
            if isinstance(nested_dropped_concepts, list)
            else None
        )
        or semantic_proposal_record.get("quarantined_concept_digests")
        != nested_drop_digests
        or registry_record.get(
            "orientation_membership_discarded_before_registered_visual_calls"
        )
        is not True
        or registry_record.get("registry_derivation_mode")
        != ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        or registry_record.get("registered_visual_evaluators_receive_roles") is not False
        or registry_record.get(
            "persisted_and_reloaded_before_registered_pass_a"
        )
        is not True
        or ir_record.get("roles_revealed_after_discovery_before_semantic_proposer")
        is not True
        or ir_record.get("registered_visual_passes_were_role_blind") is not True
        or ir_record.get("model_calls_during_python_build_or_replay") != 0
        or rank_input.get("ranker_input_frozen_before_call") is not True
        or any(
            record.get("command_id") != COMMAND_ID
            for record in (
                role_reveal,
                semantic_prepared_record,
                semantic_proposal_record,
                registry_record,
                ir_record,
                rank_input,
                rank_result,
            )
        )
        or any(
            record.get(key) != value
            for record in (
                role_reveal,
                semantic_prepared_record,
                semantic_proposal_record,
                registry_record,
                ir_record,
                rank_input,
                rank_result,
            )
            for key, value in _authority_data().items()
        )
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task registry/IR/rank wrapper fields differ"
        )
    expected_roles = [dict(item) for item in _task_role_rows(panels)]
    discovery_artifacts = _restore_task_visual_batch(
        discovery,
        stage="discovery",
        task=task,
        panels=panels,
        registry=None,
    )
    semantic_prepared = ObjectScenePreparedSemanticRegistryProposal.from_data(
        semantic_prepared_record["prepared_input"]
    )
    rebuilt_semantic_prepared = prepare_object_scene_semantic_registry_proposal(
        discovery_artifacts, expected_roles
    )
    semantic_proposal, semantic_registry = _restore_task_semantic_proposal(
        semantic_proposal_record,
        task=task,
        semantic_prepared_record=semantic_prepared_record,
        semantic_prepared=semantic_prepared,
        discovery_artifacts=discovery_artifacts,
        role_rows=expected_roles,
    )
    registry = ObjectSceneSoftTagRegistry.from_data(registry_record["registry"])
    artifacts = (
        discovery_artifacts,
        _restore_task_visual_batch(
            pass_a,
            stage="registered_a",
            task=task,
            panels=panels,
            registry=registry,
        ),
        _restore_task_visual_batch(
            pass_b,
            stage="registered_b",
            task=task,
            panels=panels,
            registry=registry,
        ),
    )
    if (
        semantic_prepared != rebuilt_semantic_prepared
        or semantic_prepared_record
        != _semantic_prepared_record(
            task=task,
            discovery_batch=discovery,
            role_reveal=role_reveal,
            semantic_prepared=rebuilt_semantic_prepared,
        )
        or role_reveal.get("task_plan_digest") != getattr(task, "record_digest")
        or role_reveal.get("discovery_batch_digest") != discovery["batch_digest"]
        or role_reveal.get("rows") != expected_roles
        or registry != semantic_registry
        or registry_record["task_plan_digest"] != getattr(task, "record_digest")
        or registry_record["discovery_batch_digest"] != discovery["batch_digest"]
        or registry_record["role_reveal_digest"]
        != role_reveal["role_reveal_digest"]
        or registry_record["semantic_prepared_digest"]
        != semantic_prepared_record["semantic_prepared_digest"]
        or registry_record["semantic_proposal_result_digest"]
        != semantic_proposal_record["semantic_proposal_result_digest"]
        or registry_record["semantic_proposal_digest"]
        != semantic_proposal.proposal_digest
        or registry_record["semantic_proposal_status"] != semantic_proposal.status
        or registry_record["registry_digest"] != registry.registry_digest
        or ir_record["task_plan_digest"] != getattr(task, "record_digest")
        or ir_record["discovery_batch_digest"] != discovery["batch_digest"]
        or ir_record["registered_a_batch_digest"] != pass_a["batch_digest"]
        or ir_record["registered_b_batch_digest"] != pass_b["batch_digest"]
        or ir_record["role_reveal_digest"] != role_reveal["role_reveal_digest"]
        or ir_record["semantic_proposal_result_digest"]
        != semantic_proposal_record["semantic_proposal_result_digest"]
        or ir_record["semantic_proposal_digest"] != semantic_proposal.proposal_digest
        or ir_record["semantic_proposal_status"] != semantic_proposal.status
        or ir_record["role_rows"] != expected_roles
        or task_result["semantic_proposal_digest"]
        != semantic_proposal.proposal_digest
        or task_result["semantic_proposal_status"] != semantic_proposal.status
        or task_result["semantic_proposal_valid"]
        is not (semantic_proposal.status == "proposed")
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task visual/registry/IR parent binding differs"
        )
    verify_object_scene_semantic_registry_proposal(
        semantic_proposal, registry, discovery_artifacts, expected_roles
    )
    persisted_support_summaries = [
        row["journal_summary_digest"]
        for batch in (discovery, pass_a, pass_b)
        for row in batch["rows"]
    ]
    replayed_support_summaries = [
        *_cold_replay_task_visual_batch(
            task_root,
            stage="discovery",
            prepared=prepared,
            task=task,
            panels=panels,
            registry=None,
            runtime=runtime,
            batch=discovery,
        ),
        *_cold_replay_task_visual_batch(
            task_root,
            stage="registered_a",
            prepared=prepared,
            task=task,
            panels=panels,
            registry=registry,
            runtime=runtime,
            batch=pass_a,
        ),
        *_cold_replay_task_visual_batch(
            task_root,
            stage="registered_b",
            prepared=prepared,
            task=task,
            panels=panels,
            registry=registry,
            runtime=runtime,
            batch=pass_b,
        ),
    ]
    if (
        persisted_support_summaries != replayed_support_summaries
        or replayed_support_summaries
        != task_result["support_journal_summary_digests"]
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "support journals differ on disk cold replay"
        )
    replayed_semantic_proposal, replayed_registry, semantic_summary = (
        _cold_replay_task_semantic_proposal(
            task_root,
            prepared=prepared,
            task=task,
            task_index=task_index,
            runtime=runtime,
            semantic_prepared_record=semantic_prepared_record,
            semantic_prepared=semantic_prepared,
            semantic_proposal_record=semantic_proposal_record,
            discovery_artifacts=discovery_artifacts,
            role_rows=expected_roles,
        )
    )
    if (
        replayed_semantic_proposal != semantic_proposal
        or replayed_registry != registry
        or semantic_summary
        != task_result["semantic_proposer_journal_summary_digest"]
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "semantic proposer differs on disk cold replay"
        )

    bundle = ScenePredicateCalibrationBundle.from_data(ir_record["bundle"])
    replayed_bundle = cold_replay_object_bongard_scene_predicate_calibration_bundle(
        bundle,
        registry,
        semantic_registry_proposal=semantic_proposal,
        discovery_artifacts=artifacts[0],
        role_rows=ir_record["role_rows"],
    )
    rebuilt_bundle = build_object_bongard_scene_predicate_calibration_bundle(
        registry,
        artifacts[0],
        artifacts[1],
        artifacts[2],
        ir_record["role_rows"],
        semantic_registry_proposal=semantic_proposal,
    )
    semantic_valid = semantic_proposal.status == "proposed"
    expected_gap_status = _task_typed_gap_status(
        semantic_valid=semantic_valid,
        bundle=bundle,
    )
    expected_ranker_slate = (
        [_digest_free_task_ranker_row(item) for item in bundle.ranker_slate]
        if semantic_valid
        else []
    )
    expected_omitted = (
        [dict(item) for item in bundle.omitted_survivors]
        if semantic_valid
        else [
            {
                "candidate_digest": item,
                "reason": "mandatory_semantic_proposal_gap",
            }
            for item in bundle.complete_survivor_digests
        ]
    )
    if (
        replayed_bundle != bundle
        or rebuilt_bundle != bundle
        or ir_record["bundle"].get("schema")
        != SCENE_CALIBRATION_BUNDLE_SCHEMA
        or bundle.registry_derivation_mode
        != ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        or bundle.registry_derivation_digest != semantic_proposal.proposal_digest
        or ir_record["bundle_digest"] != bundle.bundle_digest
        or task_result["bundle_digest"] != bundle.bundle_digest
        or rank_input["task_plan_digest"] != getattr(task, "record_digest")
        or rank_input["ir_freeze_digest"] != ir_record["ir_freeze_digest"]
        or rank_input["bundle_digest"] != bundle.bundle_digest
        or rank_input["semantic_proposal_result_digest"]
        != semantic_proposal_record["semantic_proposal_result_digest"]
        or rank_input["semantic_proposal_digest"] != semantic_proposal.proposal_digest
        or rank_input["semantic_proposal_status"] != semantic_proposal.status
        or rank_input["semantic_proposal_valid"] is not semantic_valid
        or rank_input["typed_gap_status"] != expected_gap_status
        or rank_input["complete_survivor_digests"]
        != list(bundle.complete_survivor_digests)
        or rank_input["ranker_slate"] != expected_ranker_slate
        or rank_input["omitted_survivors"] != expected_omitted
        or rank_result["task_plan_digest"] != getattr(task, "record_digest")
        or rank_result["rank_input_digest"] != rank_input["rank_input_digest"]
        or task_result["rank_result_digest"] != rank_result["rank_result_digest"]
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task Python version-space/rank lineage differs"
        )
    replayed_selected = _cold_replay_task_ranker(
        task_root,
        prepared=prepared,
        task=task,
        task_index=task_index,
        runtime=runtime,
        rank_input=rank_input,
        rank_result=rank_result,
    )
    selected = task_result["selected_survivor_digest"]
    if replayed_selected != selected:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "task rank selection differs on disk cold replay"
        )
    if selected is None:
        if (
            expected_gap_status is None
            or (not semantic_valid and tuple(registry.tags))
            or rank_result.get("ranker_called") is not False
            or rank_result.get("status") != expected_gap_status
            or task_result.get("status") != expected_gap_status
            or rank_result.get("ranker_fresh_call_count") != 0
            or rank_result.get("ranker_payload") is not None
            or rank_result.get("ranker_journal_directory") is not None
            or dependencies["query_batch"] is not None
            or (task_root / JOURNAL_DIRECTORY / "ranker").exists()
            or (task_root / JOURNAL_DIRECTORY / "query").exists()
        ):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "typed gap rank/query custody differs"
            )
        return int(task_result["correct_count"]), False

    if not semantic_valid or expected_gap_status is not None:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "semantic proposal gap reached persisted ranker/query path"
        )

    query_batch, _ = _load_stored_record(
        store,
        dependencies["query_batch"],
        "query batch",
        expected_object_kind="scene-task-query-batch",
    )
    _validate_self_sealed_record(
        query_batch,
        schema=TASK_BATCH_SCHEMA,
        digest_field="batch_digest",
        label="query batch",
    )
    query_batch_fields = {
        "schema",
        "command_id",
        "task_plan_digest",
        "stage",
        "formula_freeze",
        "formula_freeze_receipt",
        "decision_commit",
        "decision_commit_receipt",
        "artifacts",
        "release_receipts",
        "custody_witness_receipts",
        "journal_summary_digests",
        "score_rows",
        "fresh_visual_call_count",
        "formula_frozen_and_committed_before_query_release",
        *_authority_data(),
        "batch_digest",
    }
    if (
        set(query_batch) != query_batch_fields
        or query_batch.get("command_id") != COMMAND_ID
        or query_batch["task_plan_digest"] != getattr(task, "record_digest")
        or query_batch["stage"] != "query"
        or query_batch["fresh_visual_call_count"] != QUERY_CALLS_PER_TASK
        or query_batch.get("formula_frozen_and_committed_before_query_release")
        is not True
        or any(
            query_batch.get(key) != value
            for key, value in _authority_data().items()
        )
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query batch policy differs"
        )
    freeze = ObjectBongardScenePredicateTaskFreeze.from_data(
        query_batch["formula_freeze"]
    )
    commit = ObjectBongardScenePredicateTaskCommit.from_data(
        query_batch["decision_commit"]
    )
    freeze_raw, freeze_receipt = _load_stored_record(
        store,
        query_batch["formula_freeze_receipt"],
        "task formula freeze",
        expected_object_kind="task-freeze",
    )
    commit_raw, commit_receipt = _load_stored_record(
        store,
        query_batch["decision_commit_receipt"],
        "task decision commit",
        expected_object_kind="task-decision-commit",
    )
    if freeze_raw != freeze.to_data() or commit_raw != commit.to_data():
        raise ObjectBongardScenePredicateCampaignCommandError(
            "durable formula freeze/commit bytes differ"
        )
    expected_query_ids = (
        getattr(task, "side_0_query_panel_id"),
        getattr(task, "side_1_query_panel_id"),
    )
    release_receipt_data = query_batch.get("release_receipts")
    custody_receipt_data = query_batch.get("custody_witness_receipts")
    artifact_data = query_batch.get("artifacts")
    if (
        not isinstance(release_receipt_data, list)
        or len(release_receipt_data) != QUERY_CALLS_PER_TASK
        or not isinstance(custody_receipt_data, list)
        or len(custody_receipt_data) != QUERY_CALLS_PER_TASK
        or not isinstance(artifact_data, list)
        or len(artifact_data) != QUERY_CALLS_PER_TASK
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query release/artifact inventory differs"
        )
    query_released: list[object] = []
    query_release_receipts: list[object] = []
    query_custody_receipts: list[object] = []
    for index, (receipt_data, expected_panel_id) in enumerate(
        zip(release_receipt_data, expected_query_ids, strict=True)
    ):
        released_raw, released_receipt = _load_stored_record(
            store,
            receipt_data,
            f"released query panel {task_index}:{index}",
            expected_object_kind="released-query-panel",
        )
        released = ReleasedOfficialPanel.from_data(released_raw)
        released.cold_verify(
            prepared.archive,
            expected_execution_precommit_digest=(
                prepared.release.precommit.record_digest
            ),
            expected_exposure_successor_digest=prepared.release.successor.digest,
        )
        if released.panel_id != expected_panel_id:
            raise ObjectBongardScenePredicateCampaignCommandError(
                "released query panel differs from sealed task query"
            )
        custody_raw, custody_receipt = _load_stored_record(
            store,
            custody_receipt_data[index],
            f"query release custody {task_index}:{index}",
            expected_object_kind="scene-query-release-custody",
        )
        _validate_self_sealed_record(
            custody_raw,
            schema=QUERY_RELEASE_CUSTODY_SCHEMA,
            digest_field="custody_digest",
            label="query release custody",
        )
        custody_fields = {
            "schema",
            "command_id",
            "task_id",
            "task_plan_digest",
            "execution_precommit_digest",
            "query_side",
            "sealed_query_panel_id",
            "formula_freeze_digest",
            "formula_freeze_payload_digest",
            "formula_freeze_receipt_digest",
            "decision_commit_digest",
            "decision_commit_payload_digest",
            "decision_commit_receipt_digest",
            "released_query_panel_digest",
            "released_query_store_receipt",
            "release_gate_verified_exact_durable_freeze_and_commit",
            "custody_witness_persisted_before_visual_observation",
            *_authority_data(),
            "custody_digest",
        }
        if (
            set(custody_raw) != custody_fields
            or custody_raw.get("command_id") != COMMAND_ID
            or custody_raw.get("task_id") != freeze.task_id
            or custody_raw.get("task_plan_digest") != freeze.task_plan_digest
            or custody_raw.get("execution_precommit_digest")
            != freeze.execution_precommit_digest
            or custody_raw.get("query_side") != f"side_{index}"
            or custody_raw.get("sealed_query_panel_id") != expected_panel_id
            or custody_raw.get("formula_freeze_digest") != freeze.record_digest
            or custody_raw.get("formula_freeze_payload_digest")
            != getattr(freeze_receipt, "payload_digest", None)
            or custody_raw.get("formula_freeze_receipt_digest")
            != getattr(freeze_receipt, "record_digest", None)
            or custody_raw.get("decision_commit_digest") != commit.record_digest
            or custody_raw.get("decision_commit_payload_digest")
            != getattr(commit_receipt, "payload_digest", None)
            or custody_raw.get("decision_commit_receipt_digest")
            != getattr(commit_receipt, "record_digest", None)
            or custody_raw.get("released_query_panel_digest")
            != released.record_digest
            or custody_raw.get("released_query_store_receipt") != receipt_data
            or custody_raw.get(
                "release_gate_verified_exact_durable_freeze_and_commit"
            )
            is not True
            or custody_raw.get(
                "custody_witness_persisted_before_visual_observation"
            )
            is not True
            or custody_receipt.object_digest != custody_raw["custody_digest"]
            or any(
                custody_raw.get(key) != value
                for key, value in _authority_data().items()
            )
        ):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "query release custody differs"
            )
        query_released.append(released)
        query_release_receipts.append(released_receipt)
        query_custody_receipts.append(custody_receipt)
    query_artifacts = tuple(
        ObjectSceneTranscriptArtifact.from_data(item) for item in artifact_data
    )
    query_summary_digests = tuple(
        _cold_replay_task_query(
            task_root,
            side=side,
            released=released,
            expected_artifact=artifact,
            prepared=prepared,
            task=task,
            task_index=task_index,
            registry=registry,
            runtime=runtime,
        )
        for side, released, artifact in zip(
            ("side_0", "side_1"),
            query_released,
            query_artifacts,
            strict=True,
        )
    )
    if list(query_summary_digests) != query_batch["journal_summary_digests"]:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query journals differ on disk cold replay"
        )
    phase = ObjectBongardScenePredicateQueryPhase(
        freeze,
        freeze_receipt,
        commit,
        commit_receipt,
        (query_artifacts[0], query_artifacts[1]),
        (query_release_receipts[0], query_release_receipts[1]),
        (query_custody_receipts[0], query_custody_receipts[1]),
    )
    replay_object_bongard_scene_predicate_query_phase(phase)
    language = ScenePredicateLanguage.from_data(bundle.version_space["language"])
    candidate = ScenePredicateCandidate.from_data(
        freeze.selected_predicate, language=language
    )
    expected_rank_payload_digest = _address(
        getattr(rank_result_receipt, "payload_digest"), "rank result payload digest"
    )[7:]
    if (
        selected not in bundle.complete_survivor_digests
        or rank_result.get("ranker_called") is not True
        or rank_result.get("status") != "selected_frozen_survivor"
        or rank_result.get("ranker_fresh_call_count") != 1
        or rank_result.get("ranker_journal_directory")
        != str(Path(JOURNAL_DIRECTORY) / "ranker")
        or rank_result.get("selected_survivor_digest") != selected
        or freeze.selected_survivor_digest != selected
        or candidate.candidate_digest != selected
        or freeze.version_space_digest != bundle.bundle_digest
        or freeze.rank_response_digest != expected_rank_payload_digest
        or freeze.record_digest != task_result["task_formula_freeze_digest"]
        or commit.record_digest != task_result["task_decision_commit_digest"]
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "rank result/formula/query commitment differs"
        )
    score_rows = _query_score_rows(
        bundle=bundle,
        selected_candidate_data=freeze.selected_predicate,
        artifacts=query_artifacts,
    )
    if (
        query_batch["score_rows"] != [dict(item) for item in score_rows]
        or task_result["score_rows"] != [dict(item) for item in score_rows]
        or query_batch["journal_summary_digests"]
        != task_result["query_journal_summary_digests"]
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "query score differs on model-free replay"
        )
    return int(task_result["correct_count"]), True


def verify_object_bongard_scene_predicate_campaign(
    output_root: str | os.PathLike[str],
    *,
    calibration_root: str | os.PathLike[str],
    calibration_verifier: CalibrationVerifier = _default_calibration_verifier,
    preregistration_path: str | os.PathLike[str] = DEFAULT_PREREGISTRATION,
    plan_path: str | os.PathLike[str] = DEFAULT_PLAN,
    descriptor_path: str | os.PathLike[str] = DEFAULT_DESCRIPTOR,
    archive_path: str | os.PathLike[str] = DEFAULT_ARCHIVE,
    exposure_predecessor_path: str | os.PathLike[str] = DEFAULT_EXPOSURE_PREDECESSOR,
) -> VerifiedObjectBongardScenePredicateCampaign:
    """Cold, model-free verification of the persisted 12-task result."""

    # Preserve the same first-touch rule as launch: rejected calibration means
    # campaign artifacts, metadata, and pixels are not inspected.
    calibration = _verify_accepted_calibration_first(
        calibration_root, calibration_verifier
    )

    from bongard import object_bongard_rubric_nomination_command as _durable
    from bongard.exposure import ExposureLedger
    from bongard.object_bongard_release_gate import (
        ObjectBongardExecutionPrecommit,
        ObjectBongardReleaseAuthorization,
        ObjectBongardReleaseStore,
        ObjectBongardWriteOnceReceipt,
        PreparedObjectBongardRelease,
        verify_prepared_object_bongard_release,
    )
    from bongard.official_panel_archive import OfficialPanelArchive
    from bongard.release import OfficialReleaseDescriptor

    root = _existing_campaign_root(output_root)
    result = _durable._read_record(
        root / RESULT_FILENAME, "scene-predicate campaign result"
    )
    body = {key: item for key, item in result.items() if key != "result_digest"}
    task_receipts = result.get("task_result_receipts")
    task_digests = result.get("task_result_digests")
    calls = result.get("physical_calls")
    score = result.get("score")
    replay = result.get("replay")
    envelope = result.get("execution_envelope")
    release_receipts = result.get("release_receipts")
    result_fields = {
        "schema",
        "command_id",
        "campaign_command_source_sha256",
        "calibration_result_digest",
        "batch_plan_digest",
        "execution_precommit_digest",
        "exposure_predecessor_digest",
        "exposure_successor_digest",
        "runtime_record_receipt",
        "runtime_custody_witness_receipt",
        "release_receipts",
        "task_result_receipts",
        "task_result_digests",
        "evaluated_task_count",
        "typed_gap_count",
        "score",
        "physical_calls",
        "execution_envelope",
        "replay",
        "exact_denominator_includes_typed_gaps",
        "all_support_and_query_pixels_release_gated",
        *_authority_data(),
        "result_digest",
    }
    replay_fields = {
        "schema",
        "command_id",
        "execution_precommit_digest",
        "exposure_successor_digest",
        "task_result_digests",
        "task_python_version_spaces_cold_replayed",
        "query_formula_evaluations_recomputed",
        "support_visual_journals_cold_replayed",
        "semantic_proposer_journals_cold_replayed",
        "semantic_registry_proposals_cold_replayed",
        "query_visual_journals_cold_replayed",
        "ranker_journals_cold_replayed",
        "model_calls_during_replay",
        "query_pixels_created_during_replay",
        *_authority_data(),
        "replay_digest",
    }
    if (
        set(result) != result_fields
        or result.get("schema") != CAMPAIGN_RESULT_SCHEMA
        or result.get("command_id") != COMMAND_ID
        or result.get("result_digest") != "sha256:" + canonical_digest(body)
        or result.get("campaign_command_source_sha256")
        != object_bongard_scene_predicate_campaign_command_source_digest()
        or result.get("calibration_result_digest")
        != getattr(calibration, "result_digest", None)
        or not isinstance(task_receipts, list)
        or len(task_receipts) != TASK_COUNT
        or not isinstance(task_digests, list)
        or len(task_digests) != TASK_COUNT
        or any(_ADDRESS.fullmatch(str(item)) is None for item in task_digests)
        or len(set(task_digests)) != TASK_COUNT
        or not isinstance(calls, Mapping)
        or not isinstance(score, Mapping)
        or set(score)
        != {
            "correct_count",
            "denominator",
            "accuracy",
            "typed_gap_query_items_scored_incorrect_without_release",
        }
        or set(calls)
        != {
            "discovery_calls",
            "semantic_proposer_calls",
            "registered_a_calls",
            "registered_b_calls",
            "ranker_calls",
            "query_calls",
            "visual_calls",
        }
        or any(type(calls.get(key)) is not int for key in calls)
        or type(score.get("correct_count")) is not int
        or type(score.get("denominator")) is not int
        or type(score.get("accuracy")) is not float
        or type(
            score.get("typed_gap_query_items_scored_incorrect_without_release")
        )
        is not int
        or not isinstance(replay, Mapping)
        or set(replay) != replay_fields
        or replay.get("schema") != CAMPAIGN_REPLAY_SCHEMA
        or replay.get("command_id") != COMMAND_ID
        or replay.get("execution_precommit_digest")
        != result.get("execution_precommit_digest")
        or replay.get("exposure_successor_digest")
        != result.get("exposure_successor_digest")
        or replay.get("task_python_version_spaces_cold_replayed") != TASK_COUNT
        or any(
            type(replay.get(key)) is not int
            for key in (
                "task_python_version_spaces_cold_replayed",
                "query_formula_evaluations_recomputed",
                "support_visual_journals_cold_replayed",
                "semantic_proposer_journals_cold_replayed",
                "semantic_registry_proposals_cold_replayed",
                "query_visual_journals_cold_replayed",
                "ranker_journals_cold_replayed",
                "model_calls_during_replay",
                "query_pixels_created_during_replay",
            )
        )
        or any(replay.get(key) != item for key, item in _authority_data().items())
        or not isinstance(envelope, Mapping)
        or set(envelope)
        != {
            "parallel_workers",
            "campaign_wall_clock_minutes",
            "per_turn_timeout_minutes",
            "deadline_checked_before_and_after_every_physical_turn",
            "maximum_deadline_overrun_is_one_per_turn_timeout",
        }
        or not isinstance(release_receipts, Mapping)
        or set(release_receipts)
        != {"plan", "precommit", "exposure_successor", "authorization"}
        or type(envelope.get("parallel_workers")) is not int
        or not 1 <= envelope["parallel_workers"] <= MAX_PARALLEL_WORKERS
        or type(envelope.get("campaign_wall_clock_minutes")) is not int
        or not 1 <= envelope["campaign_wall_clock_minutes"] <= 24 * 60
        or type(envelope.get("per_turn_timeout_minutes")) is not int
        or not 1 <= envelope["per_turn_timeout_minutes"] <= 120
        or envelope.get("deadline_checked_before_and_after_every_physical_turn")
        is not True
        or envelope.get("maximum_deadline_overrun_is_one_per_turn_timeout") is not True
        or score.get("denominator") != QUERY_DENOMINATOR
        or type(result.get("evaluated_task_count")) is not int
        or type(result.get("typed_gap_count")) is not int
        or result["evaluated_task_count"] + result["typed_gap_count"]
        != TASK_COUNT
        or replay.get("task_result_digests") != task_digests
        or replay.get("model_calls_during_replay") != 0
        or replay.get("query_pixels_created_during_replay") != 0
        or result.get("exact_denominator_includes_typed_gaps") is not True
        or result.get("all_support_and_query_pixels_release_gated") is not True
        or any(result.get(key) != item for key, item in _authority_data().items())
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign result policy differs"
        )
    replay_body = {key: item for key, item in replay.items() if key != "replay_digest"}
    if replay.get("replay_digest") != "sha256:" + canonical_digest(replay_body):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign replay digest differs"
        )
    store = ObjectBongardReleaseStore(root)
    preregistration, exact_plan = _load_exact_cohort(
        Path(preregistration_path), Path(plan_path)
    )
    plan_raw, plan_receipt = _load_stored_record(
        store,
        release_receipts["plan"],
        "batch plan",
        expected_object_kind="batch-plan",
    )
    plan = ObjectBongardBatchPlan.from_data(plan_raw)
    precommit_raw, precommit_receipt = _load_stored_record(
        store,
        release_receipts["precommit"],
        "execution precommit",
        expected_object_kind="execution-precommit",
    )
    precommit = ObjectBongardExecutionPrecommit.from_data(precommit_raw)
    predecessor = ExposureLedger.from_dict(
        _read_exact_json(
            Path(exposure_predecessor_path),
            EXPOSURE_PREDECESSOR_FILE_SHA256,
            "exposure predecessor",
        )
    )
    if predecessor.digest != EXPOSURE_PREDECESSOR_DIGEST:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "exposure predecessor digest differs during replay"
        )
    successor_raw, exposure_receipt = _load_stored_record(
        store,
        release_receipts["exposure_successor"],
        "exposure successor",
        expected_object_kind="exposure-successor",
    )
    successor = ExposureLedger.from_dict(successor_raw)
    authorization_raw, authorization_receipt = _load_stored_record(
        store,
        release_receipts["authorization"],
        "release authorization",
        expected_object_kind="release-authorization",
    )
    authorization = ObjectBongardReleaseAuthorization.from_data(authorization_raw)
    if plan != exact_plan:
        raise ObjectBongardScenePredicateCampaignCommandError(
            "persisted plan differs from the exact preregistered cohort"
        )
    descriptor = OfficialReleaseDescriptor.from_dict(
        _read_exact_json_unpinned(
            Path(descriptor_path), "official release descriptor"
        )
    )
    archive = OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    release = PreparedObjectBongardRelease(
        store,
        plan,
        precommit,
        predecessor,
        successor,
        authorization,
        plan_receipt,
        precommit_receipt,
        exposure_receipt,
        authorization_receipt,
    )
    verify_prepared_object_bongard_release(release)
    verify_object_bongard_scene_predicate_exposure_transition(
        predecessor=predecessor, plan=plan, prepared=release
    )
    if (
        descriptor.digest != plan.release_descriptor_digest
        or archive.record_digest != authorization.archive_record_digest
        or result["batch_plan_digest"] != plan.record_digest
        or result["execution_precommit_digest"] != precommit.record_digest
        or result["exposure_predecessor_digest"] != predecessor.digest
        or result["exposure_successor_digest"] != successor.digest
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign official release lineage differs"
        )
    runtime_record, runtime_receipt = _load_stored_record(
        store,
        result["runtime_record_receipt"],
        "campaign runtime",
        expected_object_kind="scene-campaign-runtime",
    )
    runtime_body = {
        key: item for key, item in runtime_record.items() if key != "runtime_digest"
    }
    runtime = _restore_campaign_runtime(runtime_record)
    runtime_custody, runtime_custody_receipt = _load_stored_record(
        store,
        result["runtime_custody_witness_receipt"],
        "campaign runtime custody witness",
        expected_object_kind="scene-campaign-runtime-custody",
    )
    _validate_self_sealed_record(
        runtime_custody,
        schema=CAMPAIGN_RUNTIME_CUSTODY_SCHEMA,
        digest_field="custody_digest",
        label="campaign runtime custody witness",
    )
    runtime_custody_fields = {
        "schema",
        "command_id",
        "runtime_digest",
        "runtime_store_receipt",
        "batch_plan_digest",
        "exposure_predecessor_digest",
        "release_descriptor_digest",
        "archive_record_digest",
        "witness_persisted_and_bound_into_precommit_before_exposure",
        *_authority_data(),
        "custody_digest",
    }
    if (
        set(runtime_custody) != runtime_custody_fields
        or runtime_custody.get("command_id") != COMMAND_ID
        or runtime_custody.get("runtime_digest") != runtime_record["runtime_digest"]
        or runtime_custody.get("runtime_store_receipt")
        != result["runtime_record_receipt"]
        or runtime_custody.get("batch_plan_digest") != plan.record_digest
        or runtime_custody.get("exposure_predecessor_digest") != predecessor.digest
        or runtime_custody.get("release_descriptor_digest") != descriptor.digest
        or runtime_custody.get("archive_record_digest") != archive.record_digest
        or runtime_custody.get(
            "witness_persisted_and_bound_into_precommit_before_exposure"
        )
        is not True
        or runtime_custody_receipt.object_digest
        != runtime_custody["custody_digest"]
        or any(
            runtime_custody.get(key) != value
            for key, value in _authority_data().items()
        )
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign pre-exposure runtime custody differs"
        )
    configuration = dict(precommit.configuration)
    source_bindings = dict(precommit.runtime_source_bindings)
    from bongard.object_bongard_scene_predicate_calibration_command import (
        object_bongard_scene_predicate_calibration_command_source_digest,
    )
    from bongard.object_bongard_scene_predicate_ir import (
        object_bongard_scene_predicate_ir_source_digest,
    )
    from bongard.object_bongard_turn_journal import (
        object_bongard_turn_journal_source_digest,
    )
    from bongard.object_scene_visual_frontend import (
        object_scene_visual_frontend_source_digest,
    )
    from bongard.object_scene_semantic_registry import (
        object_scene_semantic_registry_source_digest,
    )
    from bongard.prototype_scene_observer import (
        prototype_scene_transport_source_digest,
    )
    from bongard.transport import PINNED_CODEX_CLI_VERSION

    expected_source_bindings = {
        "campaign_command": "sha256:"
        + object_bongard_scene_predicate_campaign_command_source_digest(),
        "calibration_command": "sha256:"
        + object_bongard_scene_predicate_calibration_command_source_digest(),
        "calibration_result": getattr(calibration, "result_digest"),
        "scene_visual_frontend": "sha256:"
        + object_scene_visual_frontend_source_digest(),
        "scene_semantic_registry": "sha256:"
        + object_scene_semantic_registry_source_digest(),
        "scene_predicate_ir": "sha256:"
        + object_bongard_scene_predicate_ir_source_digest(),
        "turn_journal": "sha256:" + object_bongard_turn_journal_source_digest(),
        "transport": "sha256:" + prototype_scene_transport_source_digest(),
        "authenticated_runtime_record": runtime_record["runtime_digest"],
        "runtime_preexposure_custody": runtime_custody["custody_digest"],
        "runtime_preexposure_custody_receipt": runtime_custody_receipt.record_digest,
        **_automatic_release_source_bindings(),
    }
    expected_configuration = {
        "task_count": TASK_COUNT,
        "discovery_calls_per_task": DISCOVERY_CALLS_PER_TASK,
        "semantic_proposer_calls_per_task": SEMANTIC_PROPOSER_CALLS_PER_TASK,
        "registered_a_calls_per_task": REGISTERED_A_CALLS_PER_TASK,
        "registered_b_calls_per_task": REGISTERED_B_CALLS_PER_TASK,
        "ranker_calls_max_per_task": MAX_RANKER_CALLS_PER_TASK,
        "query_calls_per_task": QUERY_CALLS_PER_TASK,
        "score_denominator": QUERY_DENOMINATOR,
        "parallel_workers": envelope["parallel_workers"],
        "campaign_wall_clock_minutes": envelope["campaign_wall_clock_minutes"],
        "maximum_visual_calls": MAX_VISUAL_CALLS,
        "maximum_ranker_calls": MAX_RANKER_CALLS,
        "maximum_semantic_proposer_calls": MAX_SEMANTIC_PROPOSER_CALLS,
        "authenticated_runtime_persisted_before_exposure": True,
        "python_canonical": True,
        "lean_required": False,
    }
    if (
        runtime_record.get("schema") != CAMPAIGN_RUNTIME_SCHEMA
        or runtime_record.get("runtime_digest")
        != "sha256:" + canonical_digest(runtime_body)
        or runtime_receipt.object_digest != runtime_record["runtime_digest"]
        or source_bindings.get("authenticated_runtime_record")
        != runtime_record["runtime_digest"]
        or source_bindings != expected_source_bindings
        or configuration != expected_configuration
        or precommit.record_digest != result["execution_precommit_digest"]
        or runtime.binding != runtime_record["runtime_binding"]
        or runtime.minutes != envelope["per_turn_timeout_minutes"]
        or runtime.model != MODEL
        or runtime.reasoning_effort != REASONING_EFFORT
        or runtime_record.get("launcher_fingerprint")
        != {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": runtime.expected_launcher_digest,
        }
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign runtime/precommit binding differs"
        )
    prepared = PreparedObjectBongardScenePredicateCampaign(
        root,
        calibration,
        preregistration,
        plan,
        descriptor,
        archive,
        release,
        runtime_record,
        runtime_receipt,
        runtime_custody,
        runtime_custody_receipt,
    )
    correct = 0
    queried = 0
    expected_journal_roots: set[str] = set()
    for index, (receipt_data, expected_digest) in enumerate(
        zip(task_receipts, task_digests, strict=True)
    ):
        task_raw, receipt = _load_stored_record(
            store,
            receipt_data,
            f"task result {index}",
            expected_object_kind="scene-task-result",
        )
        task_result = _validate_task_result_record(task_raw)
        if (
            task_result["task_ordinal"] != index
            or task_result["task_result_digest"] != expected_digest
            or receipt.object_digest != expected_digest
        ):
            raise ObjectBongardScenePredicateCampaignCommandError(
                "campaign task result ordering differs"
            )
        for stage in ("discovery", "registered_a", "registered_b"):
            for panel_index in range(SUPPORT_PANEL_COUNT_PER_TASK):
                expected_journal_roots.add(
                    (
                        Path("tasks")
                        / f"task_{index:02d}"
                        / JOURNAL_DIRECTORY
                        / stage
                        / f"panel_{panel_index:02d}"
                    ).as_posix()
                )
        expected_journal_roots.add(
            (
                Path("tasks")
                / f"task_{index:02d}"
                / JOURNAL_DIRECTORY
                / "semantic_registry_proposer"
            ).as_posix()
        )
        if task_result["selected_survivor_digest"] is not None:
            expected_journal_roots.add(
                (
                    Path("tasks")
                    / f"task_{index:02d}"
                    / JOURNAL_DIRECTORY
                    / "ranker"
                ).as_posix()
            )
            for side in ("side_0", "side_1"):
                expected_journal_roots.add(
                    (
                        Path("tasks")
                        / f"task_{index:02d}"
                        / JOURNAL_DIRECTORY
                        / "query"
                        / side
                    ).as_posix()
                )
        task_correct, task_queried = _verify_task_from_store(
            prepared=prepared,
            task=plan.tasks[index],
            task_index=index,
            runtime=runtime,
            task_result=task_result,
        )
        correct += task_correct
        queried += int(task_queried)
    object_entries = tuple((root / "objects").rglob("*"))
    task_entries = tuple((root / "tasks").rglob("*"))
    if any(path.is_symlink() for path in (*object_entries, *task_entries)):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign store or journal tree contains a symlink"
        )
    actual_object_paths = {
        path.relative_to(root).as_posix() for path in object_entries if path.is_file()
    }
    reachable_object_paths = _reachable_store_object_paths(store, result)
    actual_journal_roots = {
        path.parent.relative_to(root).as_posix()
        for path in (root / "tasks").glob("task_*/journals/**/manifest.json")
        if path.is_file()
    }
    if (
        actual_object_paths != reachable_object_paths
        or actual_journal_roots != expected_journal_roots
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign contains unreferenced store objects or journal turns"
        )
    expected_calls = {
        "discovery_calls": TASK_COUNT * DISCOVERY_CALLS_PER_TASK,
        "semantic_proposer_calls": TASK_COUNT * SEMANTIC_PROPOSER_CALLS_PER_TASK,
        "registered_a_calls": TASK_COUNT * REGISTERED_A_CALLS_PER_TASK,
        "registered_b_calls": TASK_COUNT * REGISTERED_B_CALLS_PER_TASK,
        "ranker_calls": queried,
        "query_calls": queried * QUERY_CALLS_PER_TASK,
        "visual_calls": TASK_COUNT * SUPPORT_VISUAL_CALLS_PER_TASK
        + queried * QUERY_CALLS_PER_TASK,
    }
    if (
        dict(calls) != expected_calls
        or result["evaluated_task_count"] != queried
        or result["typed_gap_count"] != TASK_COUNT - queried
        or score["correct_count"] != correct
        or score["typed_gap_query_items_scored_incorrect_without_release"]
        != (TASK_COUNT - queried) * QUERY_CALLS_PER_TASK
        or score["accuracy"] != correct / QUERY_DENOMINATOR
        or replay.get("support_visual_journals_cold_replayed")
        != TASK_COUNT * SUPPORT_VISUAL_CALLS_PER_TASK
        or replay.get("semantic_proposer_journals_cold_replayed") != TASK_COUNT
        or replay.get("semantic_registry_proposals_cold_replayed") != TASK_COUNT
        or replay.get("query_visual_journals_cold_replayed")
        != queried * QUERY_CALLS_PER_TASK
        or replay.get("ranker_journals_cold_replayed") != queried
        or replay.get("query_formula_evaluations_recomputed")
        != queried * QUERY_CALLS_PER_TASK
    ):
        raise ObjectBongardScenePredicateCampaignCommandError(
            "campaign aggregate score or budget differs"
        )
    return _verified_campaign(root, result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Release-gated Python scene-predicate Bongard campaign"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--output-root", required=True)
    launch.add_argument("--calibration-root", required=True)
    launch.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    launch.add_argument("--campaign-minutes", type=int, default=DEFAULT_CAMPAIGN_MINUTES)
    launch.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    launch.add_argument("--executable", default=DEFAULT_EXECUTABLE)
    launch.add_argument(
        "--expected-launcher-sha256", default=DEFAULT_EXPECTED_LAUNCHER_SHA256
    )
    verify = sub.add_parser("verify")
    verify.add_argument("--output-root", required=True)
    verify.add_argument("--calibration-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "launch":
            verified = run_object_bongard_scene_predicate_campaign_command(
                args.output_root,
                calibration_root=args.calibration_root,
                parallel_workers=args.parallel_workers,
                campaign_minutes=args.campaign_minutes,
                minutes=args.minutes,
                executable=args.executable,
                expected_launcher_sha256=args.expected_launcher_sha256,
            )
        else:
            verified = verify_object_bongard_scene_predicate_campaign(
                args.output_root, calibration_root=args.calibration_root
            )
    except Exception as exc:
        print(f"scene-predicate campaign failed: {exc}", file=sys.stderr)
        return 1
    print(
        canonical_json(
            {
                "result_digest": verified.result_digest,
                "correct_count": verified.correct_count,
                "denominator": verified.denominator,
                "typed_gap_count": verified.typed_gap_count,
                "evaluated_task_count": verified.evaluated_task_count,
                "visual_fresh_call_count": verified.visual_fresh_call_count,
                "semantic_proposer_fresh_call_count": (
                    verified.semantic_proposer_fresh_call_count
                ),
                "ranker_fresh_call_count": verified.ranker_fresh_call_count,
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
