"""Sealed 12-task TRAIN campaign for prose-grounded Python predicates.

The accepted calibration directory is cold-verified before this command may
read the cohort plan, inspect the archive, create its output directory, release
panel bytes, or call a model.  The official release gate durably appends one
12-task exposure event before support bytes can be released.

For every task, twelve support panels receive one discovery observation and
two independent registered-evaluation observations.  Python constructs a
closed, conservative predicate version space.  A typed gap makes no ranker or
query calls; otherwise one zero-image ranker may select one frozen survivor.
The exact formula is durably frozen and committed before exactly two sealed
query panels are released.  Lean is absent and removable: Python artifacts are
the sole decision and replay authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Lock
from typing import Any, Callable, Mapping, Protocol, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardBatchPlan
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


COMMAND_ID = "bongard.scene-predicate-campaign/exact-unused-train-12-v1"
TASK_COUNT = 12
SUPPORT_PANEL_COUNT_PER_TASK = 12
DISCOVERY_CALLS_PER_TASK = 12
REGISTERED_A_CALLS_PER_TASK = 12
REGISTERED_B_CALLS_PER_TASK = 12
SUPPORT_VISUAL_CALLS_PER_TASK = 36
MAX_RANKER_CALLS_PER_TASK = 1
QUERY_CALLS_PER_TASK = 2
QUERY_DENOMINATOR = TASK_COUNT * QUERY_CALLS_PER_TASK
MAX_VISUAL_CALLS = TASK_COUNT * (SUPPORT_VISUAL_CALLS_PER_TASK + QUERY_CALLS_PER_TASK)
MAX_RANKER_CALLS = TASK_COUNT

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


class ObjectBongardScenePredicateCampaignCommandError(RuntimeError):
    """Calibration, custody, budget, formula freeze, or replay failed closed."""


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
    def __init__(self) -> None:
        self._lock = Lock()
        self._counts = {stage: 0 for stage in (
            "discovery", "registered_a", "registered_b", "ranker", "query"
        )}

    def count(self, stage: str, limit: int) -> None:
        with self._lock:
            if stage not in self._counts or self._counts[stage] >= limit:
                raise ObjectBongardScenePredicateCampaignCommandError(
                    f"{stage} physical-call budget exhausted"
                )
            self._counts[stage] += 1

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
            "rank_response_digest", "selected_predicate_digest",
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


@dataclass(frozen=True, slots=True)
class ObjectBongardScenePredicateQueryPhase:
    freeze: ObjectBongardScenePredicateTaskFreeze
    freeze_receipt: object
    commit: ObjectBongardScenePredicateTaskCommit
    commit_receipt: object
    query_artifacts: tuple[object, object]
    query_release_receipts: tuple[object, object]


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
        artifacts.append(query_observer(side, released))
        receipts.append(receipt)
    if len(artifacts) != QUERY_CALLS_PER_TASK or len(receipts) != QUERY_CALLS_PER_TASK:
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
        or commit.selected_predicate_digest != freeze.selected_predicate_digest
        or commit.task_freeze_digest != freeze.record_digest
        or commit.exact_freeze_payload_digest
        != getattr(phase.freeze_receipt, "payload_digest", None)
        or commit.task_freeze_store_receipt_digest
        != getattr(phase.freeze_receipt, "record_digest", None)
        or getattr(phase.commit_receipt, "object_digest", None) != commit.record_digest
        or len(phase.query_artifacts) != QUERY_CALLS_PER_TASK
        or len(phase.query_release_receipts) != QUERY_CALLS_PER_TASK
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
) -> PreparedObjectBongardScenePredicateCampaign:
    """Verify calibration, bind metadata, and persist exposure before pixels."""

    # Keep this call literally before every cohort/archive/output operation.
    calibration = _verify_accepted_calibration_first(
        calibration_root, calibration_verifier
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
    precommit = create_object_bongard_execution_precommit(
        plan=plan,
        predecessor=predecessor,
        descriptor=descriptor,
        archive=archive,
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        exact_used_task_ids=exact_used_task_ids,
        runtime_source_bindings={
            "campaign_command": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "calibration_result": calibration_result,
        },
        configuration={
            "task_count": TASK_COUNT,
            "discovery_calls_per_task": DISCOVERY_CALLS_PER_TASK,
            "registered_a_calls_per_task": REGISTERED_A_CALLS_PER_TASK,
            "registered_b_calls_per_task": REGISTERED_B_CALLS_PER_TASK,
            "ranker_calls_max_per_task": MAX_RANKER_CALLS_PER_TASK,
            "query_calls_per_task": QUERY_CALLS_PER_TASK,
            "score_denominator": QUERY_DENOMINATOR,
            "python_canonical": True,
            "lean_required": False,
        },
        exposure_observed_at=timestamp,
        exposure_actor="headless-codex-scene-predicate-campaign",
        exposure_purpose="prose-grounded-python-predicate-support-and-sealed-query",
        exposure_source=f"{COMMAND_ID}:{plan.record_digest}",
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
        root, calibration, preregistration, plan, descriptor, archive, prepared
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


def run_object_bongard_scene_predicate_campaign_command(*args: object, **kwargs: object) -> object:
    """Run the sealed campaign; task execution binds after the frozen IR lands."""

    raise ObjectBongardScenePredicateCampaignCommandError(
        "scene-predicate campaign execution is not yet bound to the frozen IR API"
    )
