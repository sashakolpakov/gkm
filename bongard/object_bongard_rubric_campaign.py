"""Production orchestration for the preregistered object-rubric campaign.

The campaign is deliberately Python-authoritative.  Codex may describe the
support panels, score a fixed prose rubric, and rank an already verified
finite version space.  Python owns every identity, admissibility decision,
freeze, query-release condition, score, and cold replay.  Lean is absent and
removable by construction.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.corpus import SplitIndex
from bongard.exposure import ExposureLedger
from bongard.object_bongard_batch import (
    ObjectBongardBatchPlan,
    ObjectBongardTaskPlan,
    object_bongard_batch_source_digest,
    object_bongard_task_inventory_digest,
    verify_object_bongard_batch_plan,
)
from bongard.object_bongard_release_gate import (
    ObjectBongardExecutionPrecommit,
    ObjectBongardReleaseAuthorization,
    ObjectBongardReleaseStore,
    ObjectBongardWriteOnceReceipt,
    PreparedObjectBongardRelease,
    create_object_bongard_execution_precommit,
    object_bongard_release_gate_source_digest,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
    prepare_object_bongard_release,
    release_object_bongard_query_panel,
    release_object_bongard_support_panel,
    verify_prepared_object_bongard_release,
)
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricObserverArtifact,
    ObjectBongardRubricSpec,
    object_bongard_rubric_observer_output_schema,
    object_bongard_rubric_observer_prompt,
    object_bongard_rubric_observer_protocol_digest,
    object_bongard_rubric_observer_source_digest,
    observe_object_bongard_rubric,
    verify_object_bongard_rubric_observer_artifact,
)
from bongard.object_bongard_rubric_ranker import (
    ObjectBongardRubricRankResponse,
    ObjectBongardRubricRanker,
    ObjectBongardRubricRankerError,
    object_bongard_rubric_rank_input_digest,
    object_bongard_rubric_ranker_output_schema,
    object_bongard_rubric_ranker_prompt,
    object_bongard_rubric_ranker_protocol_digest,
    object_bongard_rubric_ranker_source_digest,
    object_bongard_rubric_ranker_transport_source_digest,
)
from bongard.object_bongard_rubric_task_runner import (
    ObjectBongardRubricTaskFreeze,
    ObjectBongardRubricTaskFreezeCommit,
    ObjectBongardRubricTaskRunArchive,
    ObjectBongardRubricTaskRunStatus,
    cold_replay_object_bongard_rubric_task,
    object_bongard_rubric_task_runner_source_digest,
    run_object_bongard_rubric_task,
)
from bongard.object_bongard_rubric_version_space import (
    ObjectBongardRubricSupportVersionSpace,
    build_object_bongard_rubric_support_version_space,
    cold_verify_object_bongard_rubric_support_version_space,
    object_bongard_rubric_version_space_algorithm_digest,
    object_bongard_rubric_version_space_source_digest,
)
from bongard.object_bongard_semantics import (
    ObjectBongardSemanticArtifact,
    describe_object_bongard_support,
    object_bongard_semantics_prompt,
    object_bongard_semantics_protocol_digest,
    object_bongard_semantics_source_digest,
    verify_object_bongard_semantic_artifact,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTextTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
    object_bongard_turn_journal_source_digest,
)
from bongard.official_panel_archive import OfficialPanelArchive, ReleasedOfficialPanel
from bongard.prototype_object_observer_protocol import (
    prototype_object_description_output_schema,
)
from bongard.prototype_object_hypotheses import (
    ObjectHypothesisPacket,
    extract_object_hypothesis_packet,
    object_hypothesis_extractor_source_digest,
    render_object_hypothesis_atlas,
    verify_object_hypothesis_packet,
)
from bongard.prototype_object_lineages import (
    ObjectLineagePacket,
    extract_object_lineage_packet,
    object_lineage_source_digest,
    verify_object_lineage_packet,
)
from bongard.prototype_scene_observer import (
    PrototypeSceneObserverStatus,
    prototype_scene_transport_source_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.release import OfficialReleaseDescriptor
from bongard.transport import CodexStructuredResult


CAMPAIGN_ID = "bongard.object-rubric-campaign/python-release-gate-v1"
TASK_EXECUTION_SCHEMA = "gkm.bongard-object-rubric-task-execution.v1"
CAMPAIGN_SCHEMA = "gkm.bongard-object-rubric-campaign.v1"
EXPECTED_PREREG_SCHEMA = "gkm.bongard-object-batch-preregistration.v1"
EXPECTED_PREREG_SCOPE = (
    "exact-unused-train-targeted-engineering-not-official-benchmark"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ObjectBongardRubricCampaignError(RuntimeError):
    """A campaign metadata, execution, persistence, or replay check failed."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_selection_decision_scoring_or_replay": False,
    }


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardRubricCampaignError(
            f"{label} must be a sha256: content address"
        )
    return value


def _require_raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardRubricCampaignError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _verify_launch_gate_configuration(
    configuration: Mapping[str, object],
) -> None:
    """Require the immutable calibration/launcher chain on every run path."""

    values = dict(configuration)
    address_keys = (
        "launch_authorization_digest",
        "campaign_runtime_precommit_digest",
        "calibration_replay_digest",
        "calibration_observation_inventory_digest",
    )
    if any(
        not isinstance(values.get(key), str)
        or _ADDRESS.fullmatch(values[key]) is None
        for key in address_keys
    ) or (
        not isinstance(values.get("calibration_assessment_digest"), str)
        or _RAW_DIGEST.fullmatch(values["calibration_assessment_digest"])
        is None
    ):
        raise ObjectBongardRubricCampaignError(
            "campaign execution lacks a valid calibration launch gate"
        )


def _strict_json_file(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ObjectBongardRubricCampaignError(
                    f"{label} contains a duplicate JSON key"
                )
            result[key] = value
        return result
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        before = os.lstat(source)
        descriptor = os.open(source, flags)
        opened = os.fstat(descriptor)
        identity = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                opened.st_dev, opened.st_ino, opened.st_size,
                opened.st_mtime_ns, opened.st_ctime_ns,
            )
            != identity
            or not 0 < opened.st_size <= 128 * 1024 * 1024
        ):
            raise ObjectBongardRubricCampaignError(
                f"{label} is not a stable private file"
            )
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            total += len(chunk)
            if total > 128 * 1024 * 1024:
                raise ObjectBongardRubricCampaignError(
                    f"{label} exceeds its byte bound"
                )
        after = os.fstat(descriptor)
        after_path = os.lstat(source)
        if (
            total != opened.st_size
            or (
                after.st_dev, after.st_ino, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns,
            )
            != identity
            or (
                after_path.st_dev, after_path.st_ino, after_path.st_size,
                after_path.st_mtime_ns, after_path.st_ctime_ns,
            )
            != identity
        ):
            raise ObjectBongardRubricCampaignError(
                f"{label} changed during its authenticated read"
            )
        payload = b"".join(chunks)
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardRubricCampaignError(f"cannot read {label}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        not isinstance(value, dict)
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
    ):
        raise ObjectBongardRubricCampaignError(f"{label} is not a stable JSON object")
    return value


def object_bongard_rubric_campaign_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_bongard_rubric_campaign_source_bindings() -> dict[str, str]:
    """Freeze every loaded source that can affect the broad campaign."""

    raw = {
        "campaign_source": object_bongard_rubric_campaign_source_digest(),
        "batch_source": object_bongard_batch_source_digest(),
        "release_gate_source": object_bongard_release_gate_source_digest(),
        "semantic_source": object_bongard_semantics_source_digest(),
        "semantic_protocol": object_bongard_semantics_protocol_digest(),
        "hypothesis_source": object_hypothesis_extractor_source_digest(),
        "lineage_source": object_lineage_source_digest(),
        "rubric_observer_source": object_bongard_rubric_observer_source_digest(),
        "rubric_observer_protocol": (
            object_bongard_rubric_observer_protocol_digest()
        ),
        "rubric_version_space_source": (
            object_bongard_rubric_version_space_source_digest()
        ),
        "rubric_version_space_algorithm": (
            object_bongard_rubric_version_space_algorithm_digest()
        ),
        "rubric_ranker_source": object_bongard_rubric_ranker_source_digest(),
        "rubric_ranker_protocol": object_bongard_rubric_ranker_protocol_digest(),
        "rubric_ranker_transport_source": (
            object_bongard_rubric_ranker_transport_source_digest()
        ),
        "rubric_task_runner_source": (
            object_bongard_rubric_task_runner_source_digest()
        ),
        "turn_journal_source": object_bongard_turn_journal_source_digest(),
        "named_image_transport_source": prototype_scene_transport_source_digest(),
    }
    for name, value in raw.items():
        _require_raw_digest(value, name)
    return {name: "sha256:" + value for name, value in sorted(raw.items())}


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCampaignMetadata:
    preregistration: Mapping[str, Any]
    preregistration_digest: str
    plan: ObjectBongardBatchPlan
    descriptor: OfficialReleaseDescriptor
    split: SplitIndex
    task_ids: tuple[str, ...]
    train_task_ids: tuple[str, ...]
    exact_used_task_ids: tuple[str, ...]
    predecessor: ExposureLedger

    def __post_init__(self) -> None:
        _require_address(self.preregistration_digest, "preregistration digest")
        if self.preregistration.get("record_digest") != self.preregistration_digest:
            raise ObjectBongardRubricCampaignError(
                "preregistration digest differs from loaded record"
            )
        if self.plan.record_digest != self.preregistration.get("batch_plan_digest"):
            raise ObjectBongardRubricCampaignError(
                "preregistration and batch plan differ"
            )


def verify_object_bongard_rubric_campaign_metadata(
    *,
    preregistration_path: str | Path,
    expected_preregistration_digest: str,
    plan_path: str | Path,
    descriptor_path: str | Path,
    split_path: str | Path,
    predecessor_path: str | Path,
) -> ObjectBongardRubricCampaignMetadata:
    """Strictly replay the committed broad cohort without opening panel bytes."""

    expected = _require_address(
        expected_preregistration_digest, "expected preregistration digest"
    )
    prereg = _strict_json_file(preregistration_path, "preregistration")
    prereg_fields = {
        "schema", "created_at", "scope", "record_digest",
        "batch_plan_digest", "release_descriptor_digest",
        "split_source_digest", "task_inventory_digest",
        "exposure_predecessor_digest", "selection_seed",
        "selection_seed_digest", "requested_per_family",
        "selected_task_ids_digest", "sealed_query_panel_ids_digest",
        "selection_inputs_include_pixels",
        "selection_inputs_include_action_programs",
        "panel_bytes_opened_before_preregistration",
        "query_identities_sealed_before_support_pixels",
        "official_test_authorized", "execution_precommit_pending",
        "python_is_canonical_authority", "lean_required", "lean_removable",
    }
    if set(prereg) != prereg_fields:
        raise ObjectBongardRubricCampaignError(
            "preregistration fields differ from the committed schema"
        )
    content = {key: value for key, value in prereg.items() if key != "record_digest"}
    if (
        prereg["schema"] != EXPECTED_PREREG_SCHEMA
        or prereg["scope"] != EXPECTED_PREREG_SCOPE
        or prereg["record_digest"] != expected
        or _address(content) != expected
        or prereg["selection_inputs_include_pixels"] is not False
        or prereg["selection_inputs_include_action_programs"] is not False
        or prereg["panel_bytes_opened_before_preregistration"] is not False
        or prereg["query_identities_sealed_before_support_pixels"] is not True
        or prereg["official_test_authorized"] is not False
        or prereg["execution_precommit_pending"] is not True
        or prereg["python_is_canonical_authority"] is not True
        or prereg["lean_required"] is not False
        or prereg["lean_removable"] is not True
    ):
        raise ObjectBongardRubricCampaignError(
            "preregistration identity or policy differs"
        )

    plan_raw = _strict_json_file(plan_path, "batch plan")
    plan = ObjectBongardBatchPlan.from_data(plan_raw)
    descriptor_raw = _strict_json_file(descriptor_path, "release descriptor")
    descriptor = OfficialReleaseDescriptor.from_dict(descriptor_raw)
    descriptor.verify_split(split_path)
    split = SplitIndex.load(split_path)
    groups = split.canonical_groups
    task_ids = tuple(sorted((*groups["train"], *groups["val"], *groups["test"])))
    if len(task_ids) != len(set(task_ids)):
        raise ObjectBongardRubricCampaignError("primary split groups overlap")
    split.validate(task_ids, official_counts=True)
    train_task_ids = groups["train"]
    if object_bongard_task_inventory_digest(task_ids) != descriptor.task_ids_sha256:
        raise ObjectBongardRubricCampaignError(
            "split task inventory differs from release descriptor"
        )
    predecessor_raw = _strict_json_file(predecessor_path, "exposure predecessor")
    predecessor = ExposureLedger.from_dict(predecessor_raw)
    if predecessor.corpus_digest != descriptor.corpus_manifest_sha256:
        raise ObjectBongardRubricCampaignError(
            "exposure predecessor corpus differs from official release"
        )
    exact_used_task_ids = tuple(sorted(predecessor.exposed_task_ids))
    verify_object_bongard_batch_plan(
        plan,
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        exact_used_task_ids=exact_used_task_ids,
        selection_seed=prereg["selection_seed"],
    )
    expected_links = {
        "batch_plan_digest": plan.record_digest,
        "release_descriptor_digest": descriptor.digest,
        "split_source_digest": split.source_digest,
        "task_inventory_digest": object_bongard_task_inventory_digest(task_ids),
        "exposure_predecessor_digest": predecessor.digest,
        "selection_seed_digest": plan.selection_seed_digest,
        "requested_per_family": plan.requested_per_family,
        "selected_task_ids_digest": plan.to_data()["selected_task_ids_digest"],
        "sealed_query_panel_ids_digest": plan.to_data()[
            "sealed_query_panel_ids_digest"
        ],
    }
    if any(prereg[key] != value for key, value in expected_links.items()):
        raise ObjectBongardRubricCampaignError(
            "preregistration metadata links differ from exact replay"
        )
    return ObjectBongardRubricCampaignMetadata(
        preregistration=dict(prereg),
        preregistration_digest=expected,
        plan=plan,
        descriptor=descriptor,
        split=split,
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        exact_used_task_ids=exact_used_task_ids,
        predecessor=predecessor,
    )


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCampaignRuntime:
    visual: ObjectBongardTurnRuntime
    rank: ObjectBongardTurnRuntime
    max_workers: int
    max_physical_model_calls: int

    def __post_init__(self) -> None:
        if not isinstance(self.visual, ObjectBongardTurnRuntime):
            raise TypeError("visual runtime must be ObjectBongardTurnRuntime")
        if not isinstance(self.rank, ObjectBongardTurnRuntime):
            raise TypeError("rank runtime must be ObjectBongardTurnRuntime")
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or not 1 <= self.max_workers <= 12
        ):
            raise ObjectBongardRubricCampaignError("max_workers must lie in 1..12")
        if (
            isinstance(self.max_physical_model_calls, bool)
            or not isinstance(self.max_physical_model_calls, int)
            or not 1 <= self.max_physical_model_calls <= 100_000
        ):
            raise ObjectBongardRubricCampaignError(
                "physical model-call budget must lie in 1..100000"
            )
        if self.visual.transport_source_digest != prototype_scene_transport_source_digest():
            raise ObjectBongardRubricCampaignError(
                "visual runtime transport source differs"
            )
        if (
            self.rank.transport_source_digest
            != object_bongard_rubric_ranker_transport_source_digest()
        ):
            raise ObjectBongardRubricCampaignError(
                "rank runtime transport source differs"
            )

    @property
    def binding(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-object-rubric-campaign-runtime.v1",
            "visual": self.visual.binding,
            "rank": self.rank.binding,
            "max_workers": self.max_workers,
            "max_physical_model_calls": self.max_physical_model_calls,
            "bounded_concurrency_and_call_budget_frozen_before_panel_release": True,
            **_authority_data(),
        }

    @property
    def binding_digest(self) -> str:
        return _address(self.binding)


class ObjectBongardPhysicalCallBudget:
    """One campaign-wide, thread-safe admission bound for physical model calls."""

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ObjectBongardRubricCampaignError("call budget must be positive")
        self.limit = limit
        self._count = 0
        self._by_kind: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def by_kind(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._by_kind.items()))

    def wrap(self, kind: str, transport: Callable[..., CodexStructuredResult]):
        if not isinstance(kind, str) or not kind or not callable(transport):
            raise TypeError("budget wrapper requires a kind and callable transport")

        def bounded(*args: object, **kwargs: object) -> CodexStructuredResult:
            with self._lock:
                if self._count >= self.limit:
                    raise ObjectBongardRubricCampaignError(
                        "physical model-call budget exhausted"
                    )
                self._count += 1
                self._by_kind[kind] = self._by_kind.get(kind, 0) + 1
            result = transport(*args, **kwargs)
            if not isinstance(result, CodexStructuredResult):
                raise ObjectBongardRubricCampaignError(
                    "physical transport returned the wrong result type"
                )
            return result

        return bounded

    def account_journal_reuse(
        self,
        kind: str,
        journal: ObjectBongardNamedImageTurnJournalTransport
        | ObjectBongardTextTurnJournalTransport,
    ) -> None:
        """Charge a prior durable physical turn when resuming its journal."""

        if (
            journal.attempted_call_count != 1
            or journal.fresh_call_count not in (0, 1)
            or journal.reused_call_count not in (0, 1)
            or journal.fresh_call_count + journal.reused_call_count != 1
        ):
            raise ObjectBongardRubricCampaignError(
                "journal invocation accounting differs"
            )
        if journal.reused_call_count == 0:
            return
        with self._lock:
            if self._count >= self.limit:
                raise ObjectBongardRubricCampaignError(
                    "cumulative physical model-call budget exhausted"
                )
            self._count += 1
            self._by_kind[kind] = self._by_kind.get(kind, 0) + 1


def prepare_object_bongard_rubric_campaign(
    *,
    metadata: ObjectBongardRubricCampaignMetadata,
    archive: OfficialPanelArchive,
    store: ObjectBongardReleaseStore,
    runtime: ObjectBongardRubricCampaignRuntime,
    exposure_observed_at: str,
    launch_gate_bindings: Mapping[str, str],
) -> PreparedObjectBongardRelease:
    """Persist the execution precommit and exposure transition before pixels."""

    if archive.release_descriptor_digest != metadata.descriptor.digest:
        raise ObjectBongardRubricCampaignError(
            "official panel archive differs from campaign metadata"
        )
    configuration: dict[str, object] = {
        "campaign_id": CAMPAIGN_ID,
        "preregistration_digest": metadata.preregistration_digest,
        "runtime_binding_digest": runtime.binding_digest,
        "max_workers": runtime.max_workers,
        "max_physical_model_calls": runtime.max_physical_model_calls,
        "headless": True,
        "pure_python_predicates": True,
        "lean_required": False,
        "fixed_query_denominator": len(metadata.plan.tasks) * 2,
    }
    expected_keys = {
        "launch_authorization_digest",
        "campaign_runtime_precommit_digest",
        "calibration_assessment_digest",
        "calibration_replay_digest",
        "calibration_observation_inventory_digest",
    }
    bindings = dict(launch_gate_bindings)
    address_names = (
        "launch_authorization_digest",
        "campaign_runtime_precommit_digest",
        "calibration_replay_digest",
        "calibration_observation_inventory_digest",
    )
    if (
        set(bindings) != expected_keys
        or any(
            not isinstance(bindings.get(name), str)
            or _ADDRESS.fullmatch(bindings[name]) is None
            for name in address_names
        )
        or not isinstance(bindings.get("calibration_assessment_digest"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", bindings["calibration_assessment_digest"]
        )
        is None
    ):
        raise ObjectBongardRubricCampaignError(
            "campaign launch-gate bindings are malformed"
        )
    configuration.update(bindings)
    _verify_launch_gate_configuration(configuration)
    precommit = create_object_bongard_execution_precommit(
        plan=metadata.plan,
        predecessor=metadata.predecessor,
        descriptor=metadata.descriptor,
        archive=archive,
        task_ids=metadata.task_ids,
        train_task_ids=metadata.train_task_ids,
        exact_used_task_ids=metadata.exact_used_task_ids,
        runtime_source_bindings=object_bongard_rubric_campaign_source_bindings(),
        configuration=configuration,
        exposure_observed_at=exposure_observed_at,
    )
    prepared = prepare_object_bongard_release(
        store=store,
        plan=metadata.plan,
        precommit=precommit,
        predecessor=metadata.predecessor,
    )
    prepared = verify_prepared_object_bongard_release(prepared)
    return prepared


def _runtime_kwargs(runtime: ObjectBongardTurnRuntime) -> dict[str, object]:
    return {
        "model": runtime.model,
        "reasoning_effort": runtime.reasoning_effort,
        "minutes": runtime.minutes,
        "verbose": runtime.verbose,
        "executable": runtime.executable,
        "cloud_policy_cache_snapshot": runtime.cloud_policy_cache_snapshot,
        "expected_launcher_digest": runtime.expected_launcher_digest,
        "model_catalog_snapshot": runtime.model_catalog_snapshot,
        "no_tools_attestation": runtime.no_tools_attestation,
    }


def _summary_data(summary: ObjectBongardTurnJournalSummary) -> dict[str, object]:
    if not isinstance(summary, ObjectBongardTurnJournalSummary):
        raise TypeError("journal summary has the wrong type")
    return summary.to_data()


def _validate_summary_data(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ObjectBongardRubricCampaignError("journal summary must be an object")
    expected = {
        "schema", "manifest_digest", "turn_key", "terminal_status",
        "claim_digest", "result_digest", "outcome_digest", "record_digest",
        "predicate_authority_id", "python_is_canonical_authority",
        "lean_present", "lean_required", "lean_removable",
        "lean_affects_identity_or_replay",
    }
    if set(value) != expected:
        raise ObjectBongardRubricCampaignError("journal summary fields differ")
    raw = dict(value)
    if (
        raw["schema"] != "gkm.bongard-codex-turn-journal-summary.v1"
        or raw["python_is_canonical_authority"] is not True
        or raw["lean_present"] is not False
        or raw["lean_required"] is not False
        or raw["lean_removable"] is not True
        or raw["lean_affects_identity_or_replay"] is not False
        or raw["record_digest"]
        != _address({key: item for key, item in raw.items() if key != "record_digest"})
    ):
        raise ObjectBongardRubricCampaignError("journal summary differs")
    _require_address(raw["record_digest"], "journal summary digest")
    return raw


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricPanelObservation:
    released_panel: ReleasedOfficialPanel
    released_panel_store_receipt: ObjectBongardWriteOnceReceipt
    hypothesis_packet: ObjectHypothesisPacket
    lineage_packet: ObjectLineagePacket
    artifact: ObjectBongardRubricObserverArtifact
    journal_relative_directories: tuple[str, ...]
    journal_summaries: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.released_panel, ReleasedOfficialPanel):
            raise TypeError("released panel has the wrong type")
        if not isinstance(
            self.released_panel_store_receipt, ObjectBongardWriteOnceReceipt
        ):
            raise TypeError("released panel store receipt has the wrong type")
        if self.released_panel_store_receipt.object_digest != self.released_panel.record_digest:
            raise ObjectBongardRubricCampaignError(
                "released panel store receipt differs"
            )
        verify_object_hypothesis_packet(
            self.hypothesis_packet,
            expected_png_bytes=self.released_panel.exact_png_bytes,
        )
        verify_object_lineage_packet(
            self.lineage_packet, self.released_panel.exact_png_bytes
        )
        if self.lineage_packet.hypothesis_packet_digest != self.hypothesis_packet.digest():
            raise ObjectBongardRubricCampaignError(
                "lineage and hypothesis packets differ"
            )
        verify_object_bongard_rubric_observer_artifact(
            self.artifact,
            self.released_panel.exact_png_bytes,
            panel_id=self.released_panel.panel_id,
            rubric_spec=self.artifact.rubric_spec,
            hypothesis_packet=self.hypothesis_packet,
            lineage_packet=self.lineage_packet,
            expected_artifact_digest=self.artifact.artifact_digest,
        )
        if (
            len(self.journal_relative_directories)
            != len(self.hypothesis_packet.atlas_sheets)
            or len(self.journal_summaries)
            != len(self.hypothesis_packet.atlas_sheets)
            or len(set(self.journal_relative_directories))
            != len(self.journal_relative_directories)
            or any(
                not isinstance(item, str)
                or not item
                or Path(item).is_absolute()
                or ".." in Path(item).parts
                for item in self.journal_relative_directories
            )
        ):
            raise ObjectBongardRubricCampaignError(
                "rubric panel journal inventory differs"
            )
        for summary in self.journal_summaries:
            _validate_summary_data(summary)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-object-rubric-panel-observation.v1",
            "released_panel": self.released_panel.to_data(),
            "released_panel_store_receipt": (
                self.released_panel_store_receipt.to_data()
            ),
            "hypothesis_packet": self.hypothesis_packet.to_data(),
            "lineage_packet": self.lineage_packet.to_data(),
            "artifact": self.artifact.to_data(),
            "journal_relative_directories": list(
                self.journal_relative_directories
            ),
            "journal_summaries": [dict(item) for item in self.journal_summaries],
            "cold_replay_requires_exact_official_png_bytes": True,
            **_authority_data(),
        }

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "ObjectBongardRubricPanelObservation":
        expected = {
            "schema", "released_panel", "released_panel_store_receipt",
            "hypothesis_packet", "lineage_packet", "artifact",
            "journal_relative_directories", "journal_summaries",
            "cold_replay_requires_exact_official_png_bytes", *_authority_data(),
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value["schema"]
            != "gkm.bongard-object-rubric-panel-observation.v1"
            or value["cold_replay_requires_exact_official_png_bytes"] is not True
            or any(value[key] != expected_value for key, expected_value in _authority_data().items())
            or not isinstance(value["released_panel"], Mapping)
            or not isinstance(value["released_panel_store_receipt"], Mapping)
            or not isinstance(value["hypothesis_packet"], Mapping)
            or not isinstance(value["lineage_packet"], Mapping)
            or not isinstance(value["artifact"], Mapping)
            or not isinstance(value["journal_relative_directories"], list)
            or not isinstance(value["journal_summaries"], list)
        ):
            raise ObjectBongardRubricCampaignError(
                "rubric panel observation fields differ"
            )
        result = cls(
            released_panel=ReleasedOfficialPanel.from_data(value["released_panel"]),
            released_panel_store_receipt=ObjectBongardWriteOnceReceipt.from_data(
                value["released_panel_store_receipt"]
            ),
            hypothesis_packet=ObjectHypothesisPacket.from_data(
                value["hypothesis_packet"]
            ),
            lineage_packet=ObjectLineagePacket.from_data(value["lineage_packet"]),
            artifact=ObjectBongardRubricObserverArtifact.from_data(value["artifact"]),
            journal_relative_directories=tuple(
                value["journal_relative_directories"]
            ),
            journal_summaries=tuple(
                _validate_summary_data(item) for item in value["journal_summaries"]
            ),
        )
        if result.to_data() != dict(value):
            raise ObjectBongardRubricCampaignError(
                "rubric panel observation is not canonical"
            )
        return result


def _semantic_turn(
    *,
    task: ObjectBongardTaskPlan,
    support: Mapping[str, ReleasedOfficialPanel],
    prepared: PreparedObjectBongardRelease,
    runtime: ObjectBongardTurnRuntime,
    journals_root: Path,
    budget: ObjectBongardPhysicalCallBudget,
    visual_transport: Callable[..., CodexStructuredResult],
) -> tuple[ObjectBongardSemanticArtifact, str, dict[str, object]]:
    prompt = object_bongard_semantics_prompt()
    schema = prototype_object_description_output_schema()
    ids = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
    names = tuple(
        [f"group_0_ref_{index:02d}.png" for index in range(6)]
        + [f"group_1_ref_{index:02d}.png" for index in range(6)]
    )
    images = tuple(
        (name, support[panel_id].exact_png_bytes)
        for name, panel_id in zip(names, ids, strict=True)
    )
    relative = f"tasks/{task.task_id}/semantic"
    journal = ObjectBongardNamedImageTurnJournalTransport(
        journals_root / relative,
        authorization_digest=prepared.authorization.record_digest,
        execution_precommit_digest=prepared.precommit.record_digest,
        task_id=task.task_id,
        turn_kind="semantic",
        expected_prompt=prompt,
        expected_images=images,
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=budget.wrap("semantic", visual_transport),
    )
    artifact = describe_object_bongard_support(
        task_id=task.task_id,
        group_0_panel_ids=task.side_0_support_panel_ids,
        group_1_panel_ids=task.side_1_support_panel_ids,
        support_png_by_panel_id={
            panel_id: support[panel_id].exact_png_bytes for panel_id in ids
        },
        observation_context_digest=prepared.precommit.record_digest,
        **_runtime_kwargs(runtime),
        transport=journal,
    )
    budget.account_journal_reuse("semantic", journal)
    verify_object_bongard_semantic_artifact(
        artifact,
        support_png_by_panel_id={
            panel_id: support[panel_id].exact_png_bytes for panel_id in ids
        },
        expected_task_id=task.task_id,
        expected_observation_context_digest=prepared.precommit.record_digest,
        expected_artifact_digest=artifact.artifact_digest,
    )
    return artifact, relative, _summary_data(journal.verify())


def _observe_released_panel(
    *,
    task_id: str,
    released: ReleasedOfficialPanel,
    release_receipt: ObjectBongardWriteOnceReceipt,
    rubric_spec: ObjectBongardRubricSpec,
    prepared: PreparedObjectBongardRelease,
    runtime: ObjectBongardTurnRuntime,
    journals_root: Path,
    relative_prefix: str,
    turn_kind_prefix: str,
    budget: ObjectBongardPhysicalCallBudget,
    visual_transport: Callable[..., CodexStructuredResult],
) -> ObjectBongardRubricPanelObservation:
    png = released.exact_png_bytes
    hypotheses = extract_object_hypothesis_packet(png)
    lineages = extract_object_lineage_packet(png, hypotheses)
    rendered = dict(render_object_hypothesis_atlas(hypotheses, png))
    schema = object_bongard_rubric_observer_output_schema()
    journals: dict[str, ObjectBongardNamedImageTurnJournalTransport] = {}
    relative_directories: list[str] = []
    for sheet in hypotheses.atlas_sheets:
        relative = f"{relative_prefix}/sheet_{sheet.sheet_index:03d}"
        relative_directories.append(relative)
        journals[sheet.name] = ObjectBongardNamedImageTurnJournalTransport(
            journals_root / relative,
            authorization_digest=prepared.authorization.record_digest,
            execution_precommit_digest=prepared.precommit.record_digest,
            task_id=task_id,
            turn_kind=f"{turn_kind_prefix}_{sheet.sheet_index:03d}",
            expected_prompt=object_bongard_rubric_observer_prompt(
                rubric_spec, sheet
            ),
            expected_images=(
                ("scene.png", png),
                (sheet.name, rendered[sheet.name]),
            ),
            expected_output_schema=schema,
            runtime=runtime,
            underlying_transport=budget.wrap("rubric_observer", visual_transport),
        )

    def dispatch(
        prompt: str,
        paths: Sequence[str],
        names: Sequence[str],
        output_schema: Mapping[str, Any],
        **kwargs: object,
    ) -> CodexStructuredResult:
        if len(names) != 2 or names[0] != "scene.png" or names[1] not in journals:
            raise ObjectBongardRubricCampaignError(
                "rubric observer requested an uncommitted atlas sheet"
            )
        return journals[names[1]](
            prompt, paths, names, output_schema, **kwargs
        )

    artifact = observe_object_bongard_rubric(
        png,
        panel_id=released.panel_id,
        rubric_spec=rubric_spec,
        hypothesis_packet=hypotheses,
        lineage_packet=lineages,
        expected_scene_sha256=hashlib.sha256(png).hexdigest(),
        expected_rubric_spec_digest=rubric_spec.spec_digest,
        expected_hypothesis_packet_digest=hypotheses.digest(),
        expected_lineage_packet_digest=lineages.digest(),
        observation_context_digest=prepared.precommit.record_digest,
        **_runtime_kwargs(runtime),
        transport=dispatch,
    )
    for journal in journals.values():
        budget.account_journal_reuse("rubric_observer", journal)
    verify_object_bongard_rubric_observer_artifact(
        artifact,
        png,
        panel_id=released.panel_id,
        rubric_spec=rubric_spec,
        hypothesis_packet=hypotheses,
        lineage_packet=lineages,
        expected_artifact_digest=artifact.artifact_digest,
    )
    summaries = tuple(
        _summary_data(journals[sheet.name].verify())
        for sheet in hypotheses.atlas_sheets
    )
    return ObjectBongardRubricPanelObservation(
        released_panel=released,
        released_panel_store_receipt=release_receipt,
        hypothesis_packet=hypotheses,
        lineage_packet=lineages,
        artifact=artifact,
        journal_relative_directories=tuple(relative_directories),
        journal_summaries=summaries,
    )


class _JournaledRubricRanker:
    def __init__(
        self,
        *,
        task_id: str,
        prepared: PreparedObjectBongardRelease,
        runtime: ObjectBongardTurnRuntime,
        journals_root: Path,
        relative_directory: str,
        budget: ObjectBongardPhysicalCallBudget,
        rank_transport: Callable[..., CodexStructuredResult],
    ) -> None:
        if runtime.cloud_policy_cache_snapshot is None:
            raise ObjectBongardRubricCampaignError(
                "rank runtime requires an explicit policy-cache snapshot"
            )
        self.task_id = task_id
        self.prepared = prepared
        self.runtime = runtime
        self.journals_root = journals_root
        self.relative_directory = relative_directory
        self.budget = budget
        self.rank_transport = rank_transport
        self.journal: ObjectBongardTextTurnJournalTransport | None = None
        self.ranker: ObjectBongardRubricRanker | None = None
        self.summary: dict[str, object] | None = None

    def __call__(
        self,
        version_space: ObjectBongardRubricSupportVersionSpace,
        *,
        rubric_spec: ObjectBongardRubricSpec,
        semantic_artifact: ObjectBongardSemanticArtifact,
        positive_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
        negative_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
        rank_input_digest: str,
    ) -> ObjectBongardRubricRankResponse:
        prompt = object_bongard_rubric_ranker_prompt(
            version_space=version_space,
            rubric_spec=rubric_spec,
            semantic_artifact=semantic_artifact,
            positive_support_artifacts=positive_support_artifacts,
            negative_support_artifacts=negative_support_artifacts,
            rank_input_digest=rank_input_digest,
        )
        schema = object_bongard_rubric_ranker_output_schema()
        journal = ObjectBongardTextTurnJournalTransport(
            self.journals_root / self.relative_directory,
            authorization_digest=self.prepared.authorization.record_digest,
            execution_precommit_digest=self.prepared.precommit.record_digest,
            task_id=self.task_id,
            turn_kind="rank",
            expected_prompt=prompt,
            expected_output_schema=schema,
            runtime=self.runtime,
            underlying_transport=self.budget.wrap("rank", self.rank_transport),
        )
        snapshot = self.runtime.cloud_policy_cache_snapshot
        if snapshot is None:  # statically narrow the already checked invariant
            raise ObjectBongardRubricCampaignError(
                "rank policy-cache snapshot disappeared"
            )
        ranker = ObjectBongardRubricRanker(
            model=self.runtime.model,
            reasoning_effort=self.runtime.reasoning_effort,
            minutes=self.runtime.minutes,
            verbose=self.runtime.verbose,
            executable=self.runtime.executable,
            expected_launcher_digest=self.runtime.expected_launcher_digest,
            cloud_policy_cache_snapshot=snapshot,
            expected_cloud_policy_cache_binding=self.runtime.policy_cache_binding,
            expected_transport_source_digest=self.runtime.transport_source_digest,
            model_catalog_snapshot=self.runtime.model_catalog_snapshot,
            no_tools_attestation=self.runtime.no_tools_attestation,
            transport=journal,
        )
        self.journal = journal
        self.ranker = ranker
        try:
            response = ranker(
                version_space,
                rubric_spec=rubric_spec,
                semantic_artifact=semantic_artifact,
                positive_support_artifacts=positive_support_artifacts,
                negative_support_artifacts=negative_support_artifacts,
                rank_input_digest=rank_input_digest,
            )
        finally:
            self.budget.account_journal_reuse("rank", journal)
            self.summary = _summary_data(journal.verify())
        return response

    def verify_response(
        self, response: ObjectBongardRubricRankResponse, **kwargs: object
    ) -> ObjectBongardRubricRankResponse:
        if self.ranker is None:
            raise ObjectBongardRubricCampaignError(
                "rank response verification preceded the rank turn"
            )
        return self.ranker.verify_response(response, **kwargs)  # type: ignore[arg-type]


def _task_execution_content(
    value: "ObjectBongardRubricTaskExecution",
) -> dict[str, object]:
    return {
        "schema": TASK_EXECUTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "campaign_source_digest": value.campaign_source_digest,
        "task_plan": value.task_plan.to_data(),
        "execution_precommit_digest": value.execution_precommit_digest,
        "release_authorization_digest": value.release_authorization_digest,
        "runtime_binding_digest": value.runtime_binding_digest,
        "semantic_artifact": value.semantic_artifact.to_data(),
        "semantic_journal_relative_directory": (
            value.semantic_journal_relative_directory
        ),
        "semantic_journal_summary": dict(value.semantic_journal_summary),
        "support_observations": [item.to_data() for item in value.support_observations],
        "task_run": None if value.task_run is None else value.task_run.to_data(),
        "rank_journal_relative_directory": value.rank_journal_relative_directory,
        "rank_journal_summary": (
            None
            if value.rank_journal_summary is None
            else dict(value.rank_journal_summary)
        ),
        "query_observations": [item.to_data() for item in value.query_observations],
        "task_freeze_store_receipt": (
            None
            if value.task_freeze_store_receipt is None
            else value.task_freeze_store_receipt.to_data()
        ),
        "task_commit_store_receipt": (
            None
            if value.task_commit_store_receipt is None
            else value.task_commit_store_receipt.to_data()
        ),
        "fixed_score_denominator": 2,
        "correct_count": value.correct_count,
        "abstention_count": value.abstention_count,
        "gap_counts_as_two_incorrect_abstentions": True,
        "formula_frozen_and_durably_reloaded_before_query_release": (
            value.task_run is None
            or value.task_run.status is not ObjectBongardRubricTaskRunStatus.COMPLETE
            or (
                value.task_run.freeze_reload_calls_made == 1
                and value.task_run.query_source_calls_made == 1
            )
        ),
        "cold_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricTaskExecution:
    campaign_source_digest: str
    task_plan: ObjectBongardTaskPlan
    execution_precommit_digest: str
    release_authorization_digest: str
    runtime_binding_digest: str
    semantic_artifact: ObjectBongardSemanticArtifact
    semantic_journal_relative_directory: str
    semantic_journal_summary: Mapping[str, Any]
    support_observations: tuple[ObjectBongardRubricPanelObservation, ...]
    task_run: ObjectBongardRubricTaskRunArchive | None
    rank_journal_relative_directory: str | None
    rank_journal_summary: Mapping[str, Any] | None
    query_observations: tuple[ObjectBongardRubricPanelObservation, ...]
    task_freeze_store_receipt: ObjectBongardWriteOnceReceipt | None
    task_commit_store_receipt: ObjectBongardWriteOnceReceipt | None
    correct_count: int
    abstention_count: int
    record_digest: str

    def __post_init__(self) -> None:
        if self.campaign_source_digest != object_bongard_rubric_campaign_source_digest():
            raise ObjectBongardRubricCampaignError("campaign source digest differs")
        _require_address(self.execution_precommit_digest, "execution precommit digest")
        _require_address(self.release_authorization_digest, "release authorization digest")
        _require_address(self.runtime_binding_digest, "runtime binding digest")
        _require_address(self.record_digest, "task execution digest")
        if (
            not isinstance(self.task_plan, ObjectBongardTaskPlan)
            or not isinstance(self.semantic_artifact, ObjectBongardSemanticArtifact)
            or self.semantic_artifact.task_id != self.task_plan.task_id
            or self.semantic_artifact.observation_context_digest
            != self.execution_precommit_digest
            or self.semantic_artifact.group_panel_ids
            != (
                self.task_plan.side_0_support_panel_ids,
                self.task_plan.side_1_support_panel_ids,
            )
            or not isinstance(self.semantic_journal_relative_directory, str)
            or Path(self.semantic_journal_relative_directory).is_absolute()
            or ".." in Path(self.semantic_journal_relative_directory).parts
        ):
            raise ObjectBongardRubricCampaignError(
                "task execution support parents differ"
        )
        _validate_summary_data(self.semantic_journal_summary)
        if self.task_run is None:
            common_error_violation = (
                bool(self.query_observations)
                or self.task_freeze_store_receipt is not None
                or self.task_commit_store_receipt is not None
                or self.correct_count != 0
                or self.abstention_count != 2
            )
            if self.semantic_artifact.status is not PrototypeSceneObserverStatus.SUCCESS:
                if (
                    common_error_violation
                    or self.support_observations
                    or self.rank_journal_relative_directory is not None
                    or self.rank_journal_summary is not None
                ):
                    raise ObjectBongardRubricCampaignError(
                        "semantic-error task crossed the rubric or query boundary"
                    )
            else:
                spec = ObjectBongardRubricSpec.from_semantic_artifact(
                    self.semantic_artifact,
                    expected_artifact_digest=self.semantic_artifact.artifact_digest,
                )
                support_ids = (
                    *self.task_plan.side_0_support_panel_ids,
                    *self.task_plan.side_1_support_panel_ids,
                )
                if (
                    common_error_violation
                    or len(self.support_observations) != 12
                    or tuple(
                        item.released_panel.panel_id
                        for item in self.support_observations
                    )
                    != support_ids
                    or any(
                        item.artifact.rubric_spec != spec
                        for item in self.support_observations
                    )
                    or not isinstance(self.rank_journal_relative_directory, str)
                    or Path(self.rank_journal_relative_directory).is_absolute()
                    or ".." in Path(self.rank_journal_relative_directory).parts
                    or self.rank_journal_summary is None
                ):
                    raise ObjectBongardRubricCampaignError(
                        "rank-error task support or phase bindings differ"
                    )
                summary = _validate_summary_data(self.rank_journal_summary)
                if summary["terminal_status"] not in {"success", "failure"}:
                    raise ObjectBongardRubricCampaignError(
                        "rank-error task lacks a terminal rank journal"
                    )
                version = build_object_bongard_rubric_support_version_space(
                    spec,
                    tuple(
                        item.artifact for item in self.support_observations[:6]
                    ),
                    tuple(
                        item.artifact for item in self.support_observations[6:]
                    ),
                )
                if not version.survivor_candidate_digests:
                    raise ObjectBongardRubricCampaignError(
                        "rank-error task did not require a rank turn"
                    )
            if self.record_digest != _address(_task_execution_content(self)):
                raise ObjectBongardRubricCampaignError(
                    "task execution content digest differs"
                )
            return
        if (
            self.semantic_artifact.status
            is not PrototypeSceneObserverStatus.SUCCESS
            or self.task_run.task_plan != self.task_plan
            or self.task_run.execution_precommit_digest
            != self.execution_precommit_digest
            or self.semantic_artifact != self.task_run.semantic_artifact
            or len(self.support_observations) != 12
            or tuple(item.artifact for item in self.support_observations[:6])
            != self.task_run.side_0_support
            or tuple(item.artifact for item in self.support_observations[6:])
            != self.task_run.side_1_support
            or tuple(item.released_panel.panel_id for item in self.support_observations)
            != (
                *self.task_plan.side_0_support_panel_ids,
                *self.task_plan.side_1_support_panel_ids,
            )
        ):
            raise ObjectBongardRubricCampaignError(
                "successful semantic task support parents differ"
            )
        complete = self.task_run.status is ObjectBongardRubricTaskRunStatus.COMPLETE
        if complete:
            if (
                len(self.query_observations) != 2
                or tuple(item.released_panel.panel_id for item in self.query_observations)
                != (
                    self.task_plan.side_0_query_panel_id,
                    self.task_plan.side_1_query_panel_id,
                )
                or tuple(item.artifact for item in self.query_observations)
                != (self.task_run.side_0_query, self.task_run.side_1_query)
                or self.task_freeze_store_receipt is None
                or self.task_commit_store_receipt is None
                or self.task_run.freeze is None
                or self.task_run.freeze_commit is None
                or self.task_freeze_store_receipt.object_digest
                != self.task_run.freeze.record_digest
                or self.task_commit_store_receipt.object_digest
                != self.task_run.freeze_commit.record_digest
                or self.rank_journal_relative_directory is None
                or self.rank_journal_summary is None
                or self.correct_count != self.task_run.correct_count
                or self.abstention_count != self.task_run.abstention_count
            ):
                raise ObjectBongardRubricCampaignError(
                    "complete task execution phase bindings differ"
                )
            _validate_summary_data(self.rank_journal_summary)
        else:
            if (
                self.query_observations
                or self.task_freeze_store_receipt is not None
                or self.task_commit_store_receipt is not None
                or self.rank_journal_relative_directory is not None
                or self.rank_journal_summary is not None
                or self.correct_count != 0
                or self.abstention_count != 2
            ):
                raise ObjectBongardRubricCampaignError(
                    "support gap crossed a later campaign phase"
                )
        if self.record_digest != _address(_task_execution_content(self)):
            raise ObjectBongardRubricCampaignError(
                "task execution content digest differs"
            )

    @property
    def fixed_score_denominator(self) -> int:
        return 2

    def to_data(self) -> dict[str, object]:
        return {**_task_execution_content(self), "record_digest": self.record_digest}

    @classmethod
    def seal(cls, **values: object) -> "ObjectBongardRubricTaskExecution":
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(_task_execution_content(provisional)),
        )

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "ObjectBongardRubricTaskExecution":
        required = {
            "schema", "campaign_id", "campaign_source_digest", "task_plan",
            "execution_precommit_digest", "release_authorization_digest",
            "runtime_binding_digest", "semantic_artifact",
            "semantic_journal_relative_directory", "semantic_journal_summary",
            "support_observations", "task_run",
            "rank_journal_relative_directory", "rank_journal_summary",
            "query_observations", "task_freeze_store_receipt",
            "task_commit_store_receipt", "fixed_score_denominator",
            "correct_count", "abstention_count",
            "gap_counts_as_two_incorrect_abstentions",
            "formula_frozen_and_durably_reloaded_before_query_release",
            "cold_replay_model_calls", *_authority_data(), "record_digest",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != required
            or value["schema"] != TASK_EXECUTION_SCHEMA
            or value["campaign_id"] != CAMPAIGN_ID
            or value["fixed_score_denominator"] != 2
            or value["gap_counts_as_two_incorrect_abstentions"] is not True
            or value["formula_frozen_and_durably_reloaded_before_query_release"] is not True
            or value["cold_replay_model_calls"] != 0
            or any(value[key] != item for key, item in _authority_data().items())
            or not isinstance(value["task_plan"], Mapping)
            or not isinstance(value["semantic_artifact"], Mapping)
            or not isinstance(value["semantic_journal_summary"], Mapping)
            or not isinstance(value["support_observations"], list)
            or (
                value["task_run"] is not None
                and not isinstance(value["task_run"], Mapping)
            )
            or not isinstance(value["query_observations"], list)
        ):
            raise ObjectBongardRubricCampaignError(
                "task execution fields differ"
            )
        result = cls(
            campaign_source_digest=value["campaign_source_digest"],
            task_plan=ObjectBongardTaskPlan.from_data(value["task_plan"]),
            execution_precommit_digest=value["execution_precommit_digest"],
            release_authorization_digest=value["release_authorization_digest"],
            runtime_binding_digest=value["runtime_binding_digest"],
            semantic_artifact=ObjectBongardSemanticArtifact.from_data(
                value["semantic_artifact"]
            ),
            semantic_journal_relative_directory=value[
                "semantic_journal_relative_directory"
            ],
            semantic_journal_summary=_validate_summary_data(
                value["semantic_journal_summary"]
            ),
            support_observations=tuple(
                ObjectBongardRubricPanelObservation.from_data(item)
                for item in value["support_observations"]
            ),
            task_run=(
                None
                if value["task_run"] is None
                else ObjectBongardRubricTaskRunArchive.from_data(
                    value["task_run"]
                )
            ),
            rank_journal_relative_directory=value[
                "rank_journal_relative_directory"
            ],
            rank_journal_summary=(
                None
                if value["rank_journal_summary"] is None
                else _validate_summary_data(value["rank_journal_summary"])
            ),
            query_observations=tuple(
                ObjectBongardRubricPanelObservation.from_data(item)
                for item in value["query_observations"]
            ),
            task_freeze_store_receipt=(
                None
                if value["task_freeze_store_receipt"] is None
                else ObjectBongardWriteOnceReceipt.from_data(
                    value["task_freeze_store_receipt"]
                )
            ),
            task_commit_store_receipt=(
                None
                if value["task_commit_store_receipt"] is None
                else ObjectBongardWriteOnceReceipt.from_data(
                    value["task_commit_store_receipt"]
                )
            ),
            correct_count=value["correct_count"],
            abstention_count=value["abstention_count"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise ObjectBongardRubricCampaignError(
                "task execution is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class PersistedObjectBongardRubricTaskExecution:
    execution: ObjectBongardRubricTaskExecution
    store_receipt: ObjectBongardWriteOnceReceipt

    def __post_init__(self) -> None:
        if (
            not isinstance(self.execution, ObjectBongardRubricTaskExecution)
            or not isinstance(self.store_receipt, ObjectBongardWriteOnceReceipt)
            or self.store_receipt.object_kind != "rubric-task-execution"
            or self.store_receipt.object_digest != self.execution.record_digest
        ):
            raise ObjectBongardRubricCampaignError(
                "persisted task execution binding differs"
            )


def _persist_rubric_task_execution(
    prepared: PreparedObjectBongardRelease,
    execution: ObjectBongardRubricTaskExecution,
) -> PersistedObjectBongardRubricTaskExecution:
    receipt = prepared.store.persist(
        object_kind="rubric-task-execution",
        object_digest=execution.record_digest,
        data=execution.to_data(),
    )
    prepared.store.verify(receipt, expected_data=execution.to_data())
    return PersistedObjectBongardRubricTaskExecution(execution, receipt)


def run_object_bongard_rubric_campaign_task(
    *,
    task: ObjectBongardTaskPlan,
    prepared: PreparedObjectBongardRelease,
    archive: OfficialPanelArchive,
    runtime: ObjectBongardRubricCampaignRuntime,
    journals_root: str | Path,
    budget: ObjectBongardPhysicalCallBudget,
    visual_transport: Callable[..., CodexStructuredResult],
    rank_transport: Callable[..., CodexStructuredResult],
) -> PersistedObjectBongardRubricTaskExecution:
    """Execute one task through the real durable support/freeze/query gate."""

    prepared = verify_prepared_object_bongard_release(prepared)
    _verify_launch_gate_configuration(prepared.precommit.configuration)
    if (
        not isinstance(task, ObjectBongardTaskPlan)
        or task not in prepared.plan.tasks
        or prepared.precommit.runtime_source_bindings
        != tuple(sorted(object_bongard_rubric_campaign_source_bindings().items()))
        or dict(prepared.precommit.configuration).get("runtime_binding_digest")
        != runtime.binding_digest
        or dict(prepared.precommit.configuration).get("max_workers")
        != runtime.max_workers
        or dict(prepared.precommit.configuration).get(
            "max_physical_model_calls"
        )
        != runtime.max_physical_model_calls
    ):
        raise ObjectBongardRubricCampaignError(
            "task or committed campaign source bindings differ"
        )
    root = Path(journals_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise ObjectBongardRubricCampaignError(
            "campaign journal root must be a real directory"
        )

    support_ids = (
        *task.side_0_support_panel_ids,
        *task.side_1_support_panel_ids,
    )
    support_releases: dict[
        str, tuple[ReleasedOfficialPanel, ObjectBongardWriteOnceReceipt]
    ] = {}
    for panel_id in support_ids:
        support_releases[panel_id] = release_object_bongard_support_panel(
            prepared=prepared, archive=archive, panel_id=panel_id
        )
    support_by_id = {
        panel_id: pair[0] for panel_id, pair in support_releases.items()
    }
    semantic, semantic_relative, semantic_summary = _semantic_turn(
        task=task,
        support=support_by_id,
        prepared=prepared,
        runtime=runtime.visual,
        journals_root=root,
        budget=budget,
        visual_transport=visual_transport,
    )
    if semantic.status is not PrototypeSceneObserverStatus.SUCCESS:
        execution = ObjectBongardRubricTaskExecution.seal(
            campaign_source_digest=object_bongard_rubric_campaign_source_digest(),
            task_plan=task,
            execution_precommit_digest=prepared.precommit.record_digest,
            release_authorization_digest=prepared.authorization.record_digest,
            runtime_binding_digest=runtime.binding_digest,
            semantic_artifact=semantic,
            semantic_journal_relative_directory=semantic_relative,
            semantic_journal_summary=semantic_summary,
            support_observations=(),
            task_run=None,
            rank_journal_relative_directory=None,
            rank_journal_summary=None,
            query_observations=(),
            task_freeze_store_receipt=None,
            task_commit_store_receipt=None,
            correct_count=0,
            abstention_count=2,
        )
        return _persist_rubric_task_execution(prepared, execution)
    spec = ObjectBongardRubricSpec.from_semantic_artifact(
        semantic, expected_artifact_digest=semantic.artifact_digest
    )
    support_observations: list[ObjectBongardRubricPanelObservation] = []
    for panel_index, panel_id in enumerate(support_ids):
        released, receipt = support_releases[panel_id]
        support_observations.append(
            _observe_released_panel(
                task_id=task.task_id,
                released=released,
                release_receipt=receipt,
                rubric_spec=spec,
                prepared=prepared,
                runtime=runtime.visual,
                journals_root=root,
                relative_prefix=(
                    f"tasks/{task.task_id}/support/panel_{panel_index:02d}"
                ),
                turn_kind_prefix=f"s{panel_index:02d}",
                budget=budget,
                visual_transport=visual_transport,
            )
        )

    rank_relative = f"tasks/{task.task_id}/rank"
    journaled_ranker = _JournaledRubricRanker(
        task_id=task.task_id,
        prepared=prepared,
        runtime=runtime.rank,
        journals_root=root,
        relative_directory=rank_relative,
        budget=budget,
        rank_transport=rank_transport,
    )
    freeze_state: dict[str, object] = {}
    query_observations: list[ObjectBongardRubricPanelObservation] = []

    def commit_freeze(payload: bytes) -> ObjectBongardRubricTaskFreezeCommit:
        try:
            raw = json.loads(payload.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ObjectBongardRubricCampaignError(
                "task runner supplied a malformed freeze payload"
            ) from exc
        if not isinstance(raw, Mapping):
            raise ObjectBongardRubricCampaignError(
                "task runner freeze payload is not an object"
            )
        freeze = ObjectBongardRubricTaskFreeze.from_data(raw)
        freeze_receipt = persist_object_bongard_task_freeze(
            store=prepared.store, freeze=freeze
        )
        prepared.store.verify(freeze_receipt, expected_data=freeze.to_data())
        commit = ObjectBongardRubricTaskFreezeCommit.seal(
            freeze,
            payload,
            task_freeze_store_receipt_digest=freeze_receipt.record_digest,
        )
        commit_receipt = persist_object_bongard_task_commit(
            store=prepared.store, commit=commit
        )
        prepared.store.verify(commit_receipt, expected_data=commit.to_data())
        freeze_state.update(
            freeze=freeze,
            freeze_receipt=freeze_receipt,
            commit=commit,
            commit_receipt=commit_receipt,
        )
        return commit

    def reload_freeze(commit_data: Mapping[str, object]) -> bytes:
        if set(freeze_state) != {
            "freeze", "freeze_receipt", "commit", "commit_receipt"
        }:
            raise ObjectBongardRubricCampaignError(
                "freeze reload preceded durable freeze and commit persistence"
            )
        commit = freeze_state["commit"]
        receipt = freeze_state["freeze_receipt"]
        if (
            not isinstance(commit, ObjectBongardRubricTaskFreezeCommit)
            or not isinstance(receipt, ObjectBongardWriteOnceReceipt)
            or commit.to_data() != dict(commit_data)
        ):
            raise ObjectBongardRubricCampaignError(
                "freeze reload commit differs"
            )
        freeze = freeze_state["freeze"]
        if not isinstance(freeze, ObjectBongardRubricTaskFreeze):
            raise ObjectBongardRubricCampaignError("durable freeze type differs")
        reloaded = prepared.store.verify(receipt, expected_data=freeze.to_data())
        return canonical_json(dict(reloaded)) + b"\n"

    def release_and_observe_queries(
        freeze_data: Mapping[str, object], commit_data: Mapping[str, object]
    ) -> Mapping[str, ObjectBongardRubricObserverArtifact]:
        if set(freeze_state) != {
            "freeze", "freeze_receipt", "commit", "commit_receipt"
        }:
            raise ObjectBongardRubricCampaignError(
                "query source opened before the durable decision"
            )
        freeze = freeze_state["freeze"]
        commit = freeze_state["commit"]
        freeze_receipt = freeze_state["freeze_receipt"]
        commit_receipt = freeze_state["commit_receipt"]
        if (
            not isinstance(freeze, ObjectBongardRubricTaskFreeze)
            or not isinstance(commit, ObjectBongardRubricTaskFreezeCommit)
            or not isinstance(freeze_receipt, ObjectBongardWriteOnceReceipt)
            or not isinstance(commit_receipt, ObjectBongardWriteOnceReceipt)
            or freeze.to_data() != dict(freeze_data)
            or commit.to_data() != dict(commit_data)
        ):
            raise ObjectBongardRubricCampaignError(
                "query source decision bindings differ"
            )
        result: dict[str, ObjectBongardRubricObserverArtifact] = {}
        for query_index, (side, panel_id) in enumerate(
            (
                ("side_0", task.side_0_query_panel_id),
                ("side_1", task.side_1_query_panel_id),
            )
        ):
            released, release_receipt = release_object_bongard_query_panel(
                prepared=prepared,
                archive=archive,
                panel_id=panel_id,
                task_freeze=freeze,
                task_commit=commit,
                task_freeze_receipt=freeze_receipt,
                task_commit_receipt=commit_receipt,
            )
            observation = _observe_released_panel(
                task_id=task.task_id,
                released=released,
                release_receipt=release_receipt,
                rubric_spec=spec,
                prepared=prepared,
                runtime=runtime.visual,
                journals_root=root,
                relative_prefix=f"tasks/{task.task_id}/query/{side}",
                turn_kind_prefix=f"q{query_index}",
                budget=budget,
                visual_transport=visual_transport,
            )
            query_observations.append(observation)
            result[side] = observation.artifact
        return result

    try:
        task_run = run_object_bongard_rubric_task(
            task,
            semantic,
            tuple(item.artifact for item in support_observations[:6]),
            tuple(item.artifact for item in support_observations[6:]),
            execution_precommit_digest=prepared.precommit.record_digest,
            ranker=journaled_ranker,
            freeze_committer=commit_freeze,
            freeze_reloader=reload_freeze,
            query_source=release_and_observe_queries,
        )
    except ObjectBongardRubricRankerError:
        if (
            journaled_ranker.summary is None
            or freeze_state
            or query_observations
        ):
            raise
        execution = ObjectBongardRubricTaskExecution.seal(
            campaign_source_digest=object_bongard_rubric_campaign_source_digest(),
            task_plan=task,
            execution_precommit_digest=prepared.precommit.record_digest,
            release_authorization_digest=prepared.authorization.record_digest,
            runtime_binding_digest=runtime.binding_digest,
            semantic_artifact=semantic,
            semantic_journal_relative_directory=semantic_relative,
            semantic_journal_summary=semantic_summary,
            support_observations=tuple(support_observations),
            task_run=None,
            rank_journal_relative_directory=rank_relative,
            rank_journal_summary=journaled_ranker.summary,
            query_observations=(),
            task_freeze_store_receipt=None,
            task_commit_store_receipt=None,
            correct_count=0,
            abstention_count=2,
        )
        return _persist_rubric_task_execution(prepared, execution)
    cold_replay_object_bongard_rubric_task(
        task_run, expected_archive_digest=task_run.record_digest
    )
    complete = task_run.status is ObjectBongardRubricTaskRunStatus.COMPLETE
    if complete and set(freeze_state) != {
        "freeze", "freeze_receipt", "commit", "commit_receipt"
    }:
        raise ObjectBongardRubricCampaignError(
            "complete task lacks the durable decision records"
        )
    if not complete and (freeze_state or query_observations):
        raise ObjectBongardRubricCampaignError(
            "support gap unexpectedly released query material"
        )
    execution = ObjectBongardRubricTaskExecution.seal(
        campaign_source_digest=object_bongard_rubric_campaign_source_digest(),
        task_plan=task,
        execution_precommit_digest=prepared.precommit.record_digest,
        release_authorization_digest=prepared.authorization.record_digest,
        runtime_binding_digest=runtime.binding_digest,
        semantic_artifact=semantic,
        semantic_journal_relative_directory=semantic_relative,
        semantic_journal_summary=semantic_summary,
        support_observations=tuple(support_observations),
        task_run=task_run,
        rank_journal_relative_directory=(rank_relative if complete else None),
        rank_journal_summary=(journaled_ranker.summary if complete else None),
        query_observations=tuple(query_observations),
        task_freeze_store_receipt=(
            freeze_state.get("freeze_receipt") if complete else None
        ),
        task_commit_store_receipt=(
            freeze_state.get("commit_receipt") if complete else None
        ),
        correct_count=task_run.correct_count if complete else 0,
        abstention_count=task_run.abstention_count if complete else 2,
    )
    return _persist_rubric_task_execution(prepared, execution)


def _forbidden_model_transport(*_args: object, **_kwargs: object) -> CodexStructuredResult:
    raise AssertionError("model transport called during campaign cold replay")


def _verify_journal_summary(
    actual: ObjectBongardTurnJournalSummary, expected: Mapping[str, Any]
) -> None:
    if actual.to_data() != dict(expected):
        raise ObjectBongardRubricCampaignError(
            "cold-replayed journal summary differs"
        )


def _cold_replay_panel_observation(
    observation: ObjectBongardRubricPanelObservation,
    *,
    task_id: str,
    expected_store_kind: str,
    turn_kind_prefix: str,
    prepared: PreparedObjectBongardRelease,
    archive: OfficialPanelArchive,
    runtime: ObjectBongardTurnRuntime,
    journals_root: Path,
) -> None:
    released = observation.released_panel
    receipt = observation.released_panel_store_receipt
    if receipt.object_kind != expected_store_kind:
        raise ObjectBongardRubricCampaignError(
            "released panel store kind differs during replay"
        )
    prepared.store.verify(receipt, expected_data=released.to_data())
    released.cold_verify(
        archive,
        expected_execution_precommit_digest=prepared.precommit.record_digest,
        expected_exposure_successor_digest=prepared.successor.digest,
    )
    png = released.exact_png_bytes
    rendered = dict(render_object_hypothesis_atlas(observation.hypothesis_packet, png))
    verify_object_hypothesis_packet(
        observation.hypothesis_packet,
        expected_png_bytes=png,
        expected_atlas_png_by_name=rendered,
    )
    verify_object_lineage_packet(observation.lineage_packet, png)
    verify_object_bongard_rubric_observer_artifact(
        observation.artifact,
        png,
        panel_id=released.panel_id,
        rubric_spec=observation.artifact.rubric_spec,
        hypothesis_packet=observation.hypothesis_packet,
        lineage_packet=observation.lineage_packet,
        expected_artifact_digest=observation.artifact.artifact_digest,
    )
    schema = object_bongard_rubric_observer_output_schema()
    for sheet, relative, expected_summary in zip(
        observation.hypothesis_packet.atlas_sheets,
        observation.journal_relative_directories,
        observation.journal_summaries,
        strict=True,
    ):
        journal = ObjectBongardNamedImageTurnJournalTransport(
            journals_root / relative,
            authorization_digest=prepared.authorization.record_digest,
            execution_precommit_digest=prepared.precommit.record_digest,
            task_id=task_id,
            turn_kind=f"{turn_kind_prefix}_{sheet.sheet_index:03d}",
            expected_prompt=object_bongard_rubric_observer_prompt(
                observation.artifact.rubric_spec, sheet
            ),
            expected_images=(
                ("scene.png", png),
                (sheet.name, rendered[sheet.name]),
            ),
            expected_output_schema=schema,
            runtime=runtime,
            underlying_transport=_forbidden_model_transport,
        )
        _verify_journal_summary(journal.verify(), expected_summary)


def cold_replay_object_bongard_rubric_campaign_task(
    execution: ObjectBongardRubricTaskExecution | Mapping[str, Any],
    *,
    expected_execution_digest: str,
    execution_store_receipt: ObjectBongardWriteOnceReceipt,
    prepared: PreparedObjectBongardRelease,
    archive: OfficialPanelArchive,
    runtime: ObjectBongardRubricCampaignRuntime,
    journals_root: str | Path,
) -> ObjectBongardRubricTaskExecution:
    """Verify official PNGs, artifacts, journals, freezes, and score with no model."""

    expected = _require_address(expected_execution_digest, "task execution digest")
    restored = (
        ObjectBongardRubricTaskExecution.from_data(execution)
        if isinstance(execution, Mapping)
        else ObjectBongardRubricTaskExecution.from_data(execution.to_data())
    )
    if (
        restored.record_digest != expected
        or restored.runtime_binding_digest != runtime.binding_digest
        or restored.execution_precommit_digest != prepared.precommit.record_digest
        or restored.release_authorization_digest
        != prepared.authorization.record_digest
        or execution_store_receipt.object_kind != "rubric-task-execution"
        or execution_store_receipt.object_digest != expected
    ):
        raise ObjectBongardRubricCampaignError(
            "cold task execution parent differs"
        )
    prepared = verify_prepared_object_bongard_release(prepared)
    prepared.store.verify(
        execution_store_receipt, expected_data=restored.to_data()
    )
    root = Path(journals_root).absolute()
    support = restored.support_observations
    semantic_failed = (
        restored.task_run is None
        and restored.semantic_artifact.status
        is not PrototypeSceneObserverStatus.SUCCESS
    )
    if semantic_failed:
        support_ids = (
            *restored.task_plan.side_0_support_panel_ids,
            *restored.task_plan.side_1_support_panel_ids,
        )
        released_support = tuple(
            release_object_bongard_support_panel(
                prepared=prepared, archive=archive, panel_id=panel_id
            )[0]
            for panel_id in support_ids
        )
        support_png = {
            item.panel_id: item.exact_png_bytes for item in released_support
        }
        semantic_pngs = tuple(item.exact_png_bytes for item in released_support)
    else:
        support_png = {
            item.released_panel.panel_id: item.released_panel.exact_png_bytes
            for item in support
        }
        semantic_pngs = tuple(
            item.released_panel.exact_png_bytes for item in support
        )
    verify_object_bongard_semantic_artifact(
        restored.semantic_artifact,
        support_png_by_panel_id=support_png,
        expected_task_id=restored.task_plan.task_id,
        expected_observation_context_digest=prepared.precommit.record_digest,
        expected_artifact_digest=restored.semantic_artifact.artifact_digest,
    )
    semantic_names = tuple(
        [f"group_0_ref_{index:02d}.png" for index in range(6)]
        + [f"group_1_ref_{index:02d}.png" for index in range(6)]
    )
    semantic_images = tuple(
        (name, png)
        for name, png in zip(semantic_names, semantic_pngs, strict=True)
    )
    semantic_journal = ObjectBongardNamedImageTurnJournalTransport(
        root / restored.semantic_journal_relative_directory,
        authorization_digest=prepared.authorization.record_digest,
        execution_precommit_digest=prepared.precommit.record_digest,
        task_id=restored.task_plan.task_id,
        turn_kind="semantic",
        expected_prompt=object_bongard_semantics_prompt(),
        expected_images=semantic_images,
        expected_output_schema=prototype_object_description_output_schema(),
        runtime=runtime.visual,
        underlying_transport=_forbidden_model_transport,
    )
    _verify_journal_summary(
        semantic_journal.verify(), restored.semantic_journal_summary
    )
    if semantic_failed:
        return restored
    for index, observation in enumerate(support):
        _cold_replay_panel_observation(
            observation,
            task_id=restored.task_plan.task_id,
            expected_store_kind="released-support-panel",
            turn_kind_prefix=f"s{index:02d}",
            prepared=prepared,
            archive=archive,
            runtime=runtime.visual,
            journals_root=root,
        )

    if restored.task_run is None:
        if (
            restored.rank_journal_relative_directory is None
            or restored.rank_journal_summary is None
        ):
            raise ObjectBongardRubricCampaignError(
                "rank-error cold replay lacks its terminal journal"
            )
        spec = ObjectBongardRubricSpec.from_semantic_artifact(
            restored.semantic_artifact,
            expected_artifact_digest=restored.semantic_artifact.artifact_digest,
        )
        positives = tuple(item.artifact for item in support[:6])
        negatives = tuple(item.artifact for item in support[6:])
        version = build_object_bongard_rubric_support_version_space(
            spec, positives, negatives
        )
        version = cold_verify_object_bongard_rubric_support_version_space(
            version, spec, positives, negatives
        )
        rank_input = object_bongard_rubric_rank_input_digest(
            version_space=version,
            rubric_spec=spec,
            semantic_artifact=restored.semantic_artifact,
            positive_support_artifacts=positives,
            negative_support_artifacts=negatives,
        )
        prompt = object_bongard_rubric_ranker_prompt(
            version_space=version,
            rubric_spec=spec,
            semantic_artifact=restored.semantic_artifact,
            positive_support_artifacts=positives,
            negative_support_artifacts=negatives,
            rank_input_digest=rank_input,
        )
        journal = ObjectBongardTextTurnJournalTransport(
            root / restored.rank_journal_relative_directory,
            authorization_digest=prepared.authorization.record_digest,
            execution_precommit_digest=prepared.precommit.record_digest,
            task_id=restored.task_plan.task_id,
            turn_kind="rank",
            expected_prompt=prompt,
            expected_output_schema=object_bongard_rubric_ranker_output_schema(),
            runtime=runtime.rank,
            underlying_transport=_forbidden_model_transport,
        )
        snapshot = runtime.rank.cloud_policy_cache_snapshot
        if snapshot is None:
            raise ObjectBongardRubricCampaignError(
                "rank-error cold replay lacks policy-cache snapshot"
            )
        ranker = ObjectBongardRubricRanker(
            model=runtime.rank.model,
            reasoning_effort=runtime.rank.reasoning_effort,
            minutes=runtime.rank.minutes,
            verbose=runtime.rank.verbose,
            executable=runtime.rank.executable,
            expected_launcher_digest=runtime.rank.expected_launcher_digest,
            cloud_policy_cache_snapshot=snapshot,
            expected_cloud_policy_cache_binding=runtime.rank.policy_cache_binding,
            expected_transport_source_digest=runtime.rank.transport_source_digest,
            model_catalog_snapshot=runtime.rank.model_catalog_snapshot,
            no_tools_attestation=runtime.rank.no_tools_attestation,
            transport=journal,
        )
        try:
            response = ranker(
                version,
                rubric_spec=spec,
                semantic_artifact=restored.semantic_artifact,
                positive_support_artifacts=positives,
                negative_support_artifacts=negatives,
                rank_input_digest=rank_input,
            )
            ranker.verify_response(
                response,
                version_space=version,
                rubric_spec=spec,
                semantic_artifact=restored.semantic_artifact,
                positive_support_artifacts=positives,
                negative_support_artifacts=negatives,
                rank_input_digest=rank_input,
                expected_response_digest=response.response_digest,
            )
            response.assert_matches(
                survivor_candidate_digests=version.survivor_candidate_digests,
                rubric_spec_digest=spec.spec_digest,
                semantic_artifact_digest=restored.semantic_artifact.artifact_digest,
                version_space_digest=version.version_space_digest,
                rank_input_digest=rank_input,
            )
        except ObjectBongardRubricRankerError:
            pass
        else:
            raise ObjectBongardRubricCampaignError(
                "rank-error journal now yields a valid rank response"
            )
        _verify_journal_summary(
            journal.verify(), restored.rank_journal_summary
        )
        return restored

    task_run = cold_replay_object_bongard_rubric_task(
        restored.task_run,
        expected_archive_digest=restored.task_run.record_digest,
    )
    if task_run.status is ObjectBongardRubricTaskRunStatus.COMPLETE:
        if (
            restored.rank_journal_relative_directory is None
            or restored.rank_journal_summary is None
            or task_run.rank_response is None
            or task_run.rank_input_digest is None
            or task_run.freeze is None
            or task_run.freeze_commit is None
            or restored.task_freeze_store_receipt is None
            or restored.task_commit_store_receipt is None
        ):
            raise ObjectBongardRubricCampaignError(
                "complete cold task lacks later-phase records"
            )
        prompt = object_bongard_rubric_ranker_prompt(
            version_space=task_run.version_space,
            rubric_spec=task_run.rubric_spec,
            semantic_artifact=task_run.semantic_artifact,
            positive_support_artifacts=task_run.side_0_support,
            negative_support_artifacts=task_run.side_1_support,
            rank_input_digest=task_run.rank_input_digest,
        )
        rank_journal = ObjectBongardTextTurnJournalTransport(
            root / restored.rank_journal_relative_directory,
            authorization_digest=prepared.authorization.record_digest,
            execution_precommit_digest=prepared.precommit.record_digest,
            task_id=restored.task_plan.task_id,
            turn_kind="rank",
            expected_prompt=prompt,
            expected_output_schema=object_bongard_rubric_ranker_output_schema(),
            runtime=runtime.rank,
            underlying_transport=_forbidden_model_transport,
        )
        _verify_journal_summary(
            rank_journal.verify(), restored.rank_journal_summary
        )
        snapshot = runtime.rank.cloud_policy_cache_snapshot
        if snapshot is None:
            raise ObjectBongardRubricCampaignError(
                "rank cold replay lacks policy-cache snapshot"
            )
        ranker = ObjectBongardRubricRanker(
            model=runtime.rank.model,
            reasoning_effort=runtime.rank.reasoning_effort,
            minutes=runtime.rank.minutes,
            verbose=runtime.rank.verbose,
            executable=runtime.rank.executable,
            expected_launcher_digest=runtime.rank.expected_launcher_digest,
            cloud_policy_cache_snapshot=snapshot,
            expected_cloud_policy_cache_binding=runtime.rank.policy_cache_binding,
            expected_transport_source_digest=runtime.rank.transport_source_digest,
            model_catalog_snapshot=runtime.rank.model_catalog_snapshot,
            no_tools_attestation=runtime.rank.no_tools_attestation,
            transport=_forbidden_model_transport,
        )
        ranker.verify_response(
            task_run.rank_response,
            version_space=task_run.version_space,
            rubric_spec=task_run.rubric_spec,
            semantic_artifact=task_run.semantic_artifact,
            positive_support_artifacts=task_run.side_0_support,
            negative_support_artifacts=task_run.side_1_support,
            rank_input_digest=task_run.rank_input_digest,
            expected_response_digest=task_run.rank_response.response_digest,
        )
        freeze_bytes = canonical_json(task_run.freeze.to_data()) + b"\n"
        task_run.freeze_commit.assert_matches(task_run.freeze, freeze_bytes)
        prepared.store.verify(
            restored.task_freeze_store_receipt,
            expected_data=task_run.freeze.to_data(),
        )
        prepared.store.verify(
            restored.task_commit_store_receipt,
            expected_data=task_run.freeze_commit.to_data(),
        )
        for index, observation in enumerate(restored.query_observations):
            _cold_replay_panel_observation(
                observation,
                task_id=restored.task_plan.task_id,
                expected_store_kind="released-query-panel",
                turn_kind_prefix=f"q{index}",
                prepared=prepared,
                archive=archive,
                runtime=runtime.visual,
                journals_root=root,
            )
    return restored


def _campaign_content(value: "ObjectBongardRubricCampaignArchive") -> dict[str, object]:
    return {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "campaign_source_digest": value.campaign_source_digest,
        "plan": value.plan.to_data(),
        "execution_precommit": value.execution_precommit.to_data(),
        "exposure_predecessor": value.exposure_predecessor.to_dict(),
        "exposure_successor": value.exposure_successor.to_dict(),
        "release_authorization": value.release_authorization.to_data(),
        "plan_store_receipt": value.plan_store_receipt.to_data(),
        "precommit_store_receipt": value.precommit_store_receipt.to_data(),
        "exposure_store_receipt": value.exposure_store_receipt.to_data(),
        "authorization_store_receipt": value.authorization_store_receipt.to_data(),
        "runtime_binding_digest": value.runtime_binding_digest,
        "max_workers": value.max_workers,
        "max_physical_model_calls": value.max_physical_model_calls,
        "physical_model_calls": value.physical_model_calls,
        "physical_model_calls_by_kind": [
            list(item) for item in value.physical_model_calls_by_kind
        ],
        "task_executions": [item.to_data() for item in value.task_executions],
        "task_execution_store_receipts": [
            item.to_data() for item in value.task_execution_store_receipts
        ],
        "task_count": len(value.task_executions),
        "complete_task_count": value.complete_task_count,
        "gap_task_count": value.gap_task_count,
        "correct_count": value.correct_count,
        "abstention_count": value.abstention_count,
        "fixed_score_denominator": value.fixed_score_denominator,
        "accuracy_ppm": value.accuracy_ppm,
        "every_planned_task_contributes_exactly_two_query_slots": True,
        "gap_contributes_zero_correct_and_two_abstentions": True,
        "campaign_cold_replay_model_calls": 0,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCampaignArchive:
    campaign_source_digest: str
    plan: ObjectBongardBatchPlan
    execution_precommit: ObjectBongardExecutionPrecommit
    exposure_predecessor: ExposureLedger
    exposure_successor: ExposureLedger
    release_authorization: ObjectBongardReleaseAuthorization
    plan_store_receipt: ObjectBongardWriteOnceReceipt
    precommit_store_receipt: ObjectBongardWriteOnceReceipt
    exposure_store_receipt: ObjectBongardWriteOnceReceipt
    authorization_store_receipt: ObjectBongardWriteOnceReceipt
    runtime_binding_digest: str
    max_workers: int
    max_physical_model_calls: int
    physical_model_calls: int
    physical_model_calls_by_kind: tuple[tuple[str, int], ...]
    task_executions: tuple[ObjectBongardRubricTaskExecution, ...]
    task_execution_store_receipts: tuple[ObjectBongardWriteOnceReceipt, ...]
    complete_task_count: int
    gap_task_count: int
    correct_count: int
    abstention_count: int
    fixed_score_denominator: int
    accuracy_ppm: int
    record_digest: str

    def __post_init__(self) -> None:
        if self.campaign_source_digest != object_bongard_rubric_campaign_source_digest():
            raise ObjectBongardRubricCampaignError(
                "campaign archive source differs"
            )
        _verify_launch_gate_configuration(
            self.execution_precommit.configuration
        )
        for name in ("runtime_binding_digest", "record_digest"):
            _require_address(getattr(self, name), name)
        if (
            self.execution_precommit.batch_plan_digest != self.plan.record_digest
            or self.execution_precommit.exposure_predecessor_digest
            != self.exposure_predecessor.digest
            or self.release_authorization.execution_precommit_digest
            != self.execution_precommit.record_digest
            or self.release_authorization.exposure_successor_digest
            != self.exposure_successor.digest
            or self.plan_store_receipt.object_digest != self.plan.record_digest
            or self.precommit_store_receipt.object_digest
            != self.execution_precommit.record_digest
            or self.exposure_store_receipt.object_digest
            != self.exposure_successor.digest
            or self.authorization_store_receipt.object_digest
            != self.release_authorization.record_digest
            or self.plan_store_receipt.record_digest
            != self.release_authorization.plan_store_receipt_digest
            or self.precommit_store_receipt.record_digest
            != self.release_authorization.precommit_store_receipt_digest
            or self.exposure_store_receipt.record_digest
            != self.release_authorization.exposure_store_receipt_digest
            or dict(self.execution_precommit.runtime_source_bindings)
            != object_bongard_rubric_campaign_source_bindings()
            or dict(self.execution_precommit.configuration).get(
                "runtime_binding_digest"
            )
            != self.runtime_binding_digest
            or dict(self.execution_precommit.configuration).get("max_workers")
            != self.max_workers
            or dict(self.execution_precommit.configuration).get(
                "max_physical_model_calls"
            )
            != self.max_physical_model_calls
        ):
            raise ObjectBongardRubricCampaignError(
                "campaign durable release parents differ"
            )
        task_ids = tuple(item.task_plan.task_id for item in self.task_executions)
        expected_task_ids = tuple(item.task_id for item in self.plan.tasks)
        if (
            task_ids != expected_task_ids
            or len(self.task_execution_store_receipts) != len(self.task_executions)
            or any(
                receipt.object_kind != "rubric-task-execution"
                or receipt.object_digest != execution.record_digest
                for execution, receipt in zip(
                    self.task_executions,
                    self.task_execution_store_receipts,
                    strict=True,
                )
            )
            or any(
                execution.execution_precommit_digest
                != self.execution_precommit.record_digest
                or execution.release_authorization_digest
                != self.release_authorization.record_digest
                or execution.runtime_binding_digest != self.runtime_binding_digest
                for execution in self.task_executions
            )
        ):
            raise ObjectBongardRubricCampaignError(
                "campaign task inventory or parent bindings differ"
            )
        count = len(self.task_executions)
        complete = sum(
            item.task_run is not None
            and item.task_run.status is ObjectBongardRubricTaskRunStatus.COMPLETE
            for item in self.task_executions
        )
        correct = sum(item.correct_count for item in self.task_executions)
        abstentions = sum(item.abstention_count for item in self.task_executions)
        denominator = count * 2
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or not 1 <= self.max_workers <= 12
            or isinstance(self.max_physical_model_calls, bool)
            or not isinstance(self.max_physical_model_calls, int)
            or self.max_physical_model_calls <= 0
            or isinstance(self.physical_model_calls, bool)
            or not isinstance(self.physical_model_calls, int)
            or not 0 <= self.physical_model_calls <= self.max_physical_model_calls
            or self.physical_model_calls_by_kind
            != tuple(sorted(self.physical_model_calls_by_kind))
            or sum(item[1] for item in self.physical_model_calls_by_kind)
            != self.physical_model_calls
            or self.complete_task_count != complete
            or self.gap_task_count != count - complete
            or self.correct_count != correct
            or self.abstention_count != abstentions
            or self.fixed_score_denominator != denominator
            or self.accuracy_ppm
            != (0 if denominator == 0 else correct * 1_000_000 // denominator)
            or (self.plan.requested_per_family == 4 and denominator != 24)
        ):
            raise ObjectBongardRubricCampaignError(
                "campaign aggregate or fixed denominator differs"
            )
        if self.record_digest != _address(_campaign_content(self)):
            raise ObjectBongardRubricCampaignError(
                "campaign archive content digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_campaign_content(self), "record_digest": self.record_digest}

    @classmethod
    def seal(cls, **values: object) -> "ObjectBongardRubricCampaignArchive":
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(_campaign_content(provisional)),
        )

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "ObjectBongardRubricCampaignArchive":
        required = {
            "schema", "campaign_id", "campaign_source_digest", "plan",
            "execution_precommit", "exposure_predecessor", "exposure_successor",
            "release_authorization", "plan_store_receipt",
            "precommit_store_receipt", "exposure_store_receipt",
            "authorization_store_receipt", "runtime_binding_digest",
            "max_workers", "max_physical_model_calls", "physical_model_calls",
            "physical_model_calls_by_kind", "task_executions",
            "task_execution_store_receipts", "task_count",
            "complete_task_count", "gap_task_count", "correct_count",
            "abstention_count", "fixed_score_denominator", "accuracy_ppm",
            "every_planned_task_contributes_exactly_two_query_slots",
            "gap_contributes_zero_correct_and_two_abstentions",
            "campaign_cold_replay_model_calls", *_authority_data(),
            "record_digest",
        }
        mapping_fields = (
            "plan", "execution_precommit", "exposure_predecessor",
            "exposure_successor", "release_authorization", "plan_store_receipt",
            "precommit_store_receipt", "exposure_store_receipt",
            "authorization_store_receipt",
        )
        if (
            not isinstance(value, Mapping)
            or set(value) != required
            or value["schema"] != CAMPAIGN_SCHEMA
            or value["campaign_id"] != CAMPAIGN_ID
            or value["every_planned_task_contributes_exactly_two_query_slots"]
            is not True
            or value["gap_contributes_zero_correct_and_two_abstentions"]
            is not True
            or value["campaign_cold_replay_model_calls"] != 0
            or any(value[key] != item for key, item in _authority_data().items())
            or any(not isinstance(value[key], Mapping) for key in mapping_fields)
            or not isinstance(value["physical_model_calls_by_kind"], list)
            or not isinstance(value["task_executions"], list)
            or not isinstance(value["task_execution_store_receipts"], list)
        ):
            raise ObjectBongardRubricCampaignError(
                "campaign archive fields differ"
            )
        executions = tuple(
            ObjectBongardRubricTaskExecution.from_data(item)
            for item in value["task_executions"]
        )
        result = cls(
            campaign_source_digest=value["campaign_source_digest"],
            plan=ObjectBongardBatchPlan.from_data(value["plan"]),
            execution_precommit=ObjectBongardExecutionPrecommit.from_data(
                value["execution_precommit"]
            ),
            exposure_predecessor=ExposureLedger.from_dict(
                value["exposure_predecessor"]
            ),
            exposure_successor=ExposureLedger.from_dict(
                value["exposure_successor"]
            ),
            release_authorization=ObjectBongardReleaseAuthorization.from_data(
                value["release_authorization"]
            ),
            plan_store_receipt=ObjectBongardWriteOnceReceipt.from_data(
                value["plan_store_receipt"]
            ),
            precommit_store_receipt=ObjectBongardWriteOnceReceipt.from_data(
                value["precommit_store_receipt"]
            ),
            exposure_store_receipt=ObjectBongardWriteOnceReceipt.from_data(
                value["exposure_store_receipt"]
            ),
            authorization_store_receipt=ObjectBongardWriteOnceReceipt.from_data(
                value["authorization_store_receipt"]
            ),
            runtime_binding_digest=value["runtime_binding_digest"],
            max_workers=value["max_workers"],
            max_physical_model_calls=value["max_physical_model_calls"],
            physical_model_calls=value["physical_model_calls"],
            physical_model_calls_by_kind=tuple(
                tuple(item) for item in value["physical_model_calls_by_kind"]
            ),
            task_executions=executions,
            task_execution_store_receipts=tuple(
                ObjectBongardWriteOnceReceipt.from_data(item)
                for item in value["task_execution_store_receipts"]
            ),
            complete_task_count=value["complete_task_count"],
            gap_task_count=value["gap_task_count"],
            correct_count=value["correct_count"],
            abstention_count=value["abstention_count"],
            fixed_score_denominator=value["fixed_score_denominator"],
            accuracy_ppm=value["accuracy_ppm"],
            record_digest=value["record_digest"],
        )
        if value["task_count"] != len(executions) or result.to_data() != dict(value):
            raise ObjectBongardRubricCampaignError(
                "campaign archive is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class PersistedObjectBongardRubricCampaign:
    archive: ObjectBongardRubricCampaignArchive
    store_receipt: ObjectBongardWriteOnceReceipt

    def __post_init__(self) -> None:
        if (
            not isinstance(self.archive, ObjectBongardRubricCampaignArchive)
            or not isinstance(self.store_receipt, ObjectBongardWriteOnceReceipt)
            or self.store_receipt.object_kind != "rubric-campaign"
            or self.store_receipt.object_digest != self.archive.record_digest
        ):
            raise ObjectBongardRubricCampaignError(
                "persisted campaign binding differs"
            )


def run_object_bongard_rubric_campaign(
    *,
    prepared: PreparedObjectBongardRelease,
    archive: OfficialPanelArchive,
    runtime: ObjectBongardRubricCampaignRuntime,
    journals_root: str | Path,
    visual_transport: Callable[..., CodexStructuredResult],
    rank_transport: Callable[..., CodexStructuredResult],
) -> PersistedObjectBongardRubricCampaign:
    """Execute every frozen task with bounded concurrency and one call budget."""

    prepared = verify_prepared_object_bongard_release(prepared)
    budget = ObjectBongardPhysicalCallBudget(runtime.max_physical_model_calls)
    results: dict[str, PersistedObjectBongardRubricTaskExecution] = {}
    with ThreadPoolExecutor(max_workers=runtime.max_workers) as executor:
        futures = {
            executor.submit(
                run_object_bongard_rubric_campaign_task,
                task=task,
                prepared=prepared,
                archive=archive,
                runtime=runtime,
                journals_root=journals_root,
                budget=budget,
                visual_transport=visual_transport,
                rank_transport=rank_transport,
            ): task.task_id
            for task in prepared.plan.tasks
        }
        for future in as_completed(futures):
            task_id = futures[future]
            if task_id in results:
                raise ObjectBongardRubricCampaignError(
                    "campaign completed one task more than once"
                )
            results[task_id] = future.result()
    ordered = tuple(results[task.task_id] for task in prepared.plan.tasks)
    executions = tuple(item.execution for item in ordered)
    receipts = tuple(item.store_receipt for item in ordered)
    complete = sum(
        item.task_run is not None
        and item.task_run.status is ObjectBongardRubricTaskRunStatus.COMPLETE
        for item in executions
    )
    correct = sum(item.correct_count for item in executions)
    abstentions = sum(item.abstention_count for item in executions)
    denominator = len(executions) * 2
    campaign = ObjectBongardRubricCampaignArchive.seal(
        campaign_source_digest=object_bongard_rubric_campaign_source_digest(),
        plan=prepared.plan,
        execution_precommit=prepared.precommit,
        exposure_predecessor=prepared.predecessor,
        exposure_successor=prepared.successor,
        release_authorization=prepared.authorization,
        plan_store_receipt=prepared.plan_receipt,
        precommit_store_receipt=prepared.precommit_receipt,
        exposure_store_receipt=prepared.exposure_receipt,
        authorization_store_receipt=prepared.authorization_receipt,
        runtime_binding_digest=runtime.binding_digest,
        max_workers=runtime.max_workers,
        max_physical_model_calls=runtime.max_physical_model_calls,
        physical_model_calls=budget.count,
        physical_model_calls_by_kind=tuple(budget.by_kind.items()),
        task_executions=executions,
        task_execution_store_receipts=receipts,
        complete_task_count=complete,
        gap_task_count=len(executions) - complete,
        correct_count=correct,
        abstention_count=abstentions,
        fixed_score_denominator=denominator,
        accuracy_ppm=0 if denominator == 0 else correct * 1_000_000 // denominator,
    )
    receipt = prepared.store.persist(
        object_kind="rubric-campaign",
        object_digest=campaign.record_digest,
        data=campaign.to_data(),
    )
    prepared.store.verify(receipt, expected_data=campaign.to_data())
    return PersistedObjectBongardRubricCampaign(campaign, receipt)


def cold_replay_object_bongard_rubric_campaign(
    campaign: ObjectBongardRubricCampaignArchive | Mapping[str, Any],
    *,
    expected_campaign_digest: str,
    campaign_store_receipt: ObjectBongardWriteOnceReceipt,
    store: ObjectBongardReleaseStore,
    archive: OfficialPanelArchive,
    runtime: ObjectBongardRubricCampaignRuntime,
    journals_root: str | Path,
) -> ObjectBongardRubricCampaignArchive:
    """Cold-replay the whole durable campaign without accepting transports."""

    expected = _require_address(expected_campaign_digest, "campaign digest")
    restored = (
        ObjectBongardRubricCampaignArchive.from_data(campaign)
        if isinstance(campaign, Mapping)
        else ObjectBongardRubricCampaignArchive.from_data(campaign.to_data())
    )
    if (
        restored.record_digest != expected
        or restored.runtime_binding_digest != runtime.binding_digest
        or campaign_store_receipt.object_kind != "rubric-campaign"
        or campaign_store_receipt.object_digest != expected
        or restored.execution_precommit.archive_record_digest
        != archive.record_digest
    ):
        raise ObjectBongardRubricCampaignError(
            "campaign cold replay parents differ"
        )
    prepared = PreparedObjectBongardRelease(
        store=store,
        plan=restored.plan,
        precommit=restored.execution_precommit,
        predecessor=restored.exposure_predecessor,
        successor=restored.exposure_successor,
        authorization=restored.release_authorization,
        plan_receipt=restored.plan_store_receipt,
        precommit_receipt=restored.precommit_store_receipt,
        exposure_receipt=restored.exposure_store_receipt,
        authorization_receipt=restored.authorization_store_receipt,
    )
    prepared = verify_prepared_object_bongard_release(prepared)
    store.verify(campaign_store_receipt, expected_data=restored.to_data())
    replayed = tuple(
        cold_replay_object_bongard_rubric_campaign_task(
            execution,
            expected_execution_digest=execution.record_digest,
            execution_store_receipt=receipt,
            prepared=prepared,
            archive=archive,
            runtime=runtime,
            journals_root=journals_root,
        )
        for execution, receipt in zip(
            restored.task_executions,
            restored.task_execution_store_receipts,
            strict=True,
        )
    )
    if replayed != restored.task_executions:
        raise ObjectBongardRubricCampaignError(
            "campaign task cold replay differs"
        )
    return restored
