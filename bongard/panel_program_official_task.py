"""Dormant official-custody adapter for frozen panel-program predicates.

This module is an integration boundary, not launch authority.  It never opens
an official archive for support data, never issues synthetic provenance, and
never chooses a task.  Support preparation accepts only exact
``ReleasedOfficialPanel`` records that a caller has already obtained through
the release gate.  The injected observer receives one value: the exact PNG
bytes from each released record.

The current fixed-catalog connected-program observer has *not* been validated
for real Bongard-LOGO geometry.  Consequently this adapter remains dormant for
official benchmarking until a separately authorized cohort and scientifically
applicable observer are preregistered.  Synthetic tests exercise the custody
machinery without weakening that boundary.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Callable, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_release_gate import (
    ObjectBongardWriteOnceReceipt,
    PreparedObjectBongardRelease,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
    release_object_bongard_query_panel,
    verify_prepared_object_bongard_release,
)
from bongard.official_panel_archive import OfficialPanelArchive, ReleasedOfficialPanel
from bongard.panel_program_observation import (
    PanelProgramObservation,
    connected_program_observer_bindings,
    observe_authenticated_program_png,
)
from bongard import panel_program_predicate as program_predicate
from bongard.panel_program_predicate import (
    FrozenProgramRule,
    ProgramRuleDecision,
    ProgramVersionSpace,
    build_program_version_space,
    evaluate_frozen_program_rule,
    freeze_program_rule,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


PANEL_PROGRAM_OFFICIAL_SUPPORT_PANEL_SCHEMA = (
    "gkm.panel-program-official-support-panel.v2"
)
PANEL_PROGRAM_OFFICIAL_SUPPORT_SCHEMA = "gkm.panel-program-official-support.v2"
PANEL_PROGRAM_OFFICIAL_TASK_FREEZE_SCHEMA = (
    "gkm.panel-program-official-task-freeze.v2"
)
PANEL_PROGRAM_OFFICIAL_TASK_COMMIT_SCHEMA = (
    "gkm.panel-program-official-task-commit.v2"
)
PANEL_PROGRAM_OFFICIAL_QUERY_SCHEMA = "gkm.panel-program-official-query.v2"
PANEL_PROGRAM_OFFICIAL_ADAPTER_ID = (
    "bongard.panel-program-official-task/dormant-renderer-grammar-custody-v2"
)
PANEL_PROGRAM_SUPPORT_COUNT = 12
PANEL_PROGRAM_SUPPORT_BUCKET_SIZE = 6

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PANEL_ALIAS = re.compile(r"panel_[0-9]{3}\Z")


class PanelProgramOfficialTaskError(ValueError):
    """Official custody, semantic replay, or durable binding differs."""


ProgramObserver = Callable[[bytes], PanelProgramObservation]


def _require_address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise PanelProgramOfficialTaskError(f"{label} must be a sha256: address")
    return value


def _require_raw_digest(value: object, label: str) -> str:
    if type(value) is not str or _RAW_DIGEST.fullmatch(value) is None:
        raise PanelProgramOfficialTaskError(
            f"{label} must be a raw lowercase SHA-256 digest"
        )
    return value


def _content_address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _bytes_address(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or set(value) != set(expected)
    ):
        raise PanelProgramOfficialTaskError(f"{label} fields differ")
    return value


def _has_bytes(value: object) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, Mapping):
        return any(_has_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_bytes(item) for item in value)
    return False


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "model_calls_permitted": False,
        "official_test_pixels_permitted": False,
        "clean_train_task_required": True,
        "direct_archive_support_reads_permitted": False,
        "synthetic_issuer_bypass_available": False,
        "adapter_confers_launch_authority": False,
        "fixed_catalog_observer_validated_for_official_geometry": False,
    }


def panel_program_official_task_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def panel_program_official_task_algorithm_digest() -> str:
    return _content_address(
        {
            "schema": "gkm.panel-program-official-task-algorithm.v2",
            "adapter_id": PANEL_PROGRAM_OFFICIAL_ADAPTER_ID,
            "source_digest": panel_program_official_task_source_digest(),
            "support_order": "side-0-positive-six-then-side-1-contrast-six",
            "observer_contract": "exact-png-bytes-only",
            "version_space_rule": "strict-positive-all-and-contrast-all",
            "selection_rule": "minimum-atom-count-then-formula-digest",
            "query_rule": "durable-exact-freeze-and-commit-before-release",
            **_authority_data(),
        }
    )


def panel_program_required_precommit_bindings() -> dict[str, str]:
    """Exact semantic implementation addresses required before panel release."""

    return {
        **connected_program_observer_bindings(),
        "panel_program_predicate_source": "sha256:" + program_predicate.source_sha256(),
        "panel_program_official_task_source": (
            "sha256:" + panel_program_official_task_source_digest()
        ),
    }


def _require_exact_observer(observe_program: object) -> None:
    if observe_program is not observe_authenticated_program_png:
        raise PanelProgramOfficialTaskError(
            "observer callback is not the exact precommitted panel-program observer"
        )


def _observation_panel_digest(observation: PanelProgramObservation) -> str:
    """Read the canonical exact-PNG binding from the semantic observation."""

    try:
        value = observation.panel_png_digest
    except AttributeError as exc:  # fail closed if the semantic contract drifts
        raise PanelProgramOfficialTaskError(
            "panel-program observation lacks panel_png_digest"
        ) from exc
    return _require_address(value, "observation panel PNG digest")


def _support_panel_content(
    value: "PanelProgramOfficialSupportPanel",
) -> dict[str, object]:
    return {
        "schema": PANEL_PROGRAM_OFFICIAL_SUPPORT_PANEL_SCHEMA,
        "adapter_id": PANEL_PROGRAM_OFFICIAL_ADAPTER_ID,
        "panel_alias": value.panel_alias,
        "support_role": value.support_role,
        "source_ordinal": value.source_ordinal,
        "panel_id": value.panel_id,
        "released_panel_record_digest": value.released_panel_record_digest,
        "release_receipt_digest": value.release_receipt_digest,
        "released_panel_store_receipt": value.released_panel_store_receipt.to_data(),
        "released_panel_store_receipt_digest": value.released_panel_store_receipt_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "archive_digest": value.archive_digest,
        "archive_central_directory_digest": value.archive_central_directory_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "exact_png_digest": value.exact_png_digest,
        "exact_png_byte_count": value.exact_png_byte_count,
        "observation": value.observation.to_data(),
        "observation_digest": value.observation_digest,
        "raw_png_bytes_persisted": False,
    }


@dataclass(frozen=True, slots=True)
class PanelProgramOfficialSupportPanel:
    panel_alias: str
    support_role: str
    source_ordinal: int
    panel_id: str
    released_panel_record_digest: str
    release_receipt_digest: str
    released_panel_store_receipt: ObjectBongardWriteOnceReceipt
    released_panel_store_receipt_digest: str
    release_descriptor_digest: str
    archive_digest: str
    archive_central_directory_digest: str
    execution_precommit_digest: str
    exposure_successor_digest: str
    exact_png_digest: str
    exact_png_byte_count: int
    observation: PanelProgramObservation
    observation_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.panel_alias) is not str
            or _PANEL_ALIAS.fullmatch(self.panel_alias) is None
            or self.support_role not in ("positive", "contrast")
            or type(self.source_ordinal) is not int
            or not 0 <= self.source_ordinal < PANEL_PROGRAM_SUPPORT_COUNT
            or type(self.panel_id) is not str
            or not self.panel_id
        ):
            raise PanelProgramOfficialTaskError("support panel identity differs")
        expected_role = (
            "positive"
            if self.source_ordinal < PANEL_PROGRAM_SUPPORT_BUCKET_SIZE
            else "contrast"
        )
        if self.support_role != expected_role:
            raise PanelProgramOfficialTaskError("support panel role differs")
        for label, item in (
            ("released panel record digest", self.released_panel_record_digest),
            ("release receipt digest", self.release_receipt_digest),
            ("released panel store receipt digest", self.released_panel_store_receipt_digest),
            ("release descriptor digest", self.release_descriptor_digest),
            ("archive digest", self.archive_digest),
            ("archive central-directory digest", self.archive_central_directory_digest),
            ("execution precommit digest", self.execution_precommit_digest),
            ("exposure successor digest", self.exposure_successor_digest),
            ("exact PNG digest", self.exact_png_digest),
            ("observation digest", self.observation_digest),
            ("support panel record digest", self.record_digest),
        ):
            _require_address(item, label)
        if type(self.exact_png_byte_count) is not int or self.exact_png_byte_count <= 0:
            raise PanelProgramOfficialTaskError("support PNG byte count differs")
        if type(self.observation) is not PanelProgramObservation:
            raise TypeError("observation must be exact PanelProgramObservation")
        if type(self.released_panel_store_receipt) is not ObjectBongardWriteOnceReceipt:
            raise TypeError("released panel store receipt has the wrong type")
        store_receipt = ObjectBongardWriteOnceReceipt.from_data(
            self.released_panel_store_receipt.to_data()
        )
        restored = PanelProgramObservation.from_data(self.observation.to_data())
        if (
            restored != self.observation
            or store_receipt.object_kind != "released-support-panel"
            or store_receipt.object_digest != self.released_panel_record_digest
            or store_receipt.record_digest != self.released_panel_store_receipt_digest
            or self.observation.observation_digest != self.observation_digest
            or _observation_panel_digest(self.observation) != self.exact_png_digest
            or self.record_digest != _content_address(_support_panel_content(self))
            or _has_bytes(_support_panel_content(self))
        ):
            raise PanelProgramOfficialTaskError("support observation binding differs")

    def to_data(self) -> dict[str, object]:
        return {**_support_panel_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelProgramOfficialSupportPanel":
        raw = _fields(
            value,
            {
                "schema", "adapter_id", "panel_alias", "support_role",
                "source_ordinal", "panel_id", "released_panel_record_digest",
                "release_receipt_digest", "released_panel_store_receipt",
                "released_panel_store_receipt_digest", "release_descriptor_digest",
                "archive_digest", "archive_central_directory_digest",
                "execution_precommit_digest", "exposure_successor_digest",
                "exact_png_digest", "exact_png_byte_count", "observation",
                "observation_digest", "raw_png_bytes_persisted", "record_digest",
            },
            "panel-program official support panel",
        )
        if (
            raw["schema"] != PANEL_PROGRAM_OFFICIAL_SUPPORT_PANEL_SCHEMA
            or raw["adapter_id"] != PANEL_PROGRAM_OFFICIAL_ADAPTER_ID
            or raw["raw_png_bytes_persisted"] is not False
            or not isinstance(raw["observation"], Mapping)
            or not isinstance(raw["released_panel_store_receipt"], Mapping)
        ):
            raise PanelProgramOfficialTaskError("support panel policy differs")
        result = cls(
            raw["panel_alias"], raw["support_role"], raw["source_ordinal"],
            raw["panel_id"], raw["released_panel_record_digest"],
            raw["release_receipt_digest"],
            ObjectBongardWriteOnceReceipt.from_data(raw["released_panel_store_receipt"]),
            raw["released_panel_store_receipt_digest"],
            raw["release_descriptor_digest"],
            raw["archive_digest"], raw["archive_central_directory_digest"],
            raw["execution_precommit_digest"], raw["exposure_successor_digest"],
            raw["exact_png_digest"], raw["exact_png_byte_count"],
            PanelProgramObservation.from_data(raw["observation"]),
            raw["observation_digest"], raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelProgramOfficialTaskError("support panel is not canonical")
        return result


def _support_content(value: "PanelProgramOfficialSupportArtifact") -> dict[str, object]:
    return {
        "schema": PANEL_PROGRAM_OFFICIAL_SUPPORT_SCHEMA,
        "adapter_id": PANEL_PROGRAM_OFFICIAL_ADAPTER_ID,
        "adapter_algorithm_digest": value.adapter_algorithm_digest,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "prepared_batch_plan_digest": value.prepared_batch_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "release_authorization_digest": value.release_authorization_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "plan_store_receipt_digest": value.plan_store_receipt_digest,
        "precommit_store_receipt_digest": value.precommit_store_receipt_digest,
        "exposure_store_receipt_digest": value.exposure_store_receipt_digest,
        "authorization_store_receipt_digest": value.authorization_store_receipt_digest,
        "support_panel_ids": list(value.support_panel_ids),
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "support_panels": [item.to_data() for item in value.support_panels],
        "observer_algorithm_digest": value.observer_algorithm_digest,
        "search_space_digest": value.search_space_digest,
        "hypothesis_policy_digest": value.hypothesis_policy_digest,
        "support_order": "side-0-positive-six-then-side-1-contrast-six",
        "query_pixels_observed": False,
        "raw_png_bytes_persisted": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelProgramOfficialSupportArtifact:
    adapter_algorithm_digest: str
    task_id: str
    task_plan_digest: str
    prepared_batch_plan_digest: str
    execution_precommit_digest: str
    release_authorization_digest: str
    exposure_successor_digest: str
    plan_store_receipt_digest: str
    precommit_store_receipt_digest: str
    exposure_store_receipt_digest: str
    authorization_store_receipt_digest: str
    support_panel_ids: tuple[str, ...]
    sealed_query_panel_ids: tuple[str, str]
    support_panels: tuple[PanelProgramOfficialSupportPanel, ...]
    observer_algorithm_digest: str
    search_space_digest: str
    hypothesis_policy_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("adapter algorithm digest", self.adapter_algorithm_digest),
            ("task plan digest", self.task_plan_digest),
            ("prepared batch plan digest", self.prepared_batch_plan_digest),
            ("execution precommit digest", self.execution_precommit_digest),
            ("release authorization digest", self.release_authorization_digest),
            ("exposure successor digest", self.exposure_successor_digest),
            ("plan store receipt digest", self.plan_store_receipt_digest),
            ("precommit store receipt digest", self.precommit_store_receipt_digest),
            ("exposure store receipt digest", self.exposure_store_receipt_digest),
            ("authorization store receipt digest", self.authorization_store_receipt_digest),
            ("observer algorithm digest", self.observer_algorithm_digest),
            ("search-space digest", self.search_space_digest),
            ("hypothesis policy digest", self.hypothesis_policy_digest),
            ("support artifact digest", self.record_digest),
        ):
            _require_address(item, label)
        expected_aliases = tuple(
            f"panel_{index:03d}" for index in range(PANEL_PROGRAM_SUPPORT_COUNT)
        )
        if (
            type(self.task_id) is not str
            or not self.task_id
            or self.adapter_algorithm_digest
            != panel_program_official_task_algorithm_digest()
            or type(self.support_panel_ids) is not tuple
            or len(self.support_panel_ids) != PANEL_PROGRAM_SUPPORT_COUNT
            or len(set(self.support_panel_ids)) != PANEL_PROGRAM_SUPPORT_COUNT
            or type(self.sealed_query_panel_ids) is not tuple
            or len(self.sealed_query_panel_ids) != 2
            or len(set(self.sealed_query_panel_ids)) != 2
            or set(self.support_panel_ids) & set(self.sealed_query_panel_ids)
            or type(self.support_panels) is not tuple
            or len(self.support_panels) != PANEL_PROGRAM_SUPPORT_COUNT
            or any(type(item) is not PanelProgramOfficialSupportPanel for item in self.support_panels)
            or tuple(item.panel_alias for item in self.support_panels) != expected_aliases
            or tuple(item.source_ordinal for item in self.support_panels)
            != tuple(range(PANEL_PROGRAM_SUPPORT_COUNT))
            or tuple(item.panel_id for item in self.support_panels) != self.support_panel_ids
            or any(
                item.execution_precommit_digest != self.execution_precommit_digest
                or item.exposure_successor_digest != self.exposure_successor_digest
                or item.observation.observer_algorithm_digest != self.observer_algorithm_digest
                or item.observation.search_space_digest != self.search_space_digest
                or item.observation.hypothesis_policy_digest != self.hypothesis_policy_digest
                for item in self.support_panels
            )
            or self.record_digest != _content_address(_support_content(self))
            or _has_bytes(_support_content(self))
        ):
            raise PanelProgramOfficialTaskError("support artifact inventory differs")

    @property
    def positive_observations(self) -> tuple[PanelProgramObservation, ...]:
        return tuple(item.observation for item in self.support_panels[:6])

    @property
    def contrast_observations(self) -> tuple[PanelProgramObservation, ...]:
        return tuple(item.observation for item in self.support_panels[6:])

    def to_data(self) -> dict[str, object]:
        return {**_support_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelProgramOfficialSupportArtifact":
        raw = _fields(
            value,
            {
                "schema", "adapter_id", "adapter_algorithm_digest", "task_id",
                "task_plan_digest", "prepared_batch_plan_digest",
                "execution_precommit_digest", "release_authorization_digest",
                "exposure_successor_digest", "plan_store_receipt_digest",
                "precommit_store_receipt_digest", "exposure_store_receipt_digest",
                "authorization_store_receipt_digest", "support_panel_ids",
                "sealed_query_panel_ids", "support_panels",
                "observer_algorithm_digest", "search_space_digest",
                "hypothesis_policy_digest", "support_order",
                "query_pixels_observed", "raw_png_bytes_persisted",
                *_authority_data(), "record_digest",
            },
            "panel-program official support artifact",
        )
        if (
            raw["schema"] != PANEL_PROGRAM_OFFICIAL_SUPPORT_SCHEMA
            or raw["adapter_id"] != PANEL_PROGRAM_OFFICIAL_ADAPTER_ID
            or raw["support_order"] != "side-0-positive-six-then-side-1-contrast-six"
            or raw["query_pixels_observed"] is not False
            or raw["raw_png_bytes_persisted"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["support_panel_ids"], list)
            or not isinstance(raw["sealed_query_panel_ids"], list)
            or not isinstance(raw["support_panels"], list)
        ):
            raise PanelProgramOfficialTaskError("support artifact policy differs")
        result = cls(
            raw["adapter_algorithm_digest"], raw["task_id"],
            raw["task_plan_digest"], raw["prepared_batch_plan_digest"],
            raw["execution_precommit_digest"], raw["release_authorization_digest"],
            raw["exposure_successor_digest"], raw["plan_store_receipt_digest"],
            raw["precommit_store_receipt_digest"], raw["exposure_store_receipt_digest"],
            raw["authorization_store_receipt_digest"], tuple(raw["support_panel_ids"]),
            tuple(raw["sealed_query_panel_ids"]),
            tuple(PanelProgramOfficialSupportPanel.from_data(item) for item in raw["support_panels"]),
            raw["observer_algorithm_digest"], raw["search_space_digest"],
            raw["hypothesis_policy_digest"], raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelProgramOfficialTaskError("support artifact is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PanelProgramOfficialSupportRuntime:
    artifact: PanelProgramOfficialSupportArtifact
    released_panels: tuple[ReleasedOfficialPanel, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.artifact) is not PanelProgramOfficialSupportArtifact:
            raise TypeError("artifact must be exact PanelProgramOfficialSupportArtifact")
        if (
            type(self.released_panels) is not tuple
            or len(self.released_panels) != PANEL_PROGRAM_SUPPORT_COUNT
            or any(type(item) is not ReleasedOfficialPanel for item in self.released_panels)
        ):
            raise TypeError("released_panels must be an exact twelve-panel tuple")
        for binding, released in zip(
            self.artifact.support_panels, self.released_panels, strict=True
        ):
            cold = ReleasedOfficialPanel.from_data(released.to_data())
            if (
                cold != released
                or released.panel_id != binding.panel_id
                or released.record_digest != binding.released_panel_record_digest
                or released.release_receipt.record_digest != binding.release_receipt_digest
                or binding.released_panel_store_receipt.object_digest
                != released.record_digest
                or released.execution_precommit_digest
                != binding.execution_precommit_digest
                or released.exposure_successor_digest
                != binding.exposure_successor_digest
                or released.release_receipt.release_descriptor_digest
                != binding.release_descriptor_digest
                or released.release_receipt.archive_digest
                != binding.archive_digest
                or released.release_receipt.central_directory_digest
                != binding.archive_central_directory_digest
                or binding.execution_precommit_digest
                != self.artifact.execution_precommit_digest
                or binding.exposure_successor_digest
                != self.artifact.exposure_successor_digest
                or released.exact_png_digest != binding.exact_png_digest
                or len(released.exact_png_bytes) != binding.exact_png_byte_count
                or _bytes_address(released.exact_png_bytes) != binding.exact_png_digest
            ):
                raise PanelProgramOfficialTaskError("support runtime record differs")


def _expected_task_and_support(
    *,
    task: ObjectBongardTaskPlan,
    prepared: PreparedObjectBongardRelease,
    released_panels: tuple[ReleasedOfficialPanel, ...],
    released_panel_store_receipts: tuple[ObjectBongardWriteOnceReceipt, ...],
) -> tuple[
    ObjectBongardTaskPlan,
    tuple[str, ...],
    tuple[ReleasedOfficialPanel, ...],
    tuple[ObjectBongardWriteOnceReceipt, ...],
]:
    if type(task) is not ObjectBongardTaskPlan:
        raise TypeError("task must be exact ObjectBongardTaskPlan")
    if type(prepared) is not PreparedObjectBongardRelease:
        raise TypeError("prepared must be exact PreparedObjectBongardRelease")
    if type(released_panels) is not tuple or any(
        type(item) is not ReleasedOfficialPanel for item in released_panels
    ):
        raise TypeError("released_panels must be an exact typed tuple")
    if type(released_panel_store_receipts) is not tuple or any(
        type(item) is not ObjectBongardWriteOnceReceipt
        for item in released_panel_store_receipts
    ):
        raise TypeError("released_panel_store_receipts must be an exact typed tuple")
    frozen_task = ObjectBongardTaskPlan.from_data(task.to_data())
    prepared = verify_prepared_object_bongard_release(prepared)
    expected_ids = (
        *frozen_task.side_0_support_panel_ids,
        *frozen_task.side_1_support_panel_ids,
    )
    query_ids = (frozen_task.side_0_query_panel_id, frozen_task.side_1_query_panel_id)
    matches = tuple(item for item in prepared.plan.tasks if item.task_id == frozen_task.task_id)
    if (
        frozen_task.split != "train"
        or len(matches) != 1
        or matches[0] != frozen_task
        or len(released_panels) != PANEL_PROGRAM_SUPPORT_COUNT
        or len(released_panel_store_receipts) != PANEL_PROGRAM_SUPPORT_COUNT
        or tuple(item.panel_id for item in released_panels) != expected_ids
        or len({item.panel_id for item in released_panels}) != PANEL_PROGRAM_SUPPORT_COUNT
        or len({item.record_digest for item in released_panels}) != PANEL_PROGRAM_SUPPORT_COUNT
        or any(item.panel_id in query_ids for item in released_panels)
        or any(item.panel_id not in prepared.authorization.authorized_support_panel_ids for item in released_panels)
        or prepared.precommit.batch_plan_digest != prepared.plan.record_digest
        or prepared.authorization.batch_plan_digest != prepared.plan.record_digest
        or prepared.authorization.execution_precommit_digest != prepared.precommit.record_digest
        or prepared.authorization.exposure_successor_digest != prepared.successor.digest
    ):
        raise PanelProgramOfficialTaskError("task support release inventory/order differs")
    frozen_releases = tuple(
        ReleasedOfficialPanel.from_data(item.to_data()) for item in released_panels
    )
    frozen_store_receipts = tuple(
        ObjectBongardWriteOnceReceipt.from_data(item.to_data())
        for item in released_panel_store_receipts
    )
    for item, store_receipt in zip(
        frozen_releases, frozen_store_receipts, strict=True
    ):
        prepared.store.verify(store_receipt, expected_data=item.to_data())
        if (
            item.execution_precommit_digest != prepared.precommit.record_digest
            or item.exposure_successor_digest != prepared.successor.digest
            or item.release_receipt.release_descriptor_digest
            != prepared.precommit.release_descriptor_digest
            or item.release_receipt.archive_digest != prepared.precommit.archive_digest
            or item.release_receipt.central_directory_digest
            != prepared.precommit.archive_central_directory_digest
            or store_receipt.object_kind != "released-support-panel"
            or store_receipt.object_digest != item.record_digest
        ):
            raise PanelProgramOfficialTaskError("released support custody differs")
    return frozen_task, expected_ids, frozen_releases, frozen_store_receipts


def _make_support_panel(
    *, index: int, released: ReleasedOfficialPanel,
    released_panel_store_receipt: ObjectBongardWriteOnceReceipt,
    observation: PanelProgramObservation,
) -> PanelProgramOfficialSupportPanel:
    values = {
        "panel_alias": f"panel_{index:03d}",
        "support_role": "positive" if index < 6 else "contrast",
        "source_ordinal": index,
        "panel_id": released.panel_id,
        "released_panel_record_digest": released.record_digest,
        "release_receipt_digest": released.release_receipt.record_digest,
        "released_panel_store_receipt": released_panel_store_receipt,
        "released_panel_store_receipt_digest": released_panel_store_receipt.record_digest,
        "release_descriptor_digest": released.release_receipt.release_descriptor_digest,
        "archive_digest": released.release_receipt.archive_digest,
        "archive_central_directory_digest": released.release_receipt.central_directory_digest,
        "execution_precommit_digest": released.execution_precommit_digest,
        "exposure_successor_digest": released.exposure_successor_digest,
        "exact_png_digest": released.exact_png_digest,
        "exact_png_byte_count": len(released.exact_png_bytes),
        "observation": observation,
        "observation_digest": observation.observation_digest,
    }
    provisional = object.__new__(PanelProgramOfficialSupportPanel)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PanelProgramOfficialSupportPanel(
        **values,
        record_digest=_content_address(_support_panel_content(provisional)),
    )


def build_panel_program_official_support(
    *,
    task: ObjectBongardTaskPlan,
    prepared: PreparedObjectBongardRelease,
    released_panels: tuple[ReleasedOfficialPanel, ...],
    released_panel_store_receipts: tuple[ObjectBongardWriteOnceReceipt, ...],
    observe_program: ProgramObserver,
) -> PanelProgramOfficialSupportRuntime:
    """Observe twelve already released support records with a role-blind callback."""

    _require_exact_observer(observe_program)
    frozen_task, expected_ids, frozen_releases, frozen_store_receipts = (
        _expected_task_and_support(
            task=task,
            prepared=prepared,
            released_panels=released_panels,
            released_panel_store_receipts=released_panel_store_receipts,
        )
    )
    frozen_bindings = dict(prepared.precommit.runtime_source_bindings)
    required_bindings = panel_program_required_precommit_bindings()
    if any(frozen_bindings.get(key) != value for key, value in required_bindings.items()):
        raise PanelProgramOfficialTaskError(
            "observer source/algorithm/catalog/policy were not frozen in the execution precommit"
        )
    observations: list[PanelProgramObservation] = []
    observation_cache: dict[str, PanelProgramObservation] = {}
    for released in frozen_releases:
        observed = observation_cache.get(released.exact_png_digest)
        if observed is None:
            observed = observe_program(released.exact_png_bytes)
            observation_cache[released.exact_png_digest] = observed
        if type(observed) is not PanelProgramObservation:
            raise TypeError("observer must return exact PanelProgramObservation")
        restored = PanelProgramObservation.from_data(observed.to_data())
        if restored != observed or _observation_panel_digest(restored) != released.exact_png_digest:
            raise PanelProgramOfficialTaskError("observer PNG binding differs")
        observations.append(restored)
    common = {
        (
            item.observer_algorithm_digest,
            item.search_space_digest,
            item.hypothesis_policy_digest,
        )
        for item in observations
    }
    if len(common) != 1:
        raise PanelProgramOfficialTaskError("support observer policy is not common")
    panels = tuple(
        _make_support_panel(
            index=index,
            released=released,
            released_panel_store_receipt=store_receipt,
            observation=observation,
        )
        for index, (released, store_receipt, observation) in enumerate(
            zip(frozen_releases, frozen_store_receipts, observations, strict=True)
        )
    )
    observer_algorithm_digest, search_space_digest, hypothesis_policy_digest = next(iter(common))
    if (
        observer_algorithm_digest
        != required_bindings["panel_program_observer_algorithm"]
        or search_space_digest != required_bindings["panel_program_search_space"]
        or hypothesis_policy_digest
        != required_bindings["panel_program_hypothesis_policy"]
    ):
        raise PanelProgramOfficialTaskError(
            "observed support policy differs from the precommitted observer"
        )
    values = {
        "adapter_algorithm_digest": panel_program_official_task_algorithm_digest(),
        "task_id": frozen_task.task_id,
        "task_plan_digest": frozen_task.record_digest,
        "prepared_batch_plan_digest": prepared.plan.record_digest,
        "execution_precommit_digest": prepared.precommit.record_digest,
        "release_authorization_digest": prepared.authorization.record_digest,
        "exposure_successor_digest": prepared.successor.digest,
        "plan_store_receipt_digest": prepared.plan_receipt.record_digest,
        "precommit_store_receipt_digest": prepared.precommit_receipt.record_digest,
        "exposure_store_receipt_digest": prepared.exposure_receipt.record_digest,
        "authorization_store_receipt_digest": prepared.authorization_receipt.record_digest,
        "support_panel_ids": expected_ids,
        "sealed_query_panel_ids": (
            frozen_task.side_0_query_panel_id,
            frozen_task.side_1_query_panel_id,
        ),
        "support_panels": panels,
        "observer_algorithm_digest": observer_algorithm_digest,
        "search_space_digest": search_space_digest,
        "hypothesis_policy_digest": hypothesis_policy_digest,
    }
    provisional = object.__new__(PanelProgramOfficialSupportArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    artifact = PanelProgramOfficialSupportArtifact(
        **values, record_digest=_content_address(_support_content(provisional))
    )
    return PanelProgramOfficialSupportRuntime(artifact, frozen_releases)


def _selection_content(
    version_space: ProgramVersionSpace, selected_rule: FrozenProgramRule
) -> dict[str, object]:
    return {
        "schema": "gkm.panel-program-deterministic-selection.v1",
        "version_space_digest": version_space.version_space_digest,
        "selected_rule_digest": selected_rule.rule_digest,
        "selection_rule": "minimum-atom-count-then-formula-digest",
        "ranker_used": False,
        "model_used": False,
    }


def _freeze_content(value: "PanelProgramOfficialTaskFreeze") -> dict[str, object]:
    return {
        "schema": PANEL_PROGRAM_OFFICIAL_TASK_FREEZE_SCHEMA,
        "adapter_id": PANEL_PROGRAM_OFFICIAL_ADAPTER_ID,
        "algorithm_digest": value.algorithm_digest,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "prepared_batch_plan_digest": value.prepared_batch_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "release_authorization_digest": value.release_authorization_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "support_artifact": value.support_artifact.to_data(),
        "support_artifact_digest": value.support_artifact_digest,
        "support_panel_ids": list(value.support_panel_ids),
        "support_observation_digests": list(value.support_observation_digests),
        "semantic_version_space": value.semantic_version_space.to_data(),
        "semantic_version_space_digest": value.semantic_version_space_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "selection_record": _selection_content(
            value.semantic_version_space, value.selected_rule
        ),
        "selection_record_digest": value.selection_record_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_rule": value.selected_rule.to_data(),
        "selected_rule_digest": value.selected_rule_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "support_count": PANEL_PROGRAM_SUPPORT_COUNT,
        "query_pixels_observed": False,
        "selection_completed_before_query_release": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelProgramOfficialTaskFreeze:
    algorithm_digest: str
    task_id: str
    task_plan_digest: str
    prepared_batch_plan_digest: str
    execution_precommit_digest: str
    release_authorization_digest: str
    exposure_successor_digest: str
    support_artifact: PanelProgramOfficialSupportArtifact
    support_artifact_digest: str
    support_panel_ids: tuple[str, ...]
    support_observation_digests: tuple[str, ...]
    semantic_version_space: ProgramVersionSpace
    semantic_version_space_digest: str
    version_space_digest: str
    support_version_space_digest: str
    selection_record_digest: str
    rank_response_digest: str
    selected_rule: FrozenProgramRule
    selected_rule_digest: str
    selected_predicate_digest: str
    sealed_query_panel_ids: tuple[str, str]
    record_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("algorithm digest", self.algorithm_digest),
            ("task plan digest", self.task_plan_digest),
            ("prepared batch plan digest", self.prepared_batch_plan_digest),
            ("execution precommit digest", self.execution_precommit_digest),
            ("release authorization digest", self.release_authorization_digest),
            ("exposure successor digest", self.exposure_successor_digest),
            ("support artifact digest", self.support_artifact_digest),
            ("semantic version-space digest", self.semantic_version_space_digest),
            ("selection record digest", self.selection_record_digest),
            ("selected rule digest", self.selected_rule_digest),
            ("task freeze digest", self.record_digest),
        ):
            _require_address(item, label)
        for label, item in (
            ("version-space digest", self.version_space_digest),
            ("support version-space digest", self.support_version_space_digest),
            ("rank response digest", self.rank_response_digest),
            ("selected predicate digest", self.selected_predicate_digest),
        ):
            _require_raw_digest(item, label)
        if type(self.support_artifact) is not PanelProgramOfficialSupportArtifact:
            raise TypeError("support_artifact has the wrong type")
        if type(self.semantic_version_space) is not ProgramVersionSpace:
            raise TypeError("semantic_version_space has the wrong type")
        if type(self.selected_rule) is not FrozenProgramRule:
            raise TypeError("selected_rule has the wrong type")
        restored_space = ProgramVersionSpace.from_data(
            self.semantic_version_space.to_data()
        )
        restored_rule = FrozenProgramRule.from_data(self.selected_rule.to_data())
        expected_space = build_program_version_space(
            self.support_artifact.positive_observations,
            self.support_artifact.contrast_observations,
        )
        expected_rule = freeze_program_rule(expected_space)
        expected_selection_digest = _content_address(
            _selection_content(expected_space, expected_rule)
        )
        expected_observation_digests = tuple(
            item.observation_digest for item in self.support_artifact.support_panels
        )
        if (
            type(self.task_id) is not str
            or not self.task_id
            or self.support_artifact.task_id != self.task_id
            or self.support_artifact.task_plan_digest != self.task_plan_digest
            or self.support_artifact.prepared_batch_plan_digest
            != self.prepared_batch_plan_digest
            or self.support_artifact.execution_precommit_digest
            != self.execution_precommit_digest
            or self.support_artifact.release_authorization_digest
            != self.release_authorization_digest
            or self.support_artifact.exposure_successor_digest
            != self.exposure_successor_digest
            or self.support_artifact.record_digest != self.support_artifact_digest
            or self.support_panel_ids != self.support_artifact.support_panel_ids
            or self.support_observation_digests != expected_observation_digests
            or self.sealed_query_panel_ids
            != self.support_artifact.sealed_query_panel_ids
            or restored_space != self.semantic_version_space
            or restored_rule != self.selected_rule
            or self.semantic_version_space != expected_space
            or self.selected_rule != expected_rule
            or self.semantic_version_space_digest
            != self.semantic_version_space.version_space_digest
            or self.version_space_digest
            != self.semantic_version_space_digest.removeprefix("sha256:")
            or self.support_version_space_digest != self.version_space_digest
            or self.selected_rule_digest != self.selected_rule.rule_digest
            or self.selected_predicate_digest
            != self.selected_rule_digest.removeprefix("sha256:")
            or self.selection_record_digest != expected_selection_digest
            or self.rank_response_digest
            != expected_selection_digest.removeprefix("sha256:")
            or self.record_digest != _content_address(_freeze_content(self))
            or _has_bytes(_freeze_content(self))
        ):
            raise PanelProgramOfficialTaskError("task freeze semantic replay differs")

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelProgramOfficialTaskFreeze":
        raw = _fields(
            value,
            {
                "schema", "adapter_id", "algorithm_digest", "task_id",
                "task_plan_digest", "prepared_batch_plan_digest",
                "execution_precommit_digest", "release_authorization_digest",
                "exposure_successor_digest", "support_artifact",
                "support_artifact_digest", "support_panel_ids",
                "support_observation_digests", "semantic_version_space",
                "semantic_version_space_digest", "version_space_digest",
                "support_version_space_digest", "selection_record",
                "selection_record_digest", "rank_response_digest",
                "selected_rule", "selected_rule_digest",
                "selected_predicate_digest", "sealed_query_panel_ids",
                "support_count", "query_pixels_observed",
                "selection_completed_before_query_release",
                *_authority_data(), "record_digest",
            },
            "panel-program official task freeze",
        )
        if (
            raw["schema"] != PANEL_PROGRAM_OFFICIAL_TASK_FREEZE_SCHEMA
            or raw["adapter_id"] != PANEL_PROGRAM_OFFICIAL_ADAPTER_ID
            or raw["support_count"] != PANEL_PROGRAM_SUPPORT_COUNT
            or raw["query_pixels_observed"] is not False
            or raw["selection_completed_before_query_release"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["support_artifact"], Mapping)
            or not isinstance(raw["semantic_version_space"], Mapping)
            or not isinstance(raw["selected_rule"], Mapping)
            or not isinstance(raw["selection_record"], Mapping)
            or not isinstance(raw["support_panel_ids"], list)
            or not isinstance(raw["support_observation_digests"], list)
            or not isinstance(raw["sealed_query_panel_ids"], list)
        ):
            raise PanelProgramOfficialTaskError("task freeze policy differs")
        support = PanelProgramOfficialSupportArtifact.from_data(raw["support_artifact"])
        space = ProgramVersionSpace.from_data(raw["semantic_version_space"])
        rule = FrozenProgramRule.from_data(raw["selected_rule"])
        if dict(raw["selection_record"]) != _selection_content(space, rule):
            raise PanelProgramOfficialTaskError("selection record differs")
        result = cls(
            raw["algorithm_digest"], raw["task_id"], raw["task_plan_digest"],
            raw["prepared_batch_plan_digest"], raw["execution_precommit_digest"],
            raw["release_authorization_digest"], raw["exposure_successor_digest"],
            support, raw["support_artifact_digest"], tuple(raw["support_panel_ids"]),
            tuple(raw["support_observation_digests"]), space,
            raw["semantic_version_space_digest"], raw["version_space_digest"],
            raw["support_version_space_digest"], raw["selection_record_digest"],
            raw["rank_response_digest"], rule, raw["selected_rule_digest"],
            raw["selected_predicate_digest"], tuple(raw["sealed_query_panel_ids"]),
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelProgramOfficialTaskError("task freeze is not canonical")
        return result


def _freeze_support_artifact(
    support: PanelProgramOfficialSupportArtifact,
) -> PanelProgramOfficialTaskFreeze:
    frozen_support = PanelProgramOfficialSupportArtifact.from_data(support.to_data())
    version_space = build_program_version_space(
        frozen_support.positive_observations,
        frozen_support.contrast_observations,
    )
    selected_rule = freeze_program_rule(version_space)
    selection_digest = _content_address(
        _selection_content(version_space, selected_rule)
    )
    values = {
        "algorithm_digest": panel_program_official_task_algorithm_digest(),
        "task_id": frozen_support.task_id,
        "task_plan_digest": frozen_support.task_plan_digest,
        "prepared_batch_plan_digest": frozen_support.prepared_batch_plan_digest,
        "execution_precommit_digest": frozen_support.execution_precommit_digest,
        "release_authorization_digest": frozen_support.release_authorization_digest,
        "exposure_successor_digest": frozen_support.exposure_successor_digest,
        "support_artifact": frozen_support,
        "support_artifact_digest": frozen_support.record_digest,
        "support_panel_ids": frozen_support.support_panel_ids,
        "support_observation_digests": tuple(
            item.observation_digest for item in frozen_support.support_panels
        ),
        "semantic_version_space": version_space,
        "semantic_version_space_digest": version_space.version_space_digest,
        "version_space_digest": version_space.version_space_digest.removeprefix("sha256:"),
        "support_version_space_digest": version_space.version_space_digest.removeprefix("sha256:"),
        "selection_record_digest": selection_digest,
        "rank_response_digest": selection_digest.removeprefix("sha256:"),
        "selected_rule": selected_rule,
        "selected_rule_digest": selected_rule.rule_digest,
        "selected_predicate_digest": selected_rule.rule_digest.removeprefix("sha256:"),
        "sealed_query_panel_ids": frozen_support.sealed_query_panel_ids,
    }
    provisional = object.__new__(PanelProgramOfficialTaskFreeze)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PanelProgramOfficialTaskFreeze(
        **values, record_digest=_content_address(_freeze_content(provisional))
    )


def freeze_panel_program_official_task_decision(
    *, support: PanelProgramOfficialSupportRuntime
) -> PanelProgramOfficialTaskFreeze:
    """Build and deterministically freeze the exact released 6+6 runtime."""

    if type(support) is not PanelProgramOfficialSupportRuntime:
        raise TypeError("support must be exact PanelProgramOfficialSupportRuntime")
    PanelProgramOfficialSupportRuntime.__post_init__(support)
    return _freeze_support_artifact(support.artifact)


def cold_verify_panel_program_official_task_freeze(
    freeze: PanelProgramOfficialTaskFreeze,
    *, expected_freeze_digest: str,
) -> PanelProgramOfficialTaskFreeze:
    if type(freeze) is not PanelProgramOfficialTaskFreeze:
        raise TypeError("freeze has the wrong type")
    expected_digest = _require_address(expected_freeze_digest, "expected freeze digest")
    restored = PanelProgramOfficialTaskFreeze.from_data(freeze.to_data())
    expected = _freeze_support_artifact(restored.support_artifact)
    if restored.record_digest != expected_digest or restored != expected:
        raise PanelProgramOfficialTaskError("task freeze differs from cold replay")
    return restored


def _commit_content(value: "PanelProgramOfficialTaskCommit") -> dict[str, object]:
    return {
        "schema": PANEL_PROGRAM_OFFICIAL_TASK_COMMIT_SCHEMA,
        "adapter_id": PANEL_PROGRAM_OFFICIAL_ADAPTER_ID,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "version_space_digest": value.version_space_digest,
        "support_version_space_digest": value.support_version_space_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_predicate_digest": value.selected_predicate_digest,
        "task_freeze_digest": value.task_freeze_digest,
        "exact_freeze_payload_digest": value.exact_freeze_payload_digest,
        "exact_freeze_payload_size": value.exact_freeze_payload_size,
        "task_freeze_store_receipt": value.task_freeze_store_receipt.to_data(),
        "task_freeze_store_receipt_digest": value.task_freeze_store_receipt_digest,
        "durably_persisted_and_reloaded_before_query_release": True,
        "exact_canonical_freeze_payload_bound": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelProgramOfficialTaskCommit:
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    version_space_digest: str
    support_version_space_digest: str
    rank_response_digest: str
    selected_predicate_digest: str
    task_freeze_digest: str
    exact_freeze_payload_digest: str
    exact_freeze_payload_size: int
    task_freeze_store_receipt: ObjectBongardWriteOnceReceipt
    task_freeze_store_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id:
            raise PanelProgramOfficialTaskError("commit task ID differs")
        for label, item in (
            ("task plan digest", self.task_plan_digest),
            ("execution precommit digest", self.execution_precommit_digest),
            ("task freeze digest", self.task_freeze_digest),
            ("exact freeze payload digest", self.exact_freeze_payload_digest),
            ("freeze receipt digest", self.task_freeze_store_receipt_digest),
            ("task commit digest", self.record_digest),
        ):
            _require_address(item, label)
        for label, item in (
            ("version-space digest", self.version_space_digest),
            ("support version-space digest", self.support_version_space_digest),
            ("rank response digest", self.rank_response_digest),
            ("selected predicate digest", self.selected_predicate_digest),
        ):
            _require_raw_digest(item, label)
        if type(self.task_freeze_store_receipt) is not ObjectBongardWriteOnceReceipt:
            raise TypeError("task_freeze_store_receipt has the wrong type")
        receipt = ObjectBongardWriteOnceReceipt.from_data(
            self.task_freeze_store_receipt.to_data()
        )
        if (
            type(self.exact_freeze_payload_size) is not int
            or self.exact_freeze_payload_size <= 0
            or self.version_space_digest != self.support_version_space_digest
            or receipt.object_kind != "task-freeze"
            or receipt.object_digest != self.task_freeze_digest
            or receipt.payload_digest != self.exact_freeze_payload_digest
            or receipt.size_bytes != self.exact_freeze_payload_size
            or receipt.record_digest != self.task_freeze_store_receipt_digest
            or self.record_digest != _content_address(_commit_content(self))
        ):
            raise PanelProgramOfficialTaskError("task decision durable commit differs")

    def to_data(self) -> dict[str, object]:
        return {**_commit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelProgramOfficialTaskCommit":
        raw = _fields(
            value,
            {
                "schema", "adapter_id", "task_id", "task_plan_digest",
                "execution_precommit_digest", "version_space_digest",
                "support_version_space_digest", "rank_response_digest",
                "selected_predicate_digest", "task_freeze_digest",
                "exact_freeze_payload_digest", "exact_freeze_payload_size",
                "task_freeze_store_receipt", "task_freeze_store_receipt_digest",
                "durably_persisted_and_reloaded_before_query_release",
                "exact_canonical_freeze_payload_bound", *_authority_data(),
                "record_digest",
            },
            "panel-program official task commit",
        )
        if (
            raw["schema"] != PANEL_PROGRAM_OFFICIAL_TASK_COMMIT_SCHEMA
            or raw["adapter_id"] != PANEL_PROGRAM_OFFICIAL_ADAPTER_ID
            or raw["durably_persisted_and_reloaded_before_query_release"] is not True
            or raw["exact_canonical_freeze_payload_bound"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["task_freeze_store_receipt"], Mapping)
        ):
            raise PanelProgramOfficialTaskError("task commit policy differs")
        result = cls(
            raw["task_id"], raw["task_plan_digest"],
            raw["execution_precommit_digest"], raw["version_space_digest"],
            raw["support_version_space_digest"], raw["rank_response_digest"],
            raw["selected_predicate_digest"], raw["task_freeze_digest"],
            raw["exact_freeze_payload_digest"], raw["exact_freeze_payload_size"],
            ObjectBongardWriteOnceReceipt.from_data(raw["task_freeze_store_receipt"]),
            raw["task_freeze_store_receipt_digest"], raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelProgramOfficialTaskError("task commit is not canonical")
        return result

    def assert_matches(
        self,
        freeze: PanelProgramOfficialTaskFreeze,
        exact_freeze_payload: bytes,
        freeze_receipt: ObjectBongardWriteOnceReceipt,
    ) -> None:
        if self != commit_panel_program_official_task_decision(
            freeze=freeze,
            exact_freeze_payload=exact_freeze_payload,
            task_freeze_store_receipt=freeze_receipt,
        ):
            raise PanelProgramOfficialTaskError(
                "task decision commit differs from cold replay"
            )


def commit_panel_program_official_task_decision(
    *,
    freeze: PanelProgramOfficialTaskFreeze,
    exact_freeze_payload: bytes,
    task_freeze_store_receipt: ObjectBongardWriteOnceReceipt,
) -> PanelProgramOfficialTaskCommit:
    if type(freeze) is not PanelProgramOfficialTaskFreeze:
        raise TypeError("freeze has the wrong type")
    frozen = cold_verify_panel_program_official_task_freeze(
        freeze, expected_freeze_digest=freeze.record_digest
    )
    if type(exact_freeze_payload) is not bytes:
        raise TypeError("exact_freeze_payload must be exact bytes")
    expected_payload = canonical_json(frozen.to_data()) + b"\n"
    if exact_freeze_payload != expected_payload:
        raise PanelProgramOfficialTaskError("freeze payload is not exact canonical JSON")
    if type(task_freeze_store_receipt) is not ObjectBongardWriteOnceReceipt:
        raise TypeError("task_freeze_store_receipt has the wrong type")
    receipt = ObjectBongardWriteOnceReceipt.from_data(
        task_freeze_store_receipt.to_data()
    )
    payload_digest = _bytes_address(expected_payload)
    if (
        receipt.object_kind != "task-freeze"
        or receipt.object_digest != frozen.record_digest
        or receipt.payload_digest != payload_digest
        or receipt.size_bytes != len(expected_payload)
    ):
        raise PanelProgramOfficialTaskError("freeze receipt does not bind exact payload")
    values = {
        "task_id": frozen.task_id,
        "task_plan_digest": frozen.task_plan_digest,
        "execution_precommit_digest": frozen.execution_precommit_digest,
        "version_space_digest": frozen.version_space_digest,
        "support_version_space_digest": frozen.support_version_space_digest,
        "rank_response_digest": frozen.rank_response_digest,
        "selected_predicate_digest": frozen.selected_predicate_digest,
        "task_freeze_digest": frozen.record_digest,
        "exact_freeze_payload_digest": payload_digest,
        "exact_freeze_payload_size": len(expected_payload),
        "task_freeze_store_receipt": receipt,
        "task_freeze_store_receipt_digest": receipt.record_digest,
    }
    provisional = object.__new__(PanelProgramOfficialTaskCommit)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PanelProgramOfficialTaskCommit(
        **values, record_digest=_content_address(_commit_content(provisional))
    )


def cold_verify_panel_program_official_task_commit(
    commit: PanelProgramOfficialTaskCommit,
    *, freeze: PanelProgramOfficialTaskFreeze, exact_freeze_payload: bytes,
    task_freeze_store_receipt: ObjectBongardWriteOnceReceipt,
    expected_commit_digest: str,
) -> PanelProgramOfficialTaskCommit:
    if type(commit) is not PanelProgramOfficialTaskCommit:
        raise TypeError("commit has the wrong type")
    expected_digest = _require_address(expected_commit_digest, "expected commit digest")
    restored = PanelProgramOfficialTaskCommit.from_data(commit.to_data())
    expected = commit_panel_program_official_task_decision(
        freeze=freeze, exact_freeze_payload=exact_freeze_payload,
        task_freeze_store_receipt=task_freeze_store_receipt,
    )
    if restored.record_digest != expected_digest or restored != expected:
        raise PanelProgramOfficialTaskError("task commit differs from cold replay")
    return restored


@dataclass(frozen=True, slots=True)
class PanelProgramOfficialDurableDecision:
    support_runtime: PanelProgramOfficialSupportRuntime = field(repr=False)
    freeze: PanelProgramOfficialTaskFreeze
    freeze_receipt: ObjectBongardWriteOnceReceipt
    commit: PanelProgramOfficialTaskCommit
    commit_receipt: ObjectBongardWriteOnceReceipt

    def __post_init__(self) -> None:
        if type(self.support_runtime) is not PanelProgramOfficialSupportRuntime:
            raise TypeError("support_runtime has the wrong type")
        PanelProgramOfficialSupportRuntime.__post_init__(self.support_runtime)
        if type(self.freeze) is not PanelProgramOfficialTaskFreeze:
            raise TypeError("freeze has the wrong type")
        if type(self.commit) is not PanelProgramOfficialTaskCommit:
            raise TypeError("commit has the wrong type")
        if type(self.freeze_receipt) is not ObjectBongardWriteOnceReceipt:
            raise TypeError("freeze_receipt has the wrong type")
        if type(self.commit_receipt) is not ObjectBongardWriteOnceReceipt:
            raise TypeError("commit_receipt has the wrong type")
        freeze_receipt = ObjectBongardWriteOnceReceipt.from_data(
            self.freeze_receipt.to_data()
        )
        commit_receipt = ObjectBongardWriteOnceReceipt.from_data(
            self.commit_receipt.to_data()
        )
        exact_payload = canonical_json(self.freeze.to_data()) + b"\n"
        cold_verify_panel_program_official_task_commit(
            self.commit, freeze=self.freeze, exact_freeze_payload=exact_payload,
            task_freeze_store_receipt=freeze_receipt,
            expected_commit_digest=self.commit.record_digest,
        )
        if (
            self.support_runtime.artifact != self.freeze.support_artifact
            or
            freeze_receipt.object_kind != "task-freeze"
            or freeze_receipt.object_digest != self.freeze.record_digest
            or commit_receipt.object_kind != "task-decision-commit"
            or commit_receipt.object_digest != self.commit.record_digest
        ):
            raise PanelProgramOfficialTaskError("durable decision receipt differs")


def _verify_support_runtime_store(
    *,
    prepared: PreparedObjectBongardRelease,
    support_runtime: PanelProgramOfficialSupportRuntime,
) -> PanelProgramOfficialSupportRuntime:
    if type(support_runtime) is not PanelProgramOfficialSupportRuntime:
        raise TypeError("support_runtime has the wrong type")
    PanelProgramOfficialSupportRuntime.__post_init__(support_runtime)
    support = support_runtime.artifact
    if (
        support.prepared_batch_plan_digest != prepared.plan.record_digest
        or support.execution_precommit_digest != prepared.precommit.record_digest
        or support.release_authorization_digest != prepared.authorization.record_digest
        or support.exposure_successor_digest != prepared.successor.digest
        or support.plan_store_receipt_digest != prepared.plan_receipt.record_digest
        or support.precommit_store_receipt_digest != prepared.precommit_receipt.record_digest
        or support.exposure_store_receipt_digest != prepared.exposure_receipt.record_digest
        or support.authorization_store_receipt_digest
        != prepared.authorization_receipt.record_digest
    ):
        raise PanelProgramOfficialTaskError("support runtime differs from prepared release")
    for binding, released in zip(
        support.support_panels, support_runtime.released_panels, strict=True
    ):
        replay = prepared.store.verify(
            binding.released_panel_store_receipt,
            expected_data=released.to_data(),
        )
        if (
            ReleasedOfficialPanel.from_data(replay) != released
            or binding.released_panel_store_receipt.object_kind
            != "released-support-panel"
            or binding.released_panel_store_receipt.object_digest
            != released.record_digest
            or released.execution_precommit_digest
            != prepared.precommit.record_digest
            or released.exposure_successor_digest != prepared.successor.digest
            or released.release_receipt.release_descriptor_digest
            != prepared.precommit.release_descriptor_digest
            or released.release_receipt.archive_digest
            != prepared.precommit.archive_digest
            or released.release_receipt.central_directory_digest
            != prepared.precommit.archive_central_directory_digest
            or binding.execution_precommit_digest
            != prepared.precommit.record_digest
            or binding.exposure_successor_digest != prepared.successor.digest
        ):
            raise PanelProgramOfficialTaskError(
                "released support panel durable replay differs"
            )
    return support_runtime


def verify_panel_program_query_release_authority(
    *,
    prepared: PreparedObjectBongardRelease,
    freeze: PanelProgramOfficialTaskFreeze,
) -> None:
    """Gate-owned cold proof that every support release is real and current.

    This is called from the generic query-release boundary.  It deliberately
    needs no in-memory support runtime: each exact released-panel payload is
    reloaded from the receipt embedded in the typed freeze and checked against
    the current prepared authority before query bytes can be opened.
    """

    if type(prepared) is not PreparedObjectBongardRelease:
        raise TypeError("prepared has the wrong type")
    if type(freeze) is not PanelProgramOfficialTaskFreeze:
        raise TypeError("freeze has the wrong type")
    prepared = verify_prepared_object_bongard_release(prepared)
    frozen = cold_verify_panel_program_official_task_freeze(
        freeze, expected_freeze_digest=freeze.record_digest
    )
    support = frozen.support_artifact
    matching_tasks = tuple(
        item for item in prepared.plan.tasks if item.task_id == frozen.task_id
    )
    if len(matching_tasks) != 1:
        raise PanelProgramOfficialTaskError(
            "typed support task is absent or repeated in the prepared plan"
        )
    task = ObjectBongardTaskPlan.from_data(matching_tasks[0].to_data())
    expected_support_ids = (
        *task.side_0_support_panel_ids,
        *task.side_1_support_panel_ids,
    )
    expected_query_ids = (
        task.side_0_query_panel_id,
        task.side_1_query_panel_id,
    )
    required_bindings = panel_program_required_precommit_bindings()
    frozen_bindings = dict(prepared.precommit.runtime_source_bindings)
    if (
        task.split != "train"
        or support.adapter_algorithm_digest
        != panel_program_official_task_algorithm_digest()
        or frozen.task_plan_digest != task.record_digest
        or support.task_id != task.task_id
        or support.task_plan_digest != task.record_digest
        or support.prepared_batch_plan_digest != prepared.plan.record_digest
        or support.execution_precommit_digest != prepared.precommit.record_digest
        or support.release_authorization_digest != prepared.authorization.record_digest
        or support.exposure_successor_digest != prepared.successor.digest
        or support.plan_store_receipt_digest != prepared.plan_receipt.record_digest
        or support.precommit_store_receipt_digest
        != prepared.precommit_receipt.record_digest
        or support.exposure_store_receipt_digest
        != prepared.exposure_receipt.record_digest
        or support.authorization_store_receipt_digest
        != prepared.authorization_receipt.record_digest
        or support.support_panel_ids != expected_support_ids
        or support.sealed_query_panel_ids != expected_query_ids
        or frozen.support_panel_ids != expected_support_ids
        or frozen.sealed_query_panel_ids != expected_query_ids
        or tuple(item.panel_id for item in support.support_panels)
        != expected_support_ids
        or any(
            panel_id not in prepared.authorization.authorized_support_panel_ids
            or panel_id not in prepared.precommit.authorized_support_panel_ids
            for panel_id in expected_support_ids
        )
        or any(
            panel_id not in prepared.authorization.sealed_query_panel_ids
            or panel_id not in prepared.precommit.sealed_query_panel_ids
            for panel_id in expected_query_ids
        )
        or any(
            frozen_bindings.get(key) != value
            for key, value in required_bindings.items()
        )
        or support.observer_algorithm_digest
        != required_bindings["panel_program_observer_algorithm"]
        or support.search_space_digest != required_bindings["panel_program_search_space"]
        or support.hypothesis_policy_digest
        != required_bindings["panel_program_hypothesis_policy"]
    ):
        raise PanelProgramOfficialTaskError(
            "typed support freeze differs from the current prepared release"
        )
    replay_cache: dict[str, PanelProgramObservation] = {}
    for binding in support.support_panels:
        receipt = binding.released_panel_store_receipt
        try:
            raw = prepared.store.verify(receipt)
            released = ReleasedOfficialPanel.from_data(raw)
        except Exception as exc:
            raise PanelProgramOfficialTaskError(
                "cannot cold-read a typed support release"
            ) from exc
        if (
            receipt.object_kind != "released-support-panel"
            or receipt.object_digest != binding.released_panel_record_digest
            or receipt.record_digest != binding.released_panel_store_receipt_digest
            or released.panel_id != binding.panel_id
            or released.record_digest != binding.released_panel_record_digest
            or released.release_receipt.record_digest != binding.release_receipt_digest
            or released.exact_png_digest != binding.exact_png_digest
            or len(released.exact_png_bytes) != binding.exact_png_byte_count
            or released.release_receipt.size_bytes != binding.exact_png_byte_count
            or released.release_receipt.release_descriptor_digest
            != binding.release_descriptor_digest
            or released.release_receipt.archive_digest != binding.archive_digest
            or released.release_receipt.central_directory_digest
            != binding.archive_central_directory_digest
            or binding.release_descriptor_digest
            != prepared.precommit.release_descriptor_digest
            or binding.archive_digest != prepared.precommit.archive_digest
            or binding.archive_central_directory_digest
            != prepared.precommit.archive_central_directory_digest
            or released.execution_precommit_digest != prepared.precommit.record_digest
            or released.exposure_successor_digest != prepared.successor.digest
            or binding.execution_precommit_digest != prepared.precommit.record_digest
            or binding.exposure_successor_digest != prepared.successor.digest
            or released.release_receipt.release_descriptor_digest
            != prepared.precommit.release_descriptor_digest
            or released.release_receipt.archive_digest
            != prepared.precommit.archive_digest
            or released.release_receipt.central_directory_digest
            != prepared.precommit.archive_central_directory_digest
            or released.exact_png_digest
            != _bytes_address(released.exact_png_bytes)
        ):
            raise PanelProgramOfficialTaskError(
                "typed support freeze lacks a current durable panel release"
            )
        cold_observation = replay_cache.get(released.exact_png_digest)
        if cold_observation is None:
            cold_observation = observe_authenticated_program_png(
                released.exact_png_bytes
            )
            replay_cache[released.exact_png_digest] = cold_observation
        if cold_observation != binding.observation:
            raise PanelProgramOfficialTaskError(
                "typed support observation differs from cold pixel replay"
            )


def persist_panel_program_official_task_decision(
    *, prepared: PreparedObjectBongardRelease,
    support_runtime: PanelProgramOfficialSupportRuntime,
    freeze: PanelProgramOfficialTaskFreeze,
) -> PanelProgramOfficialDurableDecision:
    """Persist and reload the exact freeze and its exact binding commit."""

    if type(prepared) is not PreparedObjectBongardRelease:
        raise TypeError("prepared has the wrong type")
    prepared = verify_prepared_object_bongard_release(prepared)
    runtime = _verify_support_runtime_store(
        prepared=prepared, support_runtime=support_runtime
    )
    frozen = cold_verify_panel_program_official_task_freeze(
        freeze, expected_freeze_digest=freeze.record_digest
    )
    if (
        frozen.support_artifact != runtime.artifact
        or frozen.prepared_batch_plan_digest != prepared.plan.record_digest
        or frozen.execution_precommit_digest != prepared.precommit.record_digest
        or frozen.release_authorization_digest != prepared.authorization.record_digest
        or frozen.exposure_successor_digest != prepared.successor.digest
    ):
        raise PanelProgramOfficialTaskError("freeze differs from prepared release")
    freeze_receipt = persist_object_bongard_task_freeze(
        store=prepared.store, freeze=frozen
    )
    freeze_data = prepared.store.verify(
        freeze_receipt, expected_data=frozen.to_data()
    )
    reloaded_freeze = PanelProgramOfficialTaskFreeze.from_data(freeze_data)
    if reloaded_freeze != frozen:
        raise PanelProgramOfficialTaskError("durable freeze replay differs")
    exact_payload = canonical_json(freeze_data) + b"\n"
    commit = commit_panel_program_official_task_decision(
        freeze=reloaded_freeze, exact_freeze_payload=exact_payload,
        task_freeze_store_receipt=freeze_receipt,
    )
    commit_receipt = persist_object_bongard_task_commit(
        store=prepared.store, commit=commit
    )
    commit_data = prepared.store.verify(
        commit_receipt, expected_data=commit.to_data()
    )
    reloaded_commit = PanelProgramOfficialTaskCommit.from_data(commit_data)
    if reloaded_commit != commit:
        raise PanelProgramOfficialTaskError("durable commit replay differs")
    return PanelProgramOfficialDurableDecision(
        runtime, reloaded_freeze, freeze_receipt, reloaded_commit, commit_receipt
    )


def _query_content(value: "PanelProgramOfficialQueryResult") -> dict[str, object]:
    return {
        "schema": PANEL_PROGRAM_OFFICIAL_QUERY_SCHEMA,
        "adapter_id": PANEL_PROGRAM_OFFICIAL_ADAPTER_ID,
        "algorithm_digest": value.algorithm_digest,
        "task_id": value.task_id,
        "task_plan_digest": value.task_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "task_freeze_digest": value.task_freeze_digest,
        "task_commit_digest": value.task_commit_digest,
        "task_freeze_store_receipt_digest": value.task_freeze_store_receipt_digest,
        "task_commit_store_receipt_digest": value.task_commit_store_receipt_digest,
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "query_panel_id": value.query_panel_id,
        "released_query_panel_record_digest": value.released_query_panel_record_digest,
        "released_query_store_receipt_digest": value.released_query_store_receipt_digest,
        "release_receipt_digest": value.release_receipt_digest,
        "exact_png_digest": value.exact_png_digest,
        "exact_png_byte_count": value.exact_png_byte_count,
        "observer_algorithm_digest": value.observer_algorithm_digest,
        "search_space_digest": value.search_space_digest,
        "hypothesis_policy_digest": value.hypothesis_policy_digest,
        "observation": value.observation.to_data(),
        "observation_digest": value.observation_digest,
        "frozen_rule": value.frozen_rule.to_data(),
        "frozen_rule_digest": value.frozen_rule_digest,
        "decision": value.decision.to_data(),
        "decision_digest": value.decision_digest,
        "raw_png_bytes_persisted": False,
        "query_labels_consumed": False,
        "evaluated_only_with_precommitted_frozen_rule": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelProgramOfficialQueryResult:
    algorithm_digest: str
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    task_freeze_digest: str
    task_commit_digest: str
    task_freeze_store_receipt_digest: str
    task_commit_store_receipt_digest: str
    sealed_query_panel_ids: tuple[str, str]
    query_panel_id: str
    released_query_panel_record_digest: str
    released_query_store_receipt_digest: str
    release_receipt_digest: str
    exact_png_digest: str
    exact_png_byte_count: int
    observer_algorithm_digest: str
    search_space_digest: str
    hypothesis_policy_digest: str
    observation: PanelProgramObservation
    observation_digest: str
    frozen_rule: FrozenProgramRule
    frozen_rule_digest: str
    decision: ProgramRuleDecision
    decision_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("algorithm digest", self.algorithm_digest),
            ("task plan digest", self.task_plan_digest),
            ("execution precommit digest", self.execution_precommit_digest),
            ("task freeze digest", self.task_freeze_digest),
            ("task commit digest", self.task_commit_digest),
            ("freeze store receipt digest", self.task_freeze_store_receipt_digest),
            ("commit store receipt digest", self.task_commit_store_receipt_digest),
            ("released query record digest", self.released_query_panel_record_digest),
            ("released query store receipt digest", self.released_query_store_receipt_digest),
            ("release receipt digest", self.release_receipt_digest),
            ("exact PNG digest", self.exact_png_digest),
            ("observer algorithm digest", self.observer_algorithm_digest),
            ("search-space digest", self.search_space_digest),
            ("hypothesis policy digest", self.hypothesis_policy_digest),
            ("observation digest", self.observation_digest),
            ("frozen rule digest", self.frozen_rule_digest),
            ("decision digest", self.decision_digest),
            ("query result digest", self.record_digest),
        ):
            _require_address(item, label)
        if type(self.observation) is not PanelProgramObservation:
            raise TypeError("observation has the wrong type")
        if type(self.frozen_rule) is not FrozenProgramRule:
            raise TypeError("frozen_rule has the wrong type")
        if type(self.decision) is not ProgramRuleDecision:
            raise TypeError("decision has the wrong type")
        restored_observation = PanelProgramObservation.from_data(
            self.observation.to_data()
        )
        restored_rule = FrozenProgramRule.from_data(self.frozen_rule.to_data())
        restored_decision = ProgramRuleDecision.from_data(self.decision.to_data())
        expected_decision = evaluate_frozen_program_rule(
            restored_rule, restored_observation
        )
        if (
            type(self.task_id) is not str
            or not self.task_id
            or type(self.sealed_query_panel_ids) is not tuple
            or len(self.sealed_query_panel_ids) != 2
            or len(set(self.sealed_query_panel_ids)) != 2
            or self.query_panel_id not in self.sealed_query_panel_ids
            or type(self.exact_png_byte_count) is not int
            or self.exact_png_byte_count <= 0
            or restored_observation != self.observation
            or restored_rule != self.frozen_rule
            or restored_decision != self.decision
            or expected_decision != self.decision
            or _observation_panel_digest(self.observation) != self.exact_png_digest
            or self.observation.observation_digest != self.observation_digest
            or self.frozen_rule.rule_digest != self.frozen_rule_digest
            or self.decision.decision_digest != self.decision_digest
            or self.record_digest != _content_address(_query_content(self))
            or _has_bytes(_query_content(self))
        ):
            raise PanelProgramOfficialTaskError("query result replay differs")

    def to_data(self) -> dict[str, object]:
        return {**_query_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelProgramOfficialQueryResult":
        raw = _fields(
            value,
            {
                "schema", "adapter_id", "algorithm_digest", "task_id",
                "task_plan_digest", "execution_precommit_digest",
                "task_freeze_digest", "task_commit_digest",
                "task_freeze_store_receipt_digest",
                "task_commit_store_receipt_digest", "sealed_query_panel_ids",
                "query_panel_id", "released_query_panel_record_digest",
                "released_query_store_receipt_digest", "release_receipt_digest",
                "exact_png_digest", "exact_png_byte_count",
                "observer_algorithm_digest", "search_space_digest",
                "hypothesis_policy_digest", "observation", "observation_digest",
                "frozen_rule", "frozen_rule_digest", "decision",
                "decision_digest", "raw_png_bytes_persisted",
                "query_labels_consumed",
                "evaluated_only_with_precommitted_frozen_rule",
                *_authority_data(), "record_digest",
            },
            "panel-program official query result",
        )
        if (
            raw["schema"] != PANEL_PROGRAM_OFFICIAL_QUERY_SCHEMA
            or raw["adapter_id"] != PANEL_PROGRAM_OFFICIAL_ADAPTER_ID
            or raw["raw_png_bytes_persisted"] is not False
            or raw["query_labels_consumed"] is not False
            or raw["evaluated_only_with_precommitted_frozen_rule"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["sealed_query_panel_ids"], list)
            or not isinstance(raw["observation"], Mapping)
            or not isinstance(raw["frozen_rule"], Mapping)
            or not isinstance(raw["decision"], Mapping)
        ):
            raise PanelProgramOfficialTaskError("query result policy differs")
        result = cls(
            raw["algorithm_digest"], raw["task_id"], raw["task_plan_digest"],
            raw["execution_precommit_digest"], raw["task_freeze_digest"],
            raw["task_commit_digest"], raw["task_freeze_store_receipt_digest"],
            raw["task_commit_store_receipt_digest"],
            tuple(raw["sealed_query_panel_ids"]), raw["query_panel_id"],
            raw["released_query_panel_record_digest"],
            raw["released_query_store_receipt_digest"],
            raw["release_receipt_digest"], raw["exact_png_digest"],
            raw["exact_png_byte_count"], raw["observer_algorithm_digest"],
            raw["search_space_digest"], raw["hypothesis_policy_digest"],
            PanelProgramObservation.from_data(raw["observation"]),
            raw["observation_digest"], FrozenProgramRule.from_data(raw["frozen_rule"]),
            raw["frozen_rule_digest"], ProgramRuleDecision.from_data(raw["decision"]),
            raw["decision_digest"], raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelProgramOfficialTaskError("query result is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PanelProgramOfficialQueryRuntime:
    result: PanelProgramOfficialQueryResult
    released_panel: ReleasedOfficialPanel = field(repr=False)
    released_panel_store_receipt: ObjectBongardWriteOnceReceipt
    result_store_receipt: ObjectBongardWriteOnceReceipt

    def __post_init__(self) -> None:
        if type(self.result) is not PanelProgramOfficialQueryResult:
            raise TypeError("result has the wrong type")
        if type(self.released_panel) is not ReleasedOfficialPanel:
            raise TypeError("released_panel has the wrong type")
        if type(self.released_panel_store_receipt) is not ObjectBongardWriteOnceReceipt:
            raise TypeError("released_panel_store_receipt has the wrong type")
        if type(self.result_store_receipt) is not ObjectBongardWriteOnceReceipt:
            raise TypeError("result_store_receipt has the wrong type")
        released = ReleasedOfficialPanel.from_data(self.released_panel.to_data())
        released_receipt = ObjectBongardWriteOnceReceipt.from_data(
            self.released_panel_store_receipt.to_data()
        )
        result_receipt = ObjectBongardWriteOnceReceipt.from_data(
            self.result_store_receipt.to_data()
        )
        if (
            released != self.released_panel
            or released.panel_id != self.result.query_panel_id
            or released.record_digest
            != self.result.released_query_panel_record_digest
            or released.release_receipt.record_digest != self.result.release_receipt_digest
            or released.exact_png_digest != self.result.exact_png_digest
            or len(released.exact_png_bytes) != self.result.exact_png_byte_count
            or released_receipt.object_kind != "released-query-panel"
            or released_receipt.object_digest != released.record_digest
            or released_receipt.record_digest
            != self.result.released_query_store_receipt_digest
            or result_receipt.object_kind != "panel-program-query-result"
            or result_receipt.object_digest != self.result.record_digest
        ):
            raise PanelProgramOfficialTaskError("query runtime binding differs")


def _cold_durable_decision(
    *, prepared: PreparedObjectBongardRelease,
    task: ObjectBongardTaskPlan,
    durable_decision: PanelProgramOfficialDurableDecision,
) -> PanelProgramOfficialDurableDecision:
    if type(durable_decision) is not PanelProgramOfficialDurableDecision:
        raise TypeError(
            "durable_decision must be exact PanelProgramOfficialDurableDecision"
        )
    freeze = PanelProgramOfficialTaskFreeze.from_data(
        durable_decision.freeze.to_data()
    )
    freeze = cold_verify_panel_program_official_task_freeze(
        freeze, expected_freeze_digest=durable_decision.freeze.record_digest
    )
    freeze_receipt = ObjectBongardWriteOnceReceipt.from_data(
        durable_decision.freeze_receipt.to_data()
    )
    commit = PanelProgramOfficialTaskCommit.from_data(
        durable_decision.commit.to_data()
    )
    commit_receipt = ObjectBongardWriteOnceReceipt.from_data(
        durable_decision.commit_receipt.to_data()
    )
    freeze_data = prepared.store.verify(
        freeze_receipt, expected_data=freeze.to_data()
    )
    commit_data = prepared.store.verify(
        commit_receipt, expected_data=commit.to_data()
    )
    exact_payload = canonical_json(freeze_data) + b"\n"
    commit = cold_verify_panel_program_official_task_commit(
        PanelProgramOfficialTaskCommit.from_data(commit_data),
        freeze=freeze, exact_freeze_payload=exact_payload,
        task_freeze_store_receipt=freeze_receipt,
        expected_commit_digest=durable_decision.commit.record_digest,
    )
    support = freeze.support_artifact
    runtime = _verify_support_runtime_store(
        prepared=prepared, support_runtime=durable_decision.support_runtime
    )
    expected_query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    if (
        freeze.task_id != task.task_id
        or runtime.artifact != support
        or freeze.task_plan_digest != task.record_digest
        or freeze.prepared_batch_plan_digest != prepared.plan.record_digest
        or freeze.execution_precommit_digest != prepared.precommit.record_digest
        or freeze.release_authorization_digest != prepared.authorization.record_digest
        or freeze.exposure_successor_digest != prepared.successor.digest
        or freeze.sealed_query_panel_ids != expected_query_ids
        or support.plan_store_receipt_digest != prepared.plan_receipt.record_digest
        or support.precommit_store_receipt_digest != prepared.precommit_receipt.record_digest
        or support.exposure_store_receipt_digest != prepared.exposure_receipt.record_digest
        or support.authorization_store_receipt_digest
        != prepared.authorization_receipt.record_digest
    ):
        raise PanelProgramOfficialTaskError("durable decision custody differs")
    return PanelProgramOfficialDurableDecision(
        runtime, freeze, freeze_receipt, commit, commit_receipt
    )


def release_and_evaluate_panel_program_official_query(
    *,
    task: ObjectBongardTaskPlan,
    prepared: PreparedObjectBongardRelease,
    archive: OfficialPanelArchive,
    panel_id: str,
    durable_decision: PanelProgramOfficialDurableDecision,
    observe_program: ProgramObserver,
) -> PanelProgramOfficialQueryRuntime:
    """Release one sealed query only after cold durable replay, then evaluate it."""

    if type(task) is not ObjectBongardTaskPlan:
        raise TypeError("task must be exact ObjectBongardTaskPlan")
    if type(prepared) is not PreparedObjectBongardRelease:
        raise TypeError("prepared must be exact PreparedObjectBongardRelease")
    if type(archive) is not OfficialPanelArchive:
        raise TypeError("archive must be exact OfficialPanelArchive")
    _require_exact_observer(observe_program)
    frozen_task = ObjectBongardTaskPlan.from_data(task.to_data())
    prepared = verify_prepared_object_bongard_release(prepared)
    matches = tuple(item for item in prepared.plan.tasks if item.task_id == task.task_id)
    if len(matches) != 1 or matches[0] != frozen_task:
        raise PanelProgramOfficialTaskError("task differs from prepared batch")
    durable = _cold_durable_decision(
        prepared=prepared, task=frozen_task, durable_decision=durable_decision
    )
    if panel_id not in durable.freeze.sealed_query_panel_ids:
        raise PanelProgramOfficialTaskError("panel is not a sealed query")
    frozen_bindings = dict(prepared.precommit.runtime_source_bindings)
    required_bindings = panel_program_required_precommit_bindings()
    if any(frozen_bindings.get(key) != value for key, value in required_bindings.items()):
        raise PanelProgramOfficialTaskError(
            "query observer was not frozen in the execution precommit"
        )
    released, released_store_receipt = release_object_bongard_query_panel(
        prepared=prepared,
        archive=archive,
        panel_id=panel_id,
        task_freeze=durable.freeze,
        task_commit=durable.commit,
        task_freeze_receipt=durable.freeze_receipt,
        task_commit_receipt=durable.commit_receipt,
    )
    observation = observe_program(released.exact_png_bytes)
    if type(observation) is not PanelProgramObservation:
        raise TypeError("observer must return exact PanelProgramObservation")
    observation = PanelProgramObservation.from_data(observation.to_data())
    support = durable.freeze.support_artifact
    if (
        _observation_panel_digest(observation) != released.exact_png_digest
        or observation.observer_algorithm_digest != support.observer_algorithm_digest
        or observation.search_space_digest != support.search_space_digest
        or observation.hypothesis_policy_digest != support.hypothesis_policy_digest
    ):
        raise PanelProgramOfficialTaskError("query observer policy or PNG binding differs")
    decision = evaluate_frozen_program_rule(durable.freeze.selected_rule, observation)
    if type(decision) is not ProgramRuleDecision:
        raise TypeError("rule evaluator must return exact ProgramRuleDecision")
    decision = ProgramRuleDecision.from_data(decision.to_data())
    values = {
        "algorithm_digest": panel_program_official_task_algorithm_digest(),
        "task_id": frozen_task.task_id,
        "task_plan_digest": frozen_task.record_digest,
        "execution_precommit_digest": prepared.precommit.record_digest,
        "task_freeze_digest": durable.freeze.record_digest,
        "task_commit_digest": durable.commit.record_digest,
        "task_freeze_store_receipt_digest": durable.freeze_receipt.record_digest,
        "task_commit_store_receipt_digest": durable.commit_receipt.record_digest,
        "sealed_query_panel_ids": durable.freeze.sealed_query_panel_ids,
        "query_panel_id": released.panel_id,
        "released_query_panel_record_digest": released.record_digest,
        "released_query_store_receipt_digest": released_store_receipt.record_digest,
        "release_receipt_digest": released.release_receipt.record_digest,
        "exact_png_digest": released.exact_png_digest,
        "exact_png_byte_count": len(released.exact_png_bytes),
        "observer_algorithm_digest": observation.observer_algorithm_digest,
        "search_space_digest": observation.search_space_digest,
        "hypothesis_policy_digest": observation.hypothesis_policy_digest,
        "observation": observation,
        "observation_digest": observation.observation_digest,
        "frozen_rule": durable.freeze.selected_rule,
        "frozen_rule_digest": durable.freeze.selected_rule.rule_digest,
        "decision": decision,
        "decision_digest": decision.decision_digest,
    }
    provisional = object.__new__(PanelProgramOfficialQueryResult)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    result = PanelProgramOfficialQueryResult(
        **values, record_digest=_content_address(_query_content(provisional))
    )
    result_receipt = prepared.store.persist(
        object_kind="panel-program-query-result",
        object_digest=result.record_digest,
        data=result.to_data(),
    )
    replay = PanelProgramOfficialQueryResult.from_data(
        prepared.store.verify(result_receipt, expected_data=result.to_data())
    )
    if replay != result:
        raise PanelProgramOfficialTaskError("durable query result replay differs")
    return PanelProgramOfficialQueryRuntime(
        replay, released, released_store_receipt, result_receipt
    )


__all__ = (
    "PANEL_PROGRAM_OFFICIAL_ADAPTER_ID",
    "PANEL_PROGRAM_OFFICIAL_QUERY_SCHEMA",
    "PANEL_PROGRAM_OFFICIAL_SUPPORT_PANEL_SCHEMA",
    "PANEL_PROGRAM_OFFICIAL_SUPPORT_SCHEMA",
    "PANEL_PROGRAM_OFFICIAL_TASK_COMMIT_SCHEMA",
    "PANEL_PROGRAM_OFFICIAL_TASK_FREEZE_SCHEMA",
    "PANEL_PROGRAM_SUPPORT_BUCKET_SIZE",
    "PANEL_PROGRAM_SUPPORT_COUNT",
    "PanelProgramOfficialDurableDecision",
    "PanelProgramOfficialQueryResult",
    "PanelProgramOfficialQueryRuntime",
    "PanelProgramOfficialSupportArtifact",
    "PanelProgramOfficialSupportPanel",
    "PanelProgramOfficialSupportRuntime",
    "PanelProgramOfficialTaskCommit",
    "PanelProgramOfficialTaskError",
    "PanelProgramOfficialTaskFreeze",
    "ProgramObserver",
    "build_panel_program_official_support",
    "cold_verify_panel_program_official_task_commit",
    "cold_verify_panel_program_official_task_freeze",
    "commit_panel_program_official_task_decision",
    "freeze_panel_program_official_task_decision",
    "panel_program_official_task_algorithm_digest",
    "panel_program_required_precommit_bindings",
    "panel_program_official_task_source_digest",
    "persist_panel_program_official_task_decision",
    "release_and_evaluate_panel_program_official_query",
    "verify_panel_program_query_release_authority",
)
