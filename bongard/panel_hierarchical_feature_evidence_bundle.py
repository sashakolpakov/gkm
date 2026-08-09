"""Exact full-receipt custody for hierarchical panel-feature observations.

This layer retains the only values from which a panel observation set may be
reconstructed: exact PNG bytes and one complete
:class:`HierarchicalPanelCodexArtifact`.  Phase, index, and corpus panel ID are
outer custody metadata only; they are never added to the observer request,
prompt, schema, image presentation, or model payload.

One bundle binds the exact twelve-panel proposer call/result, twelve support
rows, and either zero query rows or the complete two-query phase.  Cold replay
revalidates the proposer and every hierarchical artifact without invoking a
model.  No API accepts a bare :class:`PanelFeatureObservationSet`.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.panel_batched_typed_codex_observer import (
    complete_whole_panel_feature_axes,
)
from bongard.panel_feature_observation import PanelFeatureObservationSet
from bongard.panel_feature_proposer import (
    PanelFeatureProposerResult,
    parse_panel_feature_proposer_payload,
)
from bongard.panel_hierarchical_visual_adapter import (
    HIERARCHICAL_PANEL_PROTOCOL_ID,
    HierarchicalPanelCodexArtifact,
    verify_hierarchical_panel_artifact,
)
from bongard.panel_typed_codex_observer import (
    TypedCodexRuntimeBinding,
    TypedProposerCodexCallArtifact,
    _exact_png,
    typed_codex_observer_contract_digest,
    typed_measurement_protocol_digest,
    verify_typed_proposer_codex_artifact,
)


HIERARCHICAL_FEATURE_EVIDENCE_ROW_SCHEMA = (
    "gkm.bongard-hierarchical-panel-feature-evidence-row.v1"
)
HIERARCHICAL_FEATURE_EVIDENCE_BUNDLE_SCHEMA = (
    "gkm.bongard-hierarchical-panel-feature-evidence-bundle.v1"
)
HIERARCHICAL_FEATURE_EVIDENCE_PROTOCOL_ID = (
    "bongard.panel-hierarchical-feature-evidence/"
    "exact-artifacts-python-zero-call-replay-v1"
)
SUPPORT_PANEL_COUNT = 12
QUERY_PANEL_COUNT = 2

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class HierarchicalPanelFeatureEvidenceError(ValueError):
    """Hierarchical full-receipt custody is incomplete or inconsistent."""


class HierarchicalFeatureEvidencePhase(str, Enum):
    SUPPORT = "support"
    QUERY = "query"


def panel_hierarchical_feature_evidence_bundle_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise HierarchicalPanelFeatureEvidenceError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise HierarchicalPanelFeatureEvidenceError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise HierarchicalPanelFeatureEvidenceError(
            f"{label} must be a sha256: address"
        )
    return value


def _panel_id(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 1024
    ):
        raise HierarchicalPanelFeatureEvidenceError(
            "hierarchical panel ID is not bounded canonical text"
        )
    return value


def _panel_bytes(value: object) -> bytes:
    try:
        result = _exact_png(value, "hierarchical evidence panel")
    except Exception as exc:
        raise HierarchicalPanelFeatureEvidenceError(
            "hierarchical evidence panel is not an exact bounded PNG"
        ) from exc
    if type(result) is not bytes:
        raise HierarchicalPanelFeatureEvidenceError(
            "hierarchical evidence PNG validator returned the wrong type"
        )
    return result


def _phase_index(phase: HierarchicalFeatureEvidencePhase, value: object) -> int:
    maximum = (
        SUPPORT_PANEL_COUNT
        if phase is HierarchicalFeatureEvidencePhase.SUPPORT
        else QUERY_PANEL_COUNT
    )
    if type(value) is not int or value not in range(maximum):
        raise HierarchicalPanelFeatureEvidenceError(
            "hierarchical panel phase index differs"
        )
    return value


def _row_content(value: "HierarchicalPanelFeatureEvidenceRow") -> dict[str, object]:
    return {
        "schema": HIERARCHICAL_FEATURE_EVIDENCE_ROW_SCHEMA,
        "phase": value.phase.value,
        "phase_index": value.phase_index,
        "panel_id": value.panel_id,
        "panel_png_base64": base64.b64encode(value.panel_png).decode("ascii"),
        "panel_png_digest": value.panel_png_digest,
        "panel_png_byte_count": len(value.panel_png),
        "hierarchical_artifact": value.artifact.to_data(),
        "hierarchical_artifact_digest": value.artifact.artifact_digest,
        "observation_set_digest": value.artifact.observation_set.observation_set_digest,
        "phase_index_or_panel_id_model_visible": False,
        "task_side_or_class_label_model_visible": False,
        "formula_or_candidate_identifier_model_visible": False,
        "observer_presentation_name": "panel.png",
        "bare_observation_set_archived": False,
        "observation_reconstruction_source": "verified_hierarchical_artifact_only",
    }


@dataclass(frozen=True, slots=True)
class HierarchicalPanelFeatureEvidenceRow:
    """One phase-local exact PNG and its sole full observer artifact."""

    phase: HierarchicalFeatureEvidencePhase
    phase_index: int
    panel_id: str
    panel_png: bytes
    artifact: HierarchicalPanelCodexArtifact
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.phase) is not HierarchicalFeatureEvidencePhase:
            raise TypeError("hierarchical evidence phase must be exact")
        _phase_index(self.phase, self.phase_index)
        _panel_id(self.panel_id)
        panel = _panel_bytes(self.panel_png)
        if panel != self.panel_png:
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence panel bytes changed during validation"
            )
        if type(self.artifact) is not HierarchicalPanelCodexArtifact:
            raise TypeError(
                "hierarchical evidence row needs HierarchicalPanelCodexArtifact"
            )
        if (
            self.artifact.panel_png_digest != self.panel_png_digest
            or self.artifact.panel_png_byte_count != len(self.panel_png)
        ):
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical artifact belongs to different exact panel bytes"
            )
        _digest(self.record_digest, "hierarchical evidence row digest")
        if self.record_digest != canonical_digest(_row_content(self)):
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence row digest differs"
            )

    @property
    def panel_png_digest(self) -> str:
        return hashlib.sha256(self.panel_png).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        phase: HierarchicalFeatureEvidencePhase,
        phase_index: int,
        panel_id: str,
        panel_png: bytes,
        artifact: HierarchicalPanelCodexArtifact,
    ) -> "HierarchicalPanelFeatureEvidenceRow":
        """Verify exact pixels/full receipt and create one artifact-only row."""

        panel = _panel_bytes(panel_png)
        if type(artifact) is not HierarchicalPanelCodexArtifact:
            raise TypeError(
                "hierarchical evidence row needs HierarchicalPanelCodexArtifact"
            )
        try:
            verified = verify_hierarchical_panel_artifact(
                artifact,
                panel,
                expected_artifact_digest=artifact.artifact_digest,
            )
        except Exception as exc:
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence row artifact fails exact zero-call verification"
            ) from exc
        values = {
            "phase": phase,
            "phase_index": phase_index,
            "panel_id": panel_id,
            "panel_png": panel,
            "artifact": verified,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest=canonical_digest(_row_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_row_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "HierarchicalPanelFeatureEvidenceRow":
        raw = _fields(
            value,
            {
                "schema",
                "phase",
                "phase_index",
                "panel_id",
                "panel_png_base64",
                "panel_png_digest",
                "panel_png_byte_count",
                "hierarchical_artifact",
                "hierarchical_artifact_digest",
                "observation_set_digest",
                "phase_index_or_panel_id_model_visible",
                "task_side_or_class_label_model_visible",
                "formula_or_candidate_identifier_model_visible",
                "observer_presentation_name",
                "bare_observation_set_archived",
                "observation_reconstruction_source",
                "record_digest",
            },
            "hierarchical panel evidence row",
        )
        if (
            raw["schema"] != HIERARCHICAL_FEATURE_EVIDENCE_ROW_SCHEMA
            or raw["phase_index_or_panel_id_model_visible"] is not False
            or raw["task_side_or_class_label_model_visible"] is not False
            or raw["formula_or_candidate_identifier_model_visible"] is not False
            or raw["observer_presentation_name"] != "panel.png"
            or raw["bare_observation_set_archived"] is not False
            or raw["observation_reconstruction_source"]
            != "verified_hierarchical_artifact_only"
            or type(raw["panel_png_base64"]) is not str
        ):
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence row policy differs"
            )
        try:
            phase = HierarchicalFeatureEvidencePhase(raw["phase"])
            panel = base64.b64decode(raw["panel_png_base64"], validate=True)
            artifact = HierarchicalPanelCodexArtifact.from_data(
                raw["hierarchical_artifact"]
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, HierarchicalPanelFeatureEvidenceError):
                raise
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence row encoding differs"
            ) from exc
        if (
            base64.b64encode(panel).decode("ascii") != raw["panel_png_base64"]
            or hashlib.sha256(panel).hexdigest() != raw["panel_png_digest"]
            or len(panel) != raw["panel_png_byte_count"]
            or artifact.artifact_digest != raw["hierarchical_artifact_digest"]
            or artifact.observation_set.observation_set_digest
            != raw["observation_set_digest"]
        ):
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence row byte or artifact commitment differs"
            )
        try:
            verified = verify_hierarchical_panel_artifact(
                artifact,
                panel,
                expected_artifact_digest=artifact.artifact_digest,
            )
            result = cls(
                phase,
                raw["phase_index"],
                raw["panel_id"],
                panel,
                verified,
                raw["record_digest"],
            )
        except Exception as exc:
            if isinstance(exc, HierarchicalPanelFeatureEvidenceError):
                raise
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence row artifact fails verification"
            ) from exc
        if result.to_data() != dict(raw):
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence row is not canonical"
            )
        return result


def _row_order_key(value: HierarchicalPanelFeatureEvidenceRow) -> tuple[int, int]:
    return (
        0 if value.phase is HierarchicalFeatureEvidencePhase.SUPPORT else 1,
        value.phase_index,
    )


def _rebuild_proposer_result(
    artifact: TypedProposerCodexCallArtifact,
) -> PanelFeatureProposerResult:
    try:
        return parse_panel_feature_proposer_payload(
            artifact.model_payload,
            proposer_receipt_digest=artifact.artifact_digest,
            support_set_digest=artifact.presentation_digest,
            task_context_digest=artifact.task_context_digest,
            block_orientations=artifact.block_orientations,
        )
    except (TypeError, ValueError) as exc:
        raise HierarchicalPanelFeatureEvidenceError(
            "proposer result cannot be replayed from its exact full artifact"
        ) from exc


def _receipt_digests(
    proposer: TypedProposerCodexCallArtifact,
    panels: Sequence[HierarchicalPanelFeatureEvidenceRow],
) -> tuple[str, ...]:
    return (
        proposer.codex_receipt.receipt_digest,
        *(item.artifact.codex_receipt.receipt_digest for item in panels),
    )


def _axis_catalog_data() -> list[dict[str, object]]:
    return [item.to_data() for item in complete_whole_panel_feature_axes()]


def _bundle_content(
    value: "HierarchicalPanelFeatureEvidenceBundle",
) -> dict[str, object]:
    support_count = sum(
        item.phase is HierarchicalFeatureEvidencePhase.SUPPORT
        for item in value.panels
    )
    query_count = len(value.panels) - support_count
    axes = _axis_catalog_data()
    receipts = _receipt_digests(value.proposer_artifact, value.panels)
    return {
        "schema": HIERARCHICAL_FEATURE_EVIDENCE_BUNDLE_SCHEMA,
        "protocol_id": HIERARCHICAL_FEATURE_EVIDENCE_PROTOCOL_ID,
        "protocol_source_digest": (
            panel_hierarchical_feature_evidence_bundle_source_digest()
        ),
        "task_context_digest": value.proposer_artifact.task_context_digest,
        "proposer_artifact": value.proposer_artifact.to_data(),
        "proposer_artifact_digest": value.proposer_artifact.artifact_digest,
        "proposer_result": value.proposer_result.to_data(),
        "proposer_result_digest": value.proposer_result.result_digest,
        "shared_runtime": value.observer_runtime.to_data(),
        "shared_runtime_digest": value.observer_runtime.runtime_digest,
        "shared_model_catalog_digest": value.observer_runtime.model_catalog_digest,
        "observer_contract_digest": typed_codex_observer_contract_digest(
            value.observer_runtime
        ),
        "measurement_protocol_digest": typed_measurement_protocol_digest(
            value.observer_runtime
        ),
        "hierarchical_contract_digest": (
            value.panels[0].artifact.hierarchical_contract_digest
        ),
        "hierarchical_protocol_id": HIERARCHICAL_PANEL_PROTOCOL_ID,
        "observer_axes": axes,
        "observer_axis_order": "axis-digest-ascending",
        "observer_axis_catalog_digest": canonical_digest(axes),
        "panels": [item.to_data() for item in value.panels],
        "panel_order": "support-index-then-query-index",
        "support_panel_count": support_count,
        "query_panel_count": query_count,
        "query_phase_complete": query_count == QUERY_PANEL_COUNT,
        "proposer_model_call_count": 1,
        "hierarchical_observer_model_call_count": len(value.panels),
        "live_model_call_count": 1 + len(value.panels),
        "physical_receipt_digests": list(receipts),
        "model_call_count_derived_from_unique_full_receipts": True,
        "all_model_payloads_and_full_receipts_retained": True,
        "bare_observation_sets_accepted": False,
        "observation_reconstruction_source": "verified_hierarchical_artifacts_only",
        "row_phase_index_or_panel_id_passed_to_observer": False,
        "task_side_or_class_label_passed_to_observer": False,
        "formula_or_candidate_identifier_passed_to_observer": False,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "cold_replay_model_call_count": 0,
    }


@dataclass(frozen=True, slots=True)
class HierarchicalPanelFeatureEvidenceBundle:
    """Exact proposer plus twelve support and optional two query artifacts."""

    proposer_artifact: TypedProposerCodexCallArtifact
    proposer_result: PanelFeatureProposerResult
    observer_runtime: TypedCodexRuntimeBinding
    panels: tuple[HierarchicalPanelFeatureEvidenceRow, ...]
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.proposer_artifact) is not TypedProposerCodexCallArtifact:
            raise TypeError(
                "hierarchical evidence bundle needs a full typed proposer artifact"
            )
        if type(self.proposer_result) is not PanelFeatureProposerResult:
            raise TypeError(
                "hierarchical evidence bundle needs an exact proposer result"
            )
        if type(self.observer_runtime) is not TypedCodexRuntimeBinding:
            raise TypeError(
                "hierarchical evidence bundle needs an exact shared runtime"
            )
        if (
            type(self.panels) is not tuple
            or any(
                type(item) is not HierarchicalPanelFeatureEvidenceRow
                for item in self.panels
            )
            or self.panels != tuple(sorted(self.panels, key=_row_order_key))
        ):
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence rows must be canonical support-then-query rows"
            )
        _digest(self.record_digest, "hierarchical evidence bundle digest")
        self._verify_full_custody(cold_replay=False)
        if self.record_digest != canonical_digest(_bundle_content(self)):
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence bundle digest differs"
            )

    @property
    def bundle_address(self) -> str:
        return "sha256:" + self.record_digest

    @property
    def observer_axes(self):
        return complete_whole_panel_feature_axes()

    @property
    def live_model_call_count(self) -> int:
        return 1 + len(self.panels)

    @property
    def physical_receipt_digests(self) -> tuple[str, ...]:
        return _receipt_digests(self.proposer_artifact, self.panels)

    def panels_for_phase(
        self, phase: HierarchicalFeatureEvidencePhase
    ) -> tuple[HierarchicalPanelFeatureEvidenceRow, ...]:
        if type(phase) is not HierarchicalFeatureEvidencePhase:
            raise TypeError("hierarchical evidence phase must be exact")
        return tuple(item for item in self.panels if item.phase is phase)

    def _verify_full_custody(self, *, cold_replay: bool) -> None:
        support = self.panels_for_phase(HierarchicalFeatureEvidencePhase.SUPPORT)
        query = self.panels_for_phase(HierarchicalFeatureEvidencePhase.QUERY)
        if (
            len(support) != SUPPORT_PANEL_COUNT
            or tuple(item.phase_index for item in support)
            != tuple(range(SUPPORT_PANEL_COUNT))
            or len(query) not in (0, QUERY_PANEL_COUNT)
            or tuple(item.phase_index for item in query) != tuple(range(len(query)))
            or len({item.panel_id for item in self.panels}) != len(self.panels)
        ):
            raise HierarchicalPanelFeatureEvidenceError(
                "bundle needs twelve support rows and zero or two canonical query rows"
            )
        support_identity = tuple(
            (len(item.panel_png), item.panel_png_digest) for item in support
        )
        proposer_identity = tuple(
            (item.byte_count, item.content_digest)
            for item in self.proposer_artifact.presentation
        )
        if support_identity != proposer_identity:
            raise HierarchicalPanelFeatureEvidenceError(
                "proposer artifact does not bind the exact ordered support PNG bytes"
            )
        replayed_result = _rebuild_proposer_result(self.proposer_artifact)
        if replayed_result != self.proposer_result:
            raise HierarchicalPanelFeatureEvidenceError(
                "proposer result differs from the exact proposer artifact"
            )
        if self.proposer_artifact.runtime != self.observer_runtime:
            raise HierarchicalPanelFeatureEvidenceError(
                "proposer and hierarchical observers do not share one exact runtime"
            )
        axes = complete_whole_panel_feature_axes()
        axis_digests = tuple(item.axis_digest for item in axes)
        observer_contract = typed_codex_observer_contract_digest(
            self.observer_runtime
        )
        measurement_protocol = typed_measurement_protocol_digest(
            self.observer_runtime
        )
        hierarchical_contract = self.panels[0].artifact.hierarchical_contract_digest
        for row in self.panels:
            artifact = row.artifact
            observation_axes = tuple(
                item.axis.axis_digest
                for item in artifact.observation_set.axis_observations
            )
            if (
                artifact.runtime != self.observer_runtime
                or artifact.runtime.model_catalog_digest
                != self.observer_runtime.model_catalog_digest
                or artifact.observer_contract_digest != observer_contract
                or artifact.measurement_protocol_digest != measurement_protocol
                or artifact.hierarchical_contract_digest != hierarchical_contract
                or tuple(item.axis_digest for item in artifact.request.axes)
                != axis_digests
                or observation_axes != axis_digests
                or artifact.panel_png_digest != row.panel_png_digest
                or artifact.panel_png_byte_count != len(row.panel_png)
            ):
                raise HierarchicalPanelFeatureEvidenceError(
                    "row runtime, catalog, contract, or exact panel custody differs"
                )
        receipts = self.physical_receipt_digests
        if len(receipts) != len(set(receipts)):
            raise HierarchicalPanelFeatureEvidenceError(
                "one physical Codex receipt is duplicated across exact artifacts"
            )
        if self.proposer_result.receipt_digest != self.proposer_artifact.artifact_digest:
            raise HierarchicalPanelFeatureEvidenceError(
                "proposer result receipt is not the retained proposer artifact address"
            )
        if cold_replay:
            self._cold_verify_artifacts(support)

    def _cold_verify_artifacts(
        self, support: Sequence[HierarchicalPanelFeatureEvidenceRow]
    ) -> None:
        jobs: list[tuple[str, object, tuple[object, ...], dict[str, object]]] = [
            (
                "proposer",
                verify_typed_proposer_codex_artifact,
                (self.proposer_artifact, tuple(item.panel_png for item in support)),
                {"expected_artifact_digest": self.proposer_artifact.artifact_digest},
            )
        ]
        jobs.extend(
            (
                f"hierarchical-panel:{row.phase.value}:{row.phase_index}",
                verify_hierarchical_panel_artifact,
                (row.artifact, row.panel_png),
                {"expected_artifact_digest": row.artifact.artifact_digest},
            )
            for row in self.panels
        )
        with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as executor:
            futures = {
                executor.submit(call, *args, **kwargs): label
                for label, call, args, kwargs in jobs
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise HierarchicalPanelFeatureEvidenceError(
                        f"full {label} artifact fails zero-call cold replay"
                    ) from exc

    @classmethod
    def create(
        cls,
        *,
        proposer_artifact: TypedProposerCodexCallArtifact,
        proposer_result: PanelFeatureProposerResult,
        panels: Sequence[HierarchicalPanelFeatureEvidenceRow],
    ) -> "HierarchicalPanelFeatureEvidenceBundle":
        if isinstance(panels, (bytes, str, Mapping)):
            raise TypeError("hierarchical evidence rows must be an ordered sequence")
        rows = tuple(panels)
        if not rows:
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence bundle cannot derive a shared runtime"
            )
        if any(
            type(item) is not HierarchicalPanelFeatureEvidenceRow for item in rows
        ):
            raise TypeError(
                "hierarchical evidence bundle needs exact evidence rows"
            )
        values = {
            "proposer_artifact": proposer_artifact,
            "proposer_result": proposer_result,
            "observer_runtime": rows[0].artifact.runtime,
            "panels": rows,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest=canonical_digest(_bundle_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_bundle_content(self),
            "record_digest": self.record_digest,
            "bundle_address": self.bundle_address,
        }

    @classmethod
    def from_data(cls, value: object) -> "HierarchicalPanelFeatureEvidenceBundle":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "protocol_source_digest",
                "task_context_digest",
                "proposer_artifact",
                "proposer_artifact_digest",
                "proposer_result",
                "proposer_result_digest",
                "shared_runtime",
                "shared_runtime_digest",
                "shared_model_catalog_digest",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "hierarchical_contract_digest",
                "hierarchical_protocol_id",
                "observer_axes",
                "observer_axis_order",
                "observer_axis_catalog_digest",
                "panels",
                "panel_order",
                "support_panel_count",
                "query_panel_count",
                "query_phase_complete",
                "proposer_model_call_count",
                "hierarchical_observer_model_call_count",
                "live_model_call_count",
                "physical_receipt_digests",
                "model_call_count_derived_from_unique_full_receipts",
                "all_model_payloads_and_full_receipts_retained",
                "bare_observation_sets_accepted",
                "observation_reconstruction_source",
                "row_phase_index_or_panel_id_passed_to_observer",
                "task_side_or_class_label_passed_to_observer",
                "formula_or_candidate_identifier_passed_to_observer",
                "python_is_canonical_authority",
                "lean_present",
                "lean_required",
                "cold_replay_model_call_count",
                "record_digest",
                "bundle_address",
            },
            "hierarchical feature evidence bundle",
        )
        if (
            raw["schema"] != HIERARCHICAL_FEATURE_EVIDENCE_BUNDLE_SCHEMA
            or raw["protocol_id"] != HIERARCHICAL_FEATURE_EVIDENCE_PROTOCOL_ID
            or raw["protocol_source_digest"]
            != panel_hierarchical_feature_evidence_bundle_source_digest()
            or raw["hierarchical_protocol_id"] != HIERARCHICAL_PANEL_PROTOCOL_ID
            or raw["observer_axis_order"] != "axis-digest-ascending"
            or raw["panel_order"] != "support-index-then-query-index"
            or raw["proposer_model_call_count"] != 1
            or raw["model_call_count_derived_from_unique_full_receipts"] is not True
            or raw["all_model_payloads_and_full_receipts_retained"] is not True
            or raw["bare_observation_sets_accepted"] is not False
            or raw["observation_reconstruction_source"]
            != "verified_hierarchical_artifacts_only"
            or raw["row_phase_index_or_panel_id_passed_to_observer"] is not False
            or raw["task_side_or_class_label_passed_to_observer"] is not False
            or raw["formula_or_candidate_identifier_passed_to_observer"] is not False
            or raw["python_is_canonical_authority"] is not True
            or raw["lean_present"] is not False
            or raw["lean_required"] is not False
            or raw["cold_replay_model_call_count"] != 0
            or type(raw["observer_axes"]) is not list
            or type(raw["panels"]) is not list
            or type(raw["physical_receipt_digests"]) is not list
        ):
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence bundle policy differs"
            )
        try:
            proposer = TypedProposerCodexCallArtifact.from_data(
                raw["proposer_artifact"]
            )
            replayed_result = _rebuild_proposer_result(proposer)
            if replayed_result.to_data() != raw["proposer_result"]:
                raise HierarchicalPanelFeatureEvidenceError(
                    "archived proposer result is not artifact-derived"
                )
            result = cls(
                proposer,
                replayed_result,
                TypedCodexRuntimeBinding.from_data(raw["shared_runtime"]),
                tuple(
                    HierarchicalPanelFeatureEvidenceRow.from_data(item)
                    for item in raw["panels"]
                ),
                raw["record_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, HierarchicalPanelFeatureEvidenceError):
                raise
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence bundle typed value differs"
            ) from exc
        if result.to_data() != dict(raw):
            raise HierarchicalPanelFeatureEvidenceError(
                "hierarchical evidence bundle derived links or counts differ"
            )
        return result


def cold_replay_hierarchical_panel_feature_evidence_bundle(
    bundle: HierarchicalPanelFeatureEvidenceBundle,
    *,
    expected_bundle_address: str,
) -> HierarchicalPanelFeatureEvidenceBundle:
    """Reverify exact support/proposer/panel artifacts with zero model calls."""

    if type(bundle) is not HierarchicalPanelFeatureEvidenceBundle:
        raise TypeError(
            "cold replay needs HierarchicalPanelFeatureEvidenceBundle"
        )
    expected = _address(
        expected_bundle_address, "expected hierarchical evidence bundle address"
    )
    restored = HierarchicalPanelFeatureEvidenceBundle.from_data(bundle.to_data())
    if restored.bundle_address != expected:
        raise HierarchicalPanelFeatureEvidenceError(
            "hierarchical evidence bundle differs from external commitment"
        )
    restored._verify_full_custody(cold_replay=True)
    return restored


def verified_hierarchical_observation_sets(
    bundle: HierarchicalPanelFeatureEvidenceBundle,
    *,
    phase: HierarchicalFeatureEvidencePhase,
    expected_bundle_address: str,
) -> tuple[PanelFeatureObservationSet, ...]:
    """Return artifact-derived observations only after complete cold replay."""

    if type(phase) is not HierarchicalFeatureEvidencePhase:
        raise TypeError("hierarchical evidence phase must be exact")
    verified = cold_replay_hierarchical_panel_feature_evidence_bundle(
        bundle,
        expected_bundle_address=expected_bundle_address,
    )
    return tuple(
        row.artifact.observation_set for row in verified.panels_for_phase(phase)
    )


__all__ = (
    "HIERARCHICAL_FEATURE_EVIDENCE_BUNDLE_SCHEMA",
    "HIERARCHICAL_FEATURE_EVIDENCE_PROTOCOL_ID",
    "HierarchicalFeatureEvidencePhase",
    "HierarchicalPanelFeatureEvidenceBundle",
    "HierarchicalPanelFeatureEvidenceError",
    "HierarchicalPanelFeatureEvidenceRow",
    "cold_replay_hierarchical_panel_feature_evidence_bundle",
    "panel_hierarchical_feature_evidence_bundle_source_digest",
    "verified_hierarchical_observation_sets",
)
