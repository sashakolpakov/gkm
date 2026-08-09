"""Content-addressed full-receipt custody for typed panel-feature evidence.

The task runner intentionally consumes compact proposer and observation values.
Those values are not, by themselves, evidence that a pinned headless Codex call
occurred.  This module closes that boundary without making Lean authoritative:
it retains every full production call artifact, the exact PNG bytes to which it
was bound, and the compact Python values derived from those artifacts.

Support/query roles and corpus panel IDs live only in the outer custody rows.
Owner and axis artifacts are replayed through the fixed ``panel.png`` typed
observer protocols; no formula, selected predicate, side label, or phase tag is
an input to an observer artifact.
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
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.panel_batched_typed_codex_observer import (
    TypedBatchedAxisCodexArtifact,
    verify_typed_batched_axis_codex_artifact,
)
from bongard.panel_feature_observation import FeatureAxis, PanelFeatureObservationSet
from bongard.panel_feature_proposer import (
    PanelFeatureProposerResult,
    parse_panel_feature_proposer_payload,
)
from bongard.panel_typed_codex_observer import (
    TypedAxisCodexArtifact,
    TypedCodexRuntimeBinding,
    TypedOwnerCodexArtifact,
    TypedProposerCodexCallArtifact,
    typed_codex_observer_contract_digest,
    typed_measurement_protocol_digest,
    verify_typed_axis_codex_artifact,
    verify_typed_owner_codex_artifact,
    verify_typed_proposer_codex_artifact,
)


PANEL_FEATURE_EVIDENCE_BUNDLE_SCHEMA = (
    "gkm.bongard-panel-feature-full-receipt-evidence-bundle.v2"
)
PANEL_FEATURE_EVIDENCE_PANEL_SCHEMA = (
    "gkm.bongard-panel-feature-full-receipt-panel.v2"
)
PANEL_FEATURE_EVIDENCE_PROTOCOL_ID = (
    "bongard.panel-feature-evidence/full-codex-receipts-python-v2"
)
SUPPORT_PANEL_COUNT = 12
QUERY_PANEL_COUNT = 2

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PanelFeatureEvidenceBundleError(ValueError):
    """Full receipt custody is missing, duplicated, opaque, or inconsistent."""


class PanelFeatureEvidencePhase(str, Enum):
    SUPPORT = "support"
    QUERY = "query"


def panel_feature_evidence_bundle_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelFeatureEvidenceBundleError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PanelFeatureEvidenceBundleError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise PanelFeatureEvidenceBundleError(f"{label} must be a sha256: address")
    return value


def _panel_id(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 1024
    ):
        raise PanelFeatureEvidenceBundleError("panel ID is not bounded canonical text")
    return value


def _panel_bytes(value: object) -> bytes:
    if (
        type(value) is not bytes
        or len(value) <= len(_PNG_SIGNATURE)
        or not value.startswith(_PNG_SIGNATURE)
    ):
        raise PanelFeatureEvidenceBundleError("panel evidence needs exact PNG bytes")
    return value


def _phase_index(phase: PanelFeatureEvidencePhase, value: object) -> int:
    maximum = (
        SUPPORT_PANEL_COUNT
        if phase is PanelFeatureEvidencePhase.SUPPORT
        else QUERY_PANEL_COUNT
    )
    if type(value) is not int or value not in range(maximum):
        raise PanelFeatureEvidenceBundleError("panel phase index differs")
    return value


def _panel_content(value: "PanelFeatureEvidencePanel") -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_EVIDENCE_PANEL_SCHEMA,
        "phase": value.phase.value,
        "phase_index": value.phase_index,
        "panel_id": value.panel_id,
        "panel_png_base64": base64.b64encode(value.panel_png).decode("ascii"),
        "panel_png_digest": value.panel_png_digest,
        "panel_png_byte_count": len(value.panel_png),
        "owner_artifact": (
            None if value.owner_artifact is None else value.owner_artifact.to_data()
        ),
        "axis_artifacts": [item.to_data() for item in value.axis_artifacts],
        "batched_axis_artifact": (
            None
            if value.batched_axis_artifact is None
            else value.batched_axis_artifact.to_data()
        ),
        "observer_call_mode": (
            "batched" if value.batched_axis_artifact is not None else "individual"
        ),
        "axis_artifact_order": "axis-digest-ascending",
        "observation_set": value.observation_set.to_data(),
        "observation_set_digest": value.observation_set.observation_set_digest,
        "phase_or_class_label_passed_to_observer": False,
        "formula_or_selected_predicate_passed_to_observer": False,
        "observer_presentation_name": "panel.png",
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureEvidencePanel:
    """One exact panel and every full observer call artifact used for it."""

    phase: PanelFeatureEvidencePhase
    phase_index: int
    panel_id: str
    panel_png: bytes
    owner_artifact: TypedOwnerCodexArtifact | None
    axis_artifacts: tuple[TypedAxisCodexArtifact, ...]
    batched_axis_artifact: TypedBatchedAxisCodexArtifact | None
    observation_set: PanelFeatureObservationSet
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.phase) is not PanelFeatureEvidencePhase:
            raise TypeError("panel evidence phase must be exact")
        _phase_index(self.phase, self.phase_index)
        _panel_id(self.panel_id)
        _panel_bytes(self.panel_png)
        if self.owner_artifact is not None and type(
            self.owner_artifact
        ) is not TypedOwnerCodexArtifact:
            raise TypeError("panel owner evidence must be a full typed artifact")
        if type(self.axis_artifacts) is not tuple or any(
            type(item) is not TypedAxisCodexArtifact for item in self.axis_artifacts
        ):
            raise TypeError("panel axis evidence must be a typed artifact tuple")
        if self.batched_axis_artifact is not None and type(
            self.batched_axis_artifact
        ) is not TypedBatchedAxisCodexArtifact:
            raise TypeError("panel batched evidence must be a full typed artifact")
        if bool(self.axis_artifacts) == (self.batched_axis_artifact is not None):
            raise PanelFeatureEvidenceBundleError(
                "panel needs exactly one individual or batched axis artifact path"
            )
        if self.batched_axis_artifact is not None and self.owner_artifact is not None:
            raise PanelFeatureEvidenceBundleError(
                "whole-panel batched evidence cannot use an owner artifact"
            )
        axes = tuple(item.observation.axis.axis_digest for item in self.axis_artifacts)
        if axes != tuple(sorted(axes)) or len(axes) != len(set(axes)):
            raise PanelFeatureEvidenceBundleError(
                "panel axis artifacts must be unique and axis-digest sorted"
            )
        if type(self.observation_set) is not PanelFeatureObservationSet:
            raise TypeError("panel evidence needs an exact observation set")
        _digest(self.record_digest, "panel evidence record digest")
        if self.record_digest != canonical_digest(_panel_content(self)):
            raise PanelFeatureEvidenceBundleError("panel evidence digest differs")

    @property
    def panel_png_digest(self) -> str:
        return hashlib.sha256(self.panel_png).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        phase: PanelFeatureEvidencePhase,
        phase_index: int,
        panel_id: str,
        panel_png: bytes,
        owner_artifact: TypedOwnerCodexArtifact | None,
        axis_artifacts: Sequence[TypedAxisCodexArtifact],
        observation_set: PanelFeatureObservationSet,
        batched_axis_artifact: TypedBatchedAxisCodexArtifact | None = None,
    ) -> "PanelFeatureEvidencePanel":
        if isinstance(axis_artifacts, (bytes, str, Mapping)):
            raise TypeError("axis artifacts must be an ordered sequence")
        axes = tuple(axis_artifacts)
        values = {
            "phase": phase,
            "phase_index": phase_index,
            "panel_id": panel_id,
            "panel_png": panel_png,
            "owner_artifact": owner_artifact,
            "axis_artifacts": axes,
            "batched_axis_artifact": batched_axis_artifact,
            "observation_set": observation_set,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest=canonical_digest(_panel_content(provisional)),
        )

    @classmethod
    def derive_from_full_artifacts(
        cls,
        *,
        phase: PanelFeatureEvidencePhase,
        phase_index: int,
        panel_id: str,
        panel_png: bytes,
        owner_artifact: TypedOwnerCodexArtifact | None,
        axis_artifacts: Sequence[TypedAxisCodexArtifact] = (),
        batched_axis_artifact: TypedBatchedAxisCodexArtifact | None = None,
    ) -> "PanelFeatureEvidencePanel":
        """Derive the compact observation set; accept no caller-supplied cells."""

        if isinstance(axis_artifacts, (bytes, str, Mapping)):
            raise TypeError("axis artifacts must be an ordered sequence")
        raw_axes = tuple(axis_artifacts)
        if any(
            type(item) is not TypedAxisCodexArtifact for item in raw_axes
        ):
            raise TypeError("observation derivation needs full typed axis artifacts")
        if bool(raw_axes) == (batched_axis_artifact is not None):
            raise PanelFeatureEvidenceBundleError(
                "observation derivation needs exactly one full individual or batched path"
            )
        if batched_axis_artifact is not None:
            if type(batched_axis_artifact) is not TypedBatchedAxisCodexArtifact:
                raise TypeError(
                    "observation derivation needs a full typed batched artifact"
                )
            return cls.create(
                phase=phase,
                phase_index=phase_index,
                panel_id=panel_id,
                panel_png=panel_png,
                owner_artifact=owner_artifact,
                axis_artifacts=(),
                batched_axis_artifact=batched_axis_artifact,
                observation_set=batched_axis_artifact.observation_set,
            )
        axes = tuple(
            sorted(
                raw_axes,
                key=lambda item: item.observation.axis.axis_digest,
            )
        )
        first = axes[0]
        observation = PanelFeatureObservationSet(
            hashlib.sha256(_panel_bytes(panel_png)).hexdigest(),
            first.observer_contract_digest,
            first.measurement_protocol_digest,
            tuple(item.observation for item in axes),
        )
        return cls.create(
            phase=phase,
            phase_index=phase_index,
            panel_id=panel_id,
            panel_png=panel_png,
            owner_artifact=owner_artifact,
            axis_artifacts=axes,
            observation_set=observation,
        )

    @classmethod
    def derive_from_batched_artifact(
        cls,
        *,
        phase: PanelFeatureEvidencePhase,
        phase_index: int,
        panel_id: str,
        panel_png: bytes,
        batched_axis_artifact: TypedBatchedAxisCodexArtifact,
    ) -> "PanelFeatureEvidencePanel":
        """Derive all compact cells from one complete whole-panel batch call."""

        return cls.derive_from_full_artifacts(
            phase=phase,
            phase_index=phase_index,
            panel_id=panel_id,
            panel_png=panel_png,
            owner_artifact=None,
            batched_axis_artifact=batched_axis_artifact,
        )

    def to_data(self) -> dict[str, object]:
        return {**_panel_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureEvidencePanel":
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
                "owner_artifact",
                "axis_artifacts",
                "batched_axis_artifact",
                "observer_call_mode",
                "axis_artifact_order",
                "observation_set",
                "observation_set_digest",
                "phase_or_class_label_passed_to_observer",
                "formula_or_selected_predicate_passed_to_observer",
                "observer_presentation_name",
                "record_digest",
            },
            "panel feature evidence row",
        )
        if (
            raw["schema"] != PANEL_FEATURE_EVIDENCE_PANEL_SCHEMA
            or raw["observer_call_mode"] not in ("individual", "batched")
            or raw["axis_artifact_order"] != "axis-digest-ascending"
            or raw["phase_or_class_label_passed_to_observer"] is not False
            or raw["formula_or_selected_predicate_passed_to_observer"] is not False
            or raw["observer_presentation_name"] != "panel.png"
            or type(raw["panel_png_base64"]) is not str
            or type(raw["axis_artifacts"]) is not list
        ):
            raise PanelFeatureEvidenceBundleError("panel evidence policy differs")
        try:
            phase = PanelFeatureEvidencePhase(raw["phase"])
            panel = base64.b64decode(raw["panel_png_base64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise PanelFeatureEvidenceBundleError("panel evidence encoding differs") from exc
        if (
            base64.b64encode(panel).decode("ascii") != raw["panel_png_base64"]
            or hashlib.sha256(panel).hexdigest() != raw["panel_png_digest"]
            or len(panel) != raw["panel_png_byte_count"]
        ):
            raise PanelFeatureEvidenceBundleError("panel byte commitment differs")
        try:
            result = cls(
                phase,
                raw["phase_index"],
                raw["panel_id"],
                panel,
                (
                    None
                    if raw["owner_artifact"] is None
                    else TypedOwnerCodexArtifact.from_data(raw["owner_artifact"])
                ),
                tuple(
                    TypedAxisCodexArtifact.from_data(item)
                    for item in raw["axis_artifacts"]
                ),
                (
                    None
                    if raw["batched_axis_artifact"] is None
                    else TypedBatchedAxisCodexArtifact.from_data(
                        raw["batched_axis_artifact"]
                    )
                ),
                PanelFeatureObservationSet.from_data(raw["observation_set"]),
                raw["record_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelFeatureEvidenceBundleError):
                raise
            raise PanelFeatureEvidenceBundleError(
                "panel evidence typed artifact differs"
            ) from exc
        if (
            raw["observation_set_digest"]
            != result.observation_set.observation_set_digest
            or result.to_data() != dict(raw)
        ):
            raise PanelFeatureEvidenceBundleError("panel evidence is not canonical")
        return result


def _panel_order_key(
    value: PanelFeatureEvidencePanel,
) -> tuple[int, int]:
    return (
        0 if value.phase is PanelFeatureEvidencePhase.SUPPORT else 1,
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
        raise PanelFeatureEvidenceBundleError(
            "proposer result cannot be replayed from its full artifact"
        ) from exc


def _vocabulary_axes(
    result: PanelFeatureProposerResult,
) -> tuple[str, ...]:
    # A typed language/nomination gap is itself a valid receipted outcome.
    # The fixed observer catalog is independent of proposer candidates, so its
    # full panel evidence must remain archivable even when no vocabulary was
    # admitted.  Synthesis still fails closed later; custody must not erase the
    # evidence explaining why.
    if result.observer_vocabulary is None:
        if result.nominations:
            raise PanelFeatureEvidenceBundleError(
                "proposer nominations lack their observer vocabulary"
            )
        return ()
    if not result.observer_vocabulary.specs:
        raise PanelFeatureEvidenceBundleError("observer vocabulary is empty")
    return tuple(
        sorted(
            {
                FeatureAxis.for_spec(spec).axis_digest
                for spec in result.observer_vocabulary.specs
            }
        )
    )


def _receipt_digests(
    proposer: TypedProposerCodexCallArtifact,
    panels: Sequence[PanelFeatureEvidencePanel],
) -> tuple[str, ...]:
    values: list[str] = [proposer.codex_receipt.receipt_digest]
    for panel in panels:
        if panel.owner_artifact is not None:
            values.append(panel.owner_artifact.codex_receipt.receipt_digest)
        values.extend(item.codex_receipt.receipt_digest for item in panel.axis_artifacts)
        if panel.batched_axis_artifact is not None:
            values.append(panel.batched_axis_artifact.codex_receipt.receipt_digest)
    return tuple(values)


def _owner_call_count(panels: Sequence[PanelFeatureEvidencePanel]) -> int:
    return sum(item.owner_artifact is not None for item in panels)


def _individual_axis_call_count(
    panels: Sequence[PanelFeatureEvidencePanel],
) -> int:
    return sum(len(item.axis_artifacts) for item in panels)


def _batched_axis_call_count(panels: Sequence[PanelFeatureEvidencePanel]) -> int:
    return sum(item.batched_axis_artifact is not None for item in panels)


def _axis_call_count(panels: Sequence[PanelFeatureEvidencePanel]) -> int:
    """Count physical axis-observer calls, not logical axis observations."""

    return _individual_axis_call_count(panels) + _batched_axis_call_count(panels)


def _panel_observer_runtime(
    panel: PanelFeatureEvidencePanel,
) -> TypedCodexRuntimeBinding:
    if panel.batched_axis_artifact is not None:
        return panel.batched_axis_artifact.runtime
    if panel.axis_artifacts:
        return panel.axis_artifacts[0].runtime
    raise PanelFeatureEvidenceBundleError(
        "evidence panel has no full axis observer artifact"
    )


def _panel_axis_digests(panel: PanelFeatureEvidencePanel) -> tuple[str, ...]:
    if panel.batched_axis_artifact is not None:
        return tuple(
            item.axis.axis_digest
            for item in panel.batched_axis_artifact.observation_set.axis_observations
        )
    return tuple(
        item.observation.axis.axis_digest for item in panel.axis_artifacts
    )


def _bundle_content(value: "PanelFeatureEvidenceBundle") -> dict[str, object]:
    receipts = _receipt_digests(value.proposer_artifact, value.panels)
    observer_axes = [item.to_data() for item in value.observer_axes]
    support_count = sum(
        item.phase is PanelFeatureEvidencePhase.SUPPORT for item in value.panels
    )
    query_count = len(value.panels) - support_count
    owner_count = _owner_call_count(value.panels)
    individual_axis_count = _individual_axis_call_count(value.panels)
    batched_axis_count = _batched_axis_call_count(value.panels)
    axis_count = _axis_call_count(value.panels)
    return {
        "schema": PANEL_FEATURE_EVIDENCE_BUNDLE_SCHEMA,
        "protocol_id": PANEL_FEATURE_EVIDENCE_PROTOCOL_ID,
        "protocol_source_digest": panel_feature_evidence_bundle_source_digest(),
        "task_context_digest": value.proposer_artifact.task_context_digest,
        "proposer_artifact": value.proposer_artifact.to_data(),
        "proposer_artifact_digest": value.proposer_artifact.artifact_digest,
        "proposer_result": value.proposer_result.to_data(),
        "proposer_result_digest": value.proposer_result.result_digest,
        "proposer_runtime": value.proposer_artifact.runtime.to_data(),
        "proposer_runtime_digest": value.proposer_artifact.runtime.runtime_digest,
        "proposer_transport_contract_digest": (
            value.proposer_artifact.transport_contract_digest
        ),
        "observer_runtime": value.observer_runtime.to_data(),
        "observer_runtime_digest": value.observer_runtime.runtime_digest,
        "observer_contract_digest": typed_codex_observer_contract_digest(
            value.observer_runtime
        ),
        "measurement_protocol_digest": typed_measurement_protocol_digest(
            value.observer_runtime
        ),
        "observer_axes": observer_axes,
        "observer_axis_order": "axis-digest-ascending",
        "observer_axis_catalog_digest": canonical_digest(observer_axes),
        "panels": [item.to_data() for item in value.panels],
        "panel_order": "support-index-then-query-index",
        "support_panel_count": support_count,
        "query_panel_count": query_count,
        "query_phase_complete": query_count == QUERY_PANEL_COUNT,
        "proposer_model_call_count": 1,
        "owner_model_call_count": owner_count,
        "individual_axis_model_call_count": individual_axis_count,
        "batched_axis_model_call_count": batched_axis_count,
        "axis_model_call_count": axis_count,
        "live_model_call_count": 1 + owner_count + axis_count,
        "physical_receipt_digests": list(receipts),
        "model_call_count_derived_from_unique_full_receipts": True,
        "opaque_receipt_digests_accepted": False,
        "all_model_payloads_and_full_receipts_retained": True,
        "observer_artifacts_receive_phase_or_class_label": False,
        "observer_artifacts_receive_formula_or_selected_predicate": False,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "cold_replay_model_call_count": 0,
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureEvidenceBundle:
    """Complete content-addressed custody for one proposer and panel episode."""

    proposer_artifact: TypedProposerCodexCallArtifact
    proposer_result: PanelFeatureProposerResult
    observer_runtime: TypedCodexRuntimeBinding
    observer_axes: tuple[FeatureAxis, ...]
    panels: tuple[PanelFeatureEvidencePanel, ...]
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.proposer_artifact) is not TypedProposerCodexCallArtifact:
            raise TypeError("evidence bundle needs a full typed proposer artifact")
        if type(self.proposer_result) is not PanelFeatureProposerResult:
            raise TypeError("evidence bundle needs an exact proposer result")
        if type(self.observer_runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("evidence bundle needs an exact observer runtime")
        if (
            type(self.observer_axes) is not tuple
            or not self.observer_axes
            or any(type(item) is not FeatureAxis for item in self.observer_axes)
        ):
            raise TypeError("evidence bundle needs a nonempty typed observer-axis tuple")
        observer_axis_digests = tuple(
            item.axis_digest for item in self.observer_axes
        )
        if (
            observer_axis_digests != tuple(sorted(observer_axis_digests))
            or len(observer_axis_digests) != len(set(observer_axis_digests))
        ):
            raise PanelFeatureEvidenceBundleError(
                "observer axes must be unique and axis-digest sorted"
            )
        if (
            type(self.panels) is not tuple
            or any(type(item) is not PanelFeatureEvidencePanel for item in self.panels)
            or self.panels != tuple(sorted(self.panels, key=_panel_order_key))
        ):
            raise PanelFeatureEvidenceBundleError(
                "evidence panels must be canonical support-then-query rows"
            )
        _digest(self.record_digest, "panel feature evidence bundle digest")
        self._verify_full_custody(cold_replay=False)
        if self.record_digest != canonical_digest(_bundle_content(self)):
            raise PanelFeatureEvidenceBundleError("evidence bundle digest differs")

    @property
    def bundle_address(self) -> str:
        return "sha256:" + self.record_digest

    @property
    def live_model_call_count(self) -> int:
        return 1 + _owner_call_count(self.panels) + _axis_call_count(self.panels)

    @property
    def physical_receipt_digests(self) -> tuple[str, ...]:
        return _receipt_digests(self.proposer_artifact, self.panels)

    def panels_for_phase(
        self, phase: PanelFeatureEvidencePhase
    ) -> tuple[PanelFeatureEvidencePanel, ...]:
        if type(phase) is not PanelFeatureEvidencePhase:
            raise TypeError("evidence phase must be exact")
        return tuple(item for item in self.panels if item.phase is phase)

    def observation_sets_for_phase(
        self, phase: PanelFeatureEvidencePhase
    ) -> tuple[PanelFeatureObservationSet, ...]:
        return tuple(item.observation_set for item in self.panels_for_phase(phase))

    def _verify_full_custody(self, *, cold_replay: bool) -> None:
        support = tuple(
            item
            for item in self.panels
            if item.phase is PanelFeatureEvidencePhase.SUPPORT
        )
        query = tuple(
            item
            for item in self.panels
            if item.phase is PanelFeatureEvidencePhase.QUERY
        )
        if (
            len(support) != SUPPORT_PANEL_COUNT
            or tuple(item.phase_index for item in support)
            != tuple(range(SUPPORT_PANEL_COUNT))
            or len(query) not in (0, QUERY_PANEL_COUNT)
            or tuple(item.phase_index for item in query) != tuple(range(len(query)))
            or len({item.panel_id for item in self.panels}) != len(self.panels)
        ):
            raise PanelFeatureEvidenceBundleError(
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
            raise PanelFeatureEvidenceBundleError(
                "full proposer artifact does not bind the exact support bytes"
            )
        replayed_result = _rebuild_proposer_result(self.proposer_artifact)
        if replayed_result != self.proposer_result:
            raise PanelFeatureEvidenceBundleError(
                "compact proposer result differs from the full proposer artifact"
            )
        required_axes = tuple(item.axis_digest for item in self.observer_axes)
        vocabulary_axes = _vocabulary_axes(replayed_result)
        if not set(vocabulary_axes).issubset(required_axes):
            raise PanelFeatureEvidenceBundleError(
                "observer-axis catalog omits an axis used by the proposer vocabulary"
            )
        observer_contract = typed_codex_observer_contract_digest(
            self.observer_runtime
        )
        measurement_protocol = typed_measurement_protocol_digest(
            self.observer_runtime
        )
        for panel in self.panels:
            actual_axes = _panel_axis_digests(panel)
            if actual_axes != required_axes:
                raise PanelFeatureEvidenceBundleError(
                    "panel has missing, extra, or duplicated axis call artifacts"
                )
            local_axes = tuple(
                item
                for item in panel.axis_artifacts
                if item.source_kind == "receipted_owner_inventory"
            )
            if bool(local_axes) != (panel.owner_artifact is not None):
                raise PanelFeatureEvidenceBundleError(
                    "owner artifact is missing when used or extra when unused"
                )
            if (
                panel.batched_axis_artifact is not None
                and panel.owner_artifact is not None
            ):
                raise PanelFeatureEvidenceBundleError(
                    "whole-panel batch has an extraneous owner artifact"
                )
            if panel.owner_artifact is not None:
                if (
                    panel.owner_artifact.runtime != self.observer_runtime
                    or panel.owner_artifact.panel_png_digest != panel.panel_png_digest
                    or panel.owner_artifact.panel_png_byte_count != len(panel.panel_png)
                ):
                    raise PanelFeatureEvidenceBundleError(
                        "owner artifact runtime or exact panel binding differs"
                    )
            if panel.batched_axis_artifact is not None:
                batch = panel.batched_axis_artifact
                if (
                    batch.runtime != self.observer_runtime
                    or batch.observer_contract_digest != observer_contract
                    or batch.measurement_protocol_digest != measurement_protocol
                    or batch.panel_png_digest != panel.panel_png_digest
                    or batch.panel_png_byte_count != len(panel.panel_png)
                    or tuple(item.axis_digest for item in batch.request.axes)
                    != required_axes
                ):
                    raise PanelFeatureEvidenceBundleError(
                        "batched axis runtime, catalog, protocol, or exact panel binding differs"
                    )
                expected_observation = batch.observation_set
            else:
                replayed_axes: list[TypedAxisCodexArtifact] = []
                for artifact in panel.axis_artifacts:
                    if (
                        artifact.runtime != self.observer_runtime
                        or artifact.observer_contract_digest != observer_contract
                        or artifact.measurement_protocol_digest != measurement_protocol
                        or artifact.panel_png_digest != panel.panel_png_digest
                        or artifact.panel_png_byte_count != len(panel.panel_png)
                    ):
                        raise PanelFeatureEvidenceBundleError(
                            "axis runtime, protocol, or exact panel binding differs"
                        )
                    replayed_axes.append(artifact)
                expected_observation = PanelFeatureObservationSet(
                    panel.panel_png_digest,
                    observer_contract,
                    measurement_protocol,
                    tuple(item.observation for item in replayed_axes),
                )
            if panel.observation_set != expected_observation:
                raise PanelFeatureEvidenceBundleError(
                    "observation set is not exactly derived from full axis artifacts"
                )
        receipts = self.physical_receipt_digests
        if len(receipts) != len(set(receipts)):
            raise PanelFeatureEvidenceBundleError(
                "one physical Codex receipt is duplicated across call artifacts"
            )
        if self.proposer_result.receipt_digest != self.proposer_artifact.artifact_digest:
            raise PanelFeatureEvidenceBundleError(
                "opaque proposer receipt digest is not a retained artifact address"
            )
        if cold_replay:
            self._cold_verify_artifacts(support)

    def _cold_verify_artifacts(
        self, support: Sequence[PanelFeatureEvidencePanel]
    ) -> None:
        """Run independent receipt replays concurrently; invoke no model boundary."""

        jobs: list[
            tuple[str, Callable[..., object], tuple[object, ...], dict[str, object]]
        ] = [
            (
                "proposer",
                verify_typed_proposer_codex_artifact,
                (self.proposer_artifact, tuple(item.panel_png for item in support)),
                {"expected_artifact_digest": self.proposer_artifact.artifact_digest},
            )
        ]
        for panel in self.panels:
            if panel.owner_artifact is not None:
                jobs.append(
                    (
                        f"owner:{panel.panel_id}",
                        verify_typed_owner_codex_artifact,
                        (panel.owner_artifact, panel.panel_png),
                        {
                            "expected_artifact_digest": (
                                panel.owner_artifact.artifact_digest
                            )
                        },
                    )
                )
            for artifact in panel.axis_artifacts:
                owner = (
                    panel.owner_artifact
                    if artifact.source_kind == "receipted_owner_inventory"
                    else None
                )
                jobs.append(
                    (
                        f"axis:{panel.panel_id}:{artifact.observation.axis.axis_digest}",
                        verify_typed_axis_codex_artifact,
                        (artifact, panel.panel_png),
                        {
                            "owner_artifact": owner,
                            "expected_artifact_digest": artifact.artifact_digest,
                        },
                    )
                )
            if panel.batched_axis_artifact is not None:
                jobs.append(
                    (
                        f"batched-axis:{panel.panel_id}",
                        verify_typed_batched_axis_codex_artifact,
                        (panel.batched_axis_artifact, panel.panel_png),
                        {
                            "expected_artifact_digest": (
                                panel.batched_axis_artifact.artifact_digest
                            )
                        },
                    )
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
                    raise PanelFeatureEvidenceBundleError(
                        f"full {label} receipt fails cold replay"
                    ) from exc

    @classmethod
    def create(
        cls,
        *,
        proposer_artifact: TypedProposerCodexCallArtifact,
        proposer_result: PanelFeatureProposerResult,
        observer_axes: Sequence[FeatureAxis],
        panels: Sequence[PanelFeatureEvidencePanel],
    ) -> "PanelFeatureEvidenceBundle":
        if isinstance(observer_axes, (bytes, str, Mapping)):
            raise TypeError("observer axes must be an ordered sequence")
        if isinstance(panels, (bytes, str, Mapping)):
            raise TypeError("evidence panels must be an ordered sequence")
        rows = tuple(panels)
        if not rows:
            raise PanelFeatureEvidenceBundleError(
                "evidence bundle cannot derive an observer runtime"
            )
        values = {
            "proposer_artifact": proposer_artifact,
            "proposer_result": proposer_result,
            "observer_runtime": _panel_observer_runtime(rows[0]),
            "observer_axes": tuple(observer_axes),
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
    def from_data(cls, value: object) -> "PanelFeatureEvidenceBundle":
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
                "proposer_runtime",
                "proposer_runtime_digest",
                "proposer_transport_contract_digest",
                "observer_runtime",
                "observer_runtime_digest",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "observer_axes",
                "observer_axis_order",
                "observer_axis_catalog_digest",
                "panels",
                "panel_order",
                "support_panel_count",
                "query_panel_count",
                "query_phase_complete",
                "proposer_model_call_count",
                "owner_model_call_count",
                "individual_axis_model_call_count",
                "batched_axis_model_call_count",
                "axis_model_call_count",
                "live_model_call_count",
                "physical_receipt_digests",
                "model_call_count_derived_from_unique_full_receipts",
                "opaque_receipt_digests_accepted",
                "all_model_payloads_and_full_receipts_retained",
                "observer_artifacts_receive_phase_or_class_label",
                "observer_artifacts_receive_formula_or_selected_predicate",
                "python_is_canonical_authority",
                "lean_present",
                "lean_required",
                "cold_replay_model_call_count",
                "record_digest",
                "bundle_address",
            },
            "panel feature evidence bundle",
        )
        if (
            raw["schema"] != PANEL_FEATURE_EVIDENCE_BUNDLE_SCHEMA
            or raw["protocol_id"] != PANEL_FEATURE_EVIDENCE_PROTOCOL_ID
            or raw["protocol_source_digest"]
            != panel_feature_evidence_bundle_source_digest()
            or raw["observer_axis_order"] != "axis-digest-ascending"
            or type(raw["observer_axes"]) is not list
            or raw["panel_order"] != "support-index-then-query-index"
            or raw["proposer_model_call_count"] != 1
            or raw["model_call_count_derived_from_unique_full_receipts"] is not True
            or raw["opaque_receipt_digests_accepted"] is not False
            or raw["all_model_payloads_and_full_receipts_retained"] is not True
            or raw["observer_artifacts_receive_phase_or_class_label"] is not False
            or raw["observer_artifacts_receive_formula_or_selected_predicate"]
            is not False
            or raw["python_is_canonical_authority"] is not True
            or raw["lean_present"] is not False
            or raw["lean_required"] is not False
            or raw["cold_replay_model_call_count"] != 0
            or type(raw["panels"]) is not list
            or type(raw["physical_receipt_digests"]) is not list
        ):
            raise PanelFeatureEvidenceBundleError("evidence bundle policy differs")
        try:
            proposer = TypedProposerCodexCallArtifact.from_data(
                raw["proposer_artifact"]
            )
            replayed_result = _rebuild_proposer_result(proposer)
            if replayed_result.to_data() != raw["proposer_result"]:
                raise PanelFeatureEvidenceBundleError(
                    "archived proposer result is not derived from its full artifact"
                )
            result = cls(
                proposer,
                replayed_result,
                TypedCodexRuntimeBinding.from_data(raw["observer_runtime"]),
                tuple(FeatureAxis.from_data(item) for item in raw["observer_axes"]),
                tuple(PanelFeatureEvidencePanel.from_data(item) for item in raw["panels"]),
                raw["record_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelFeatureEvidenceBundleError):
                raise
            raise PanelFeatureEvidenceBundleError(
                "evidence bundle typed value differs"
            ) from exc
        if result.to_data() != dict(raw):
            raise PanelFeatureEvidenceBundleError(
                "evidence bundle derived links or call counts differ"
            )
        return result


def cold_replay_panel_feature_evidence_bundle(
    bundle: PanelFeatureEvidenceBundle,
    *,
    expected_bundle_address: str,
) -> PanelFeatureEvidenceBundle:
    """Replay every parser, PNG binding, and full receipt with zero model calls."""

    if type(bundle) is not PanelFeatureEvidenceBundle:
        raise TypeError("cold replay needs an exact PanelFeatureEvidenceBundle")
    expected = _address(expected_bundle_address, "expected evidence bundle address")
    restored = PanelFeatureEvidenceBundle.from_data(bundle.to_data())
    if restored.bundle_address != expected:
        raise PanelFeatureEvidenceBundleError(
            "evidence bundle differs from the external content commitment"
        )
    restored._verify_full_custody(cold_replay=True)
    return restored


__all__ = (
    "PANEL_FEATURE_EVIDENCE_BUNDLE_SCHEMA",
    "PANEL_FEATURE_EVIDENCE_PROTOCOL_ID",
    "PanelFeatureEvidenceBundle",
    "PanelFeatureEvidenceBundleError",
    "PanelFeatureEvidencePanel",
    "PanelFeatureEvidencePhase",
    "cold_replay_panel_feature_evidence_bundle",
    "panel_feature_evidence_bundle_source_digest",
)
