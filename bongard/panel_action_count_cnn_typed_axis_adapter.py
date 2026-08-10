"""Custody-preserving CNN prediction adapter for the typed-axis core.

This module does not open images, run a model, read labels, or call Lean.  It
accepts an already-frozen twelve-row support prediction batch and an exact
population-scoped calibration release.  It rechecks logits, probabilities,
joint-q class sets, panel roles, and panel-byte bindings before producing the
``TypedSupportMatrix`` consumed by the deterministic 1,366-formula inventory.

The generic fresh-V3 release is deliberately not a distribution-shift grant.
In particular, it can never authorize the convex/four-straight-lines target
family.  Only the separately preregistered same-family release can populate
calibrated cells for ``hd_convex-has_four-straight-lines_0000``.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.panel_typed_axis_slate_v2 import (
    AXES,
    Axis,
    SupportSide,
    TypedAxisCell,
    TypedSupportMatrix,
    TypedSupportRow,
    typed_axis_slate_algorithm_digest,
    typed_axis_slate_source_digest,
)


PROTOCOL_SCHEMA = "gkm.bongard-cnn-typed-axis-observer-protocol.v1"
GRANT_SCHEMA = "gkm.bongard-cnn-typed-axis-population-grant.v1"
PANEL_SCHEMA = "gkm.bongard-cnn-typed-axis-support-prediction.v1"
BATCH_SCHEMA = "gkm.bongard-cnn-typed-axis-support-batch.v1"
ARTIFACT_SCHEMA = "gkm.bongard-cnn-typed-axis-matrix-artifact.v1"
ALGORITHM_SCHEMA = "gkm.bongard-cnn-typed-axis-adapter-algorithm.v1"
ALGORITHM_ID = "bongard.cnn-prediction-to-typed-axis/support-only-python-v1"

FINAL_TRAINER_SOURCE_SHA256 = (
    "sha256:2706faf07052e580331346ea209c60bc59987366be53f6a729570f0d2cbc9e6a"
)
POSTPREDICTION_SOURCE_SHA256 = (
    "sha256:f2b9adc70b3e16794531358e8e80613bb50546752739c2b2ccd8953019850354"
)
ARCHITECTURE_ID = "shared-cnn-16-32-64-96-three-head/v1"
FIT_RESULT_RECORD_SCHEMA = "gkm.bongard-action-count-catalog-cnn-fit-result.v2"
FIT_PRECOMMIT_RECORD_SCHEMA = (
    "gkm.bongard-action-count-catalog-cnn-fit-pixel-precommit.v2"
)
POSTPREDICTION_CONTRACT_DIGEST = (
    "sha256:cc04dfd8ca683841103d250dfb75679bf36d79b513d15a017c66cd80c2640504"
)
STRAIGHT_CLASS_ORDER = tuple(range(10))
ARC_CLASS_ORDER = tuple(range(10))
CATALOG_CLASS_ORDER = ("catalog_unresolved", "nonconvex", "convex")
SUPPORT_ORDINALS = (0, 1, 2, 3, 5, 6)

GENERIC_V3_PLAN_RECORD_DIGEST = (
    "sha256:bb4524a0958cd21f2d4d49bc6a9caa964ccb96c67fbf7c6192185f7b2f363dcb"
)
GENERIC_V3_PLAN_SOURCE_SHA256 = (
    "sha256:71c68771b356658843c3d848cdeea0ba7f2d96fffacd1816ef72934214b055d0"
)
SAME_FAMILY_PREREG_RECORD_DIGEST = (
    "sha256:77a8aba2868ab3369a40befca470ee686eb998543dcae27d4f4b1f68a7df0b5a"
)
SAME_FAMILY_PREREG_SOURCE_SHA256 = (
    "sha256:5806422f2186a412ad4eba68de0deb4ab42133713ab7f3e3c88ef0cf5ea44c9c"
)
TARGET_SEMANTIC_KEY = "hd_convex-has_four-straight-lines"
TARGET_TASK_ID = TARGET_SEMANTIC_KEY + "_0000"
SAME_FAMILY_CALIBRATION_TASK_IDS = tuple(
    f"{TARGET_SEMANTIC_KEY}_{index:04d}" for index in range(2, 18)
)

GENERIC_GRANT_RECORD_SCHEMA = (
    "gkm.bongard-action-count-cnn-calibration-grant.v3"
)
GENERIC_RELEASE_RECORD_SCHEMA = (
    "gkm.bongard-action-count-cnn-evaluation-result.v3"
)
SAME_FAMILY_GRANT_RECORD_SCHEMA = (
    "gkm.bongard-convex-four-lines-same-family-calibration-grant.v1"
)
SAME_FAMILY_RELEASE_RECORD_SCHEMA = (
    "gkm.bongard-convex-four-lines-same-family-efficiency-release.v1"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TASK = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,191}_[0-9]{4}\Z")
_PANEL = re.compile(
    r"hd/(?P<task>[A-Za-z0-9][A-Za-z0-9_-]{0,191}_[0-9]{4})/"
    r"(?P<folder>[01])/(?P<ordinal>[0-6])\.png\Z"
)


class CNNToTypedAxisError(ValueError):
    """CNN custody, scope, support bytes, or deterministic replay differs."""


class PopulationScope(str, Enum):
    GENERIC_FRESH_V3 = "generic_fresh_v3"
    SAME_FAMILY_CONVEX_FOUR_LINES = "same_family_convex_four_lines"


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise CNNToTypedAxisError(f"{label} must be a SHA-256 address")
    return value


def _task_id(value: object, label: str = "task ID") -> str:
    if type(value) is not str or _TASK.fullmatch(value) is None:
        raise CNNToTypedAxisError(f"{label} is not a bounded task ID")
    return value


def _semantic_key(task_id: str) -> str:
    _task_id(task_id)
    return task_id[:-5]


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise CNNToTypedAxisError(f"{label} fields differ")
    return value


def _canonical_match(rebuilt: object, supplied: Mapping[str, Any], label: str) -> None:
    try:
        differs = canonical_json(rebuilt) != canonical_json(dict(supplied))
    except (TypeError, ValueError) as exc:
        raise CNNToTypedAxisError(f"{label} is not canonical JSON") from exc
    if differs:
        raise CNNToTypedAxisError(f"{label} is not canonical")


def _sealed_record(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise CNNToTypedAxisError(f"{label} is not an object")
    body = dict(value)
    found = body.pop("record_digest", None)
    if found != "sha256:" + canonical_digest(body):
        raise CNNToTypedAxisError(f"{label} record digest differs")
    return value


def _finite_probability_vector(
    values: Sequence[float], size: int, label: str
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)) or len(values) != size:
        raise CNNToTypedAxisError(f"{label} cardinality differs")
    checked: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CNNToTypedAxisError(f"{label} contains a non-number")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise CNNToTypedAxisError(f"{label} leaves [0,1]")
        checked.append(number)
    if not math.isclose(sum(checked), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise CNNToTypedAxisError(f"{label} does not sum to one")
    return tuple(checked)


def _finite_logits(values: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)) or len(values) != size:
        raise CNNToTypedAxisError(f"{label} cardinality differs")
    checked: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CNNToTypedAxisError(f"{label} contains a non-number")
        number = float(value)
        if not math.isfinite(number):
            raise CNNToTypedAxisError(f"{label} contains a nonfinite number")
        checked.append(number)
    return tuple(checked)


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    largest = max(logits)
    exponentials = tuple(math.exp(value - largest) for value in logits)
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def _class_set(values: Sequence[int], size: int, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise CNNToTypedAxisError(f"{label} must be a class-index sequence")
    checked = tuple(values)
    if (
        any(type(value) is not int for value in checked)
        or checked != tuple(sorted(set(checked)))
        or any(not 0 <= value < size for value in checked)
    ):
        raise CNNToTypedAxisError(f"{label} is not a canonical class set")
    return checked


def _q(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CNNToTypedAxisError("deployment joint q is not numeric")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 <= checked <= 1.0:
        raise CNNToTypedAxisError("deployment joint q leaves [0,1]")
    return checked


def adapter_source_digest() -> str:
    """Return the authenticated source address of this loaded adapter."""

    return "sha256:" + verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def preprocess_contract_digest() -> str:
    """Bind the exact final trainer preprocessing operation without reading pixels."""

    return "sha256:" + canonical_digest(
        {
            "trainer_source_sha256": FINAL_TRAINER_SOURCE_SHA256,
            "function": "bongard.panel_action_count_cnn_train_command.preprocess_png_bytes",
            "input": "single-frame PNG",
            "grayscale": "Pillow convert L to numpy uint8",
            "ink_threshold": "gray_less_than_250",
            "crop": "inclusive_bbox_of_all_ink",
            "margin": "ceil(0.08*max(crop_height,crop_width))",
            "square_centering": "floor_leftover_to_top_and_left",
            "resize": [96, 96, "Pillow.Resampling.BILINEAR"],
            "output": "contiguous_uint8_255_minus_gray",
        }
    )


def architecture_preprocess_address() -> str:
    return "sha256:" + canonical_digest(
        {
            "architecture_id": ARCHITECTURE_ID,
            "preprocess_contract_digest": preprocess_contract_digest(),
            "straight_class_order": list(STRAIGHT_CLASS_ORDER),
            "arc_class_order": list(ARC_CLASS_ORDER),
            "catalog_class_order": list(CATALOG_CLASS_ORDER),
            "trainer_source_sha256": FINAL_TRAINER_SOURCE_SHA256,
        }
    )


def adapter_algorithm_record() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": ALGORITHM_SCHEMA,
        "algorithm_id": ALGORITHM_ID,
        "implementation_source_sha256": adapter_source_digest(),
        "typed_axis_source_sha256": typed_axis_slate_source_digest(),
        "typed_axis_algorithm_digest": typed_axis_slate_algorithm_digest(),
        "support_shape": "six_primary_then_six_contrast_excluding_ordinal_4",
        "straight_projection": "joint_q_class_indices_to_calibrated_count_set",
        "catalog_projection": {
            "empty": "ERROR",
            "contains_catalog_unresolved": "GAP",
            "class_1": "catalog_nonconvex",
            "class_2": "catalog_convex",
        },
        "other_six_axes": "GAP_under_same_observer_protocol",
        "generic_turning_axis_present": False,
        "generic_fresh_v3_authorizes_target_family": False,
        "query_rows_seen": 0,
        "pixel_reads": 0,
        "model_calls": 0,
        "lean_present": False,
        "python_is_canonical_authority": True,
    }
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


@dataclass(frozen=True, slots=True)
class CNNObserverProtocol:
    inference_source_sha256: str
    fit_precommit_record_digest: str
    fit_precommit_source_sha256: str
    fit_result_record_digest: str
    fit_result_source_sha256: str
    checkpoint_raw_sha256: str
    checkpoint_state_dict_sha256: str
    config_digest: str
    postprediction_contract_digest: str
    fit_validation_gate_record_digest: str
    fit_validation_gate_passed: bool

    def __post_init__(self) -> None:
        for label, value in (
            ("inference source", self.inference_source_sha256),
            ("fit precommit record", self.fit_precommit_record_digest),
            ("fit precommit source", self.fit_precommit_source_sha256),
            ("fit result record", self.fit_result_record_digest),
            ("fit result source", self.fit_result_source_sha256),
            ("checkpoint raw", self.checkpoint_raw_sha256),
            ("checkpoint state", self.checkpoint_state_dict_sha256),
            ("config", self.config_digest),
            ("postprediction contract", self.postprediction_contract_digest),
            ("fit validation gate", self.fit_validation_gate_record_digest),
        ):
            _address(value, label)
        if self.fit_validation_gate_passed is not True:
            raise CNNToTypedAxisError(
                "failed fit cannot instantiate a calibrated observer protocol"
            )
        if self.postprediction_contract_digest != POSTPREDICTION_CONTRACT_DIGEST:
            raise CNNToTypedAxisError("postprediction barrier contract differs")

    @property
    def protocol_address(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PROTOCOL_SCHEMA,
            "architecture_id": ARCHITECTURE_ID,
            "architecture_preprocess_address": architecture_preprocess_address(),
            "preprocess_contract_digest": preprocess_contract_digest(),
            "preprocess_source_sha256": FINAL_TRAINER_SOURCE_SHA256,
            "trainer_source_sha256": FINAL_TRAINER_SOURCE_SHA256,
            "inference_source_sha256": self.inference_source_sha256,
            "fit_precommit_record_digest": self.fit_precommit_record_digest,
            "fit_precommit_source_sha256": self.fit_precommit_source_sha256,
            "fit_result_record_digest": self.fit_result_record_digest,
            "fit_result_source_sha256": self.fit_result_source_sha256,
            "checkpoint_raw_sha256": self.checkpoint_raw_sha256,
            "checkpoint_state_dict_sha256": self.checkpoint_state_dict_sha256,
            "config_digest": self.config_digest,
            "postprediction_contract_digest": self.postprediction_contract_digest,
            "postprediction_source_sha256": POSTPREDICTION_SOURCE_SHA256,
            "fit_validation_gate_record_digest": self.fit_validation_gate_record_digest,
            "fit_validation_gate_passed": self.fit_validation_gate_passed,
            "straight_class_order": list(STRAIGHT_CLASS_ORDER),
            "arc_class_order": list(ARC_CLASS_ORDER),
            "catalog_class_order": list(CATALOG_CLASS_ORDER),
        }

    @classmethod
    def from_data(cls, value: object) -> "CNNObserverProtocol":
        raw = _fields(
            value,
            {
                "schema", "architecture_id", "architecture_preprocess_address",
                "preprocess_contract_digest", "preprocess_source_sha256",
                "trainer_source_sha256", "inference_source_sha256",
                "fit_precommit_record_digest", "fit_precommit_source_sha256",
                "fit_result_record_digest", "fit_result_source_sha256",
                "checkpoint_raw_sha256", "checkpoint_state_dict_sha256",
                "config_digest", "postprediction_contract_digest",
                "postprediction_source_sha256", "straight_class_order",
                "arc_class_order", "catalog_class_order",
                "fit_validation_gate_record_digest", "fit_validation_gate_passed",
            },
            "CNN observer protocol",
        )
        if (
            raw["schema"] != PROTOCOL_SCHEMA
            or raw["architecture_id"] != ARCHITECTURE_ID
            or raw["architecture_preprocess_address"] != architecture_preprocess_address()
            or raw["preprocess_contract_digest"] != preprocess_contract_digest()
            or raw["preprocess_source_sha256"] != FINAL_TRAINER_SOURCE_SHA256
            or raw["trainer_source_sha256"] != FINAL_TRAINER_SOURCE_SHA256
            or raw["postprediction_source_sha256"] != POSTPREDICTION_SOURCE_SHA256
            or raw["straight_class_order"] != list(STRAIGHT_CLASS_ORDER)
            or raw["arc_class_order"] != list(ARC_CLASS_ORDER)
            or raw["catalog_class_order"] != list(CATALOG_CLASS_ORDER)
        ):
            raise CNNToTypedAxisError("CNN observer protocol frozen constants differ")
        result = cls(
            raw["inference_source_sha256"], raw["fit_precommit_record_digest"],
            raw["fit_precommit_source_sha256"], raw["fit_result_record_digest"],
            raw["fit_result_source_sha256"], raw["checkpoint_raw_sha256"],
            raw["checkpoint_state_dict_sha256"], raw["config_digest"],
            raw["postprediction_contract_digest"],
            raw["fit_validation_gate_record_digest"],
            raw["fit_validation_gate_passed"],
        )
        _canonical_match(result.to_data(), raw, "CNN observer protocol")
        return result


def observer_protocol_from_fit_artifacts(
    *,
    fit_precommit: Mapping[str, Any],
    fit_precommit_source_sha256: str,
    fit_result: Mapping[str, Any],
    fit_result_source_sha256: str,
    inference_source_sha256: str,
    postprediction_contract_digest: str,
) -> CNNObserverProtocol:
    """Verify a passed frozen fit record and normalize its observer custody.

    This consumes an in-memory record only.  In particular, it has no path to
    CAL/evaluation outputs and cannot turn the current failed fit into a
    protocol by merely supplying its content address.
    """

    precommit = _sealed_record(fit_precommit, "fit precommit")
    record = _sealed_record(fit_result, "fit result")
    _address(fit_precommit_source_sha256, "fit precommit source")
    _address(fit_result_source_sha256, "fit result source")
    expected_precommit_source = "sha256:" + hashlib.sha256(
        canonical_json(dict(precommit)) + b"\n"
    ).hexdigest()
    expected_source = "sha256:" + hashlib.sha256(
        canonical_json(dict(record)) + b"\n"
    ).hexdigest()
    if (
        fit_precommit_source_sha256 != expected_precommit_source
        or fit_result_source_sha256 != expected_source
    ):
        raise CNNToTypedAxisError("fit artifact source bytes differ")
    gate = record.get("validation_gate")
    adaptive = record.get("adaptive_post_exposure_development_correction")
    precommit_gate = precommit.get("validation_decontamination_gate")
    if (
        precommit.get("schema") != FIT_PRECOMMIT_RECORD_SCHEMA
        or precommit.get("trainer_source_sha256") != FINAL_TRAINER_SOURCE_SHA256
        or record.get("schema") != FIT_RESULT_RECORD_SCHEMA
        or record.get("fit_precommit_record_digest") != precommit.get("record_digest")
        or record.get("architecture_id") != ARCHITECTURE_ID
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or not isinstance(gate.get("checks"), Mapping)
        or set(gate["checks"])
        != {
            "arc_top1",
            "known_catalog_binary_balanced_accuracy",
            "straight_top1",
        }
        or not all(value is True for value in gate["checks"].values())
        or not isinstance(precommit_gate, Mapping)
        or precommit_gate.get("passed") is not True
        or precommit.get("effective_training_panel_count") != 11_200
        or precommit.get("effective_validation_panel_count") != 1_392
        or not isinstance(precommit.get("validation_removed_due_exact_train_duplicate"), list)
        or len(precommit["validation_removed_due_exact_train_duplicate"]) != 8
        or not isinstance(adaptive, Mapping)
        or adaptive.get("validation_decontamination_gate", {}).get("passed") is not True
        or adaptive.get("effective_training_panel_count") != 11_200
        or adaptive.get("effective_validation_panel_count") != 1_392
        or not isinstance(adaptive.get("validation_removed_due_exact_train_duplicate"), list)
        or len(adaptive["validation_removed_due_exact_train_duplicate"]) != 8
    ):
        raise CNNToTypedAxisError("fit result did not pass the frozen release gates")
    gate_address = "sha256:" + canonical_digest(dict(gate))
    return CNNObserverProtocol(
        inference_source_sha256=inference_source_sha256,
        fit_precommit_record_digest=precommit["record_digest"],
        fit_precommit_source_sha256=fit_precommit_source_sha256,
        fit_result_record_digest=record["record_digest"],
        fit_result_source_sha256=fit_result_source_sha256,
        checkpoint_raw_sha256=record["checkpoint_raw_sha256"],
        checkpoint_state_dict_sha256=record["checkpoint_state_dict_sha256"],
        config_digest=record["config_digest"],
        postprediction_contract_digest=postprediction_contract_digest,
        fit_validation_gate_record_digest=gate_address,
        fit_validation_gate_passed=True,
    )


@dataclass(frozen=True, slots=True)
class CNNPopulationGrant:
    scope: PopulationScope
    protocol_address: str
    external_grant_record_digest: str
    external_grant_source_sha256: str
    calibration_prediction_record_digest: str
    calibration_prediction_source_sha256: str
    calibration_label_record_digest: str
    calibration_label_source_sha256: str
    label_bound_prediction_record_digest: str
    label_bound_prediction_source_sha256: str
    deployment_joint_q: float
    calibration_task_ids: tuple[str, ...]
    scope_preregistration_record_digest: str
    scope_preregistration_source_sha256: str
    population_release_record_digest: str
    population_release_source_sha256: str
    population_release_grant_record_digest: str
    authorized_task_ids: tuple[str, ...]
    target_release_authorization_address: str | None = None

    def __post_init__(self) -> None:
        if type(self.scope) is not PopulationScope:
            raise TypeError("population grant needs exact PopulationScope")
        for label, value in (
            ("protocol", self.protocol_address),
            ("external grant record", self.external_grant_record_digest),
            ("external grant source", self.external_grant_source_sha256),
            ("calibration prediction record", self.calibration_prediction_record_digest),
            ("calibration prediction source", self.calibration_prediction_source_sha256),
            ("calibration label record", self.calibration_label_record_digest),
            ("calibration label source", self.calibration_label_source_sha256),
            ("label-bound prediction record", self.label_bound_prediction_record_digest),
            ("label-bound prediction source", self.label_bound_prediction_source_sha256),
            ("scope preregistration record", self.scope_preregistration_record_digest),
            ("scope preregistration source", self.scope_preregistration_source_sha256),
            ("population release record", self.population_release_record_digest),
            ("population release source", self.population_release_source_sha256),
            ("population release grant", self.population_release_grant_record_digest),
        ):
            _address(value, label)
        checked_q = _q(self.deployment_joint_q)
        if type(self.calibration_task_ids) is not tuple or type(self.authorized_task_ids) is not tuple:
            raise TypeError("population grant task inventories need tuples")
        for item in (*self.calibration_task_ids, *self.authorized_task_ids):
            _task_id(item)
        if (
            len(set(self.calibration_task_ids)) != len(self.calibration_task_ids)
            or len(set(self.authorized_task_ids)) != len(self.authorized_task_ids)
            or self.label_bound_prediction_record_digest
            != self.calibration_prediction_record_digest
            or self.label_bound_prediction_source_sha256
            != self.calibration_prediction_source_sha256
            or self.population_release_grant_record_digest
            != self.external_grant_record_digest
            or checked_q != self.deployment_joint_q
        ):
            raise CNNToTypedAxisError("population grant custody differs")

        if self.scope is PopulationScope.GENERIC_FRESH_V3:
            if (
                len(self.calibration_task_ids) != 100
                or any(_semantic_key(item) == TARGET_SEMANTIC_KEY for item in self.calibration_task_ids)
                or any(_semantic_key(item) == TARGET_SEMANTIC_KEY for item in self.authorized_task_ids)
                or self.scope_preregistration_record_digest != GENERIC_V3_PLAN_RECORD_DIGEST
                or self.scope_preregistration_source_sha256 != GENERIC_V3_PLAN_SOURCE_SHA256
                or self.target_release_authorization_address is not None
            ):
                raise CNNToTypedAxisError("generic fresh-V3 population scope differs")
        else:
            if (
                self.calibration_task_ids != SAME_FAMILY_CALIBRATION_TASK_IDS
                or self.authorized_task_ids != (TARGET_TASK_ID,)
                or self.scope_preregistration_record_digest
                != SAME_FAMILY_PREREG_RECORD_DIGEST
                or self.scope_preregistration_source_sha256
                != SAME_FAMILY_PREREG_SOURCE_SHA256
                or self.target_release_authorization_address is None
            ):
                raise CNNToTypedAxisError("same-family population scope differs")
            _address(
                self.target_release_authorization_address,
                "same-family target release authorization",
            )

    @property
    def grant_address(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())

    @property
    def barrier_address(self) -> str:
        return "sha256:" + canonical_digest(
            {
                "calibration_prediction_record_digest": self.calibration_prediction_record_digest,
                "calibration_prediction_source_sha256": self.calibration_prediction_source_sha256,
                "calibration_label_record_digest": self.calibration_label_record_digest,
                "calibration_label_source_sha256": self.calibration_label_source_sha256,
                "label_bound_prediction_record_digest": self.label_bound_prediction_record_digest,
                "label_bound_prediction_source_sha256": self.label_bound_prediction_source_sha256,
                "causal_order": "prediction_fsync_reload_before_label_source_open",
            }
        )

    def authorize_task(self, task_id: str) -> None:
        _task_id(task_id)
        if (
            self.scope is PopulationScope.GENERIC_FRESH_V3
            and _semantic_key(task_id) == TARGET_SEMANTIC_KEY
        ):
            raise CNNToTypedAxisError(
                "generic fresh-V3 grant cannot authorize target-family cells"
            )
        if task_id not in self.authorized_task_ids:
            raise CNNToTypedAxisError("task is outside the exact population release")
        if task_id == TARGET_TASK_ID and self.scope is not PopulationScope.SAME_FAMILY_CONVEX_FOUR_LINES:
            raise CNNToTypedAxisError("_0000 requires the same-family calibration grant")

    def to_data(self) -> dict[str, object]:
        if self.scope is PopulationScope.GENERIC_FRESH_V3:
            external_schema = GENERIC_GRANT_RECORD_SCHEMA
            release_schema = GENERIC_RELEASE_RECORD_SCHEMA
            alpha = 0.05
            order = 96
            q_rule = "sorted_scores[95]"
            population_claim = "fresh_v3_whole_task_exchangeability_excluding_target_family_shift"
        else:
            external_schema = SAME_FAMILY_GRANT_RECORD_SCHEMA
            release_schema = SAME_FAMILY_RELEASE_RECORD_SCHEMA
            alpha = 0.1
            order = 16
            q_rule = "maximum_of_the_16_whole-task_scores"
            population_claim = "same_family_train_repetition_exchangeability_only_16_over_17"
        return {
            "schema": GRANT_SCHEMA,
            "scope": self.scope.value,
            "protocol_address": self.protocol_address,
            "external_grant_schema": external_schema,
            "external_grant_record_digest": self.external_grant_record_digest,
            "external_grant_source_sha256": self.external_grant_source_sha256,
            "calibration_prediction_record_digest": self.calibration_prediction_record_digest,
            "calibration_prediction_source_sha256": self.calibration_prediction_source_sha256,
            "calibration_label_record_digest": self.calibration_label_record_digest,
            "calibration_label_source_sha256": self.calibration_label_source_sha256,
            "label_bound_prediction_record_digest": self.label_bound_prediction_record_digest,
            "label_bound_prediction_source_sha256": self.label_bound_prediction_source_sha256,
            "postprediction_barrier_address": self.barrier_address,
            "postprediction_barrier": "prediction_fsync_reload_before_label_source_open",
            "deployment_joint_q": self.deployment_joint_q,
            "q_rule": q_rule,
            "alpha": alpha,
            "order_statistic_one_indexed": order,
            "calibration_task_ids": list(self.calibration_task_ids),
            "calibration_task_count": len(self.calibration_task_ids),
            "scope_preregistration_record_digest": self.scope_preregistration_record_digest,
            "scope_preregistration_source_sha256": self.scope_preregistration_source_sha256,
            "population_release_schema": release_schema,
            "population_release_record_digest": self.population_release_record_digest,
            "population_release_source_sha256": self.population_release_source_sha256,
            "population_release_grant_record_digest": self.population_release_grant_record_digest,
            "population_release_passed": True,
            "population_claim": population_claim,
            "authorized_task_ids": list(self.authorized_task_ids),
            "target_release_authorization_address": self.target_release_authorization_address,
            "generic_target_family_excluded": True,
        }

    @classmethod
    def from_data(cls, value: object) -> "CNNPopulationGrant":
        raw = _fields(
            value,
            {
                "schema", "scope", "protocol_address", "external_grant_schema",
                "external_grant_record_digest", "external_grant_source_sha256",
                "calibration_prediction_record_digest", "calibration_prediction_source_sha256",
                "calibration_label_record_digest", "calibration_label_source_sha256",
                "label_bound_prediction_record_digest", "label_bound_prediction_source_sha256",
                "postprediction_barrier_address", "postprediction_barrier",
                "deployment_joint_q", "q_rule", "alpha", "order_statistic_one_indexed",
                "calibration_task_ids", "calibration_task_count",
                "scope_preregistration_record_digest", "scope_preregistration_source_sha256",
                "population_release_schema", "population_release_record_digest",
                "population_release_source_sha256", "population_release_grant_record_digest",
                "population_release_passed", "population_claim", "authorized_task_ids",
                "target_release_authorization_address", "generic_target_family_excluded",
            },
            "CNN population grant",
        )
        try:
            scope = PopulationScope(raw["scope"])
        except (TypeError, ValueError) as exc:
            raise CNNToTypedAxisError("population scope differs") from exc
        if type(raw["calibration_task_ids"]) is not list or type(raw["authorized_task_ids"]) is not list:
            raise CNNToTypedAxisError("population grant task inventories differ")
        result = cls(
            scope, raw["protocol_address"], raw["external_grant_record_digest"],
            raw["external_grant_source_sha256"], raw["calibration_prediction_record_digest"],
            raw["calibration_prediction_source_sha256"], raw["calibration_label_record_digest"],
            raw["calibration_label_source_sha256"], raw["label_bound_prediction_record_digest"],
            raw["label_bound_prediction_source_sha256"], raw["deployment_joint_q"],
            tuple(raw["calibration_task_ids"]), raw["scope_preregistration_record_digest"],
            raw["scope_preregistration_source_sha256"], raw["population_release_record_digest"],
            raw["population_release_source_sha256"], raw["population_release_grant_record_digest"],
            tuple(raw["authorized_task_ids"]), raw["target_release_authorization_address"],
        )
        _canonical_match(result.to_data(), raw, "CNN population grant")
        return result


@dataclass(frozen=True, slots=True)
class SupportPanelPrediction:
    panel_id: str
    side: SupportSide
    ordinal: int
    png_sha256: str
    png_size_bytes: int
    straight_logits: tuple[float, ...]
    straight_probabilities: tuple[float, ...]
    straight_class_set: tuple[int, ...]
    catalog_logits: tuple[float, ...]
    catalog_probabilities: tuple[float, ...]
    catalog_class_set: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(
            type(value) is not tuple
            for value in (
                self.straight_logits,
                self.straight_probabilities,
                self.straight_class_set,
                self.catalog_logits,
                self.catalog_probabilities,
                self.catalog_class_set,
            )
        ):
            raise TypeError("support prediction vectors need exact tuples")
        match = _PANEL.fullmatch(self.panel_id) if type(self.panel_id) is str else None
        if match is None or type(self.side) is not SupportSide or type(self.ordinal) is not int:
            raise CNNToTypedAxisError("support prediction panel identity differs")
        expected_folder = "1" if self.side is SupportSide.PRIMARY else "0"
        if match.group("folder") != expected_folder or int(match.group("ordinal")) != self.ordinal:
            raise CNNToTypedAxisError("support role does not match panel path")
        _address(self.png_sha256, "support PNG")
        if type(self.png_size_bytes) is not int or self.png_size_bytes <= 0:
            raise CNNToTypedAxisError("support PNG size differs")
        straight_logits = _finite_logits(self.straight_logits, 10, "straight logits")
        straight_probs = _finite_probability_vector(
            self.straight_probabilities, 10, "straight probabilities"
        )
        catalog_logits = _finite_logits(self.catalog_logits, 3, "catalog logits")
        catalog_probs = _finite_probability_vector(
            self.catalog_probabilities, 3, "catalog probabilities"
        )
        if any(abs(left - right) > 1e-7 for left, right in zip(_softmax(straight_logits), straight_probs)):
            raise CNNToTypedAxisError("straight probabilities differ from logits")
        if any(abs(left - right) > 1e-7 for left, right in zip(_softmax(catalog_logits), catalog_probs)):
            raise CNNToTypedAxisError("catalog probabilities differ from logits")
        _class_set(self.straight_class_set, 10, "straight class set")
        _class_set(self.catalog_class_set, 3, "catalog class set")

    @property
    def task_id(self) -> str:
        match = _PANEL.fullmatch(self.panel_id)
        assert match is not None
        return match.group("task")

    def verify_q(self, joint_q: float) -> None:
        q = _q(joint_q)
        expected_straight = tuple(
            index for index, probability in enumerate(self.straight_probabilities)
            if 1.0 - probability <= q
        )
        expected_catalog = tuple(
            index for index, probability in enumerate(self.catalog_probabilities)
            if 1.0 - probability <= q
        )
        if self.straight_class_set != expected_straight:
            raise CNNToTypedAxisError("straight class set differs from frozen joint q")
        if self.catalog_class_set != expected_catalog:
            raise CNNToTypedAxisError("catalog class set differs from frozen joint q")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_SCHEMA,
            "panel_id": self.panel_id,
            "side": self.side.value,
            "ordinal": self.ordinal,
            "png_sha256": self.png_sha256,
            "png_size_bytes": self.png_size_bytes,
            "straight_logits": list(self.straight_logits),
            "straight_probabilities": list(self.straight_probabilities),
            "straight_class_set": list(self.straight_class_set),
            "catalog_logits": list(self.catalog_logits),
            "catalog_probabilities": list(self.catalog_probabilities),
            "catalog_class_set": list(self.catalog_class_set),
        }

    @classmethod
    def from_data(cls, value: object) -> "SupportPanelPrediction":
        raw = _fields(
            value,
            {
                "schema", "panel_id", "side", "ordinal", "png_sha256",
                "png_size_bytes", "straight_logits", "straight_probabilities",
                "straight_class_set", "catalog_logits", "catalog_probabilities",
                "catalog_class_set",
            },
            "support panel prediction",
        )
        if raw["schema"] != PANEL_SCHEMA or any(
            type(raw[key]) is not list
            for key in (
                "straight_logits", "straight_probabilities", "straight_class_set",
                "catalog_logits", "catalog_probabilities", "catalog_class_set",
            )
        ):
            raise CNNToTypedAxisError("support panel prediction schema differs")
        try:
            side = SupportSide(raw["side"])
        except (TypeError, ValueError) as exc:
            raise CNNToTypedAxisError("support side differs") from exc
        result = cls(
            raw["panel_id"], side, raw["ordinal"], raw["png_sha256"],
            raw["png_size_bytes"], tuple(raw["straight_logits"]),
            tuple(raw["straight_probabilities"]), tuple(raw["straight_class_set"]),
            tuple(raw["catalog_logits"]), tuple(raw["catalog_probabilities"]),
            tuple(raw["catalog_class_set"]),
        )
        _canonical_match(result.to_data(), raw, "support panel prediction")
        return result


@dataclass(frozen=True, slots=True)
class FrozenSupportPredictionBatch:
    task_id: str
    protocol_address: str
    population_grant_address: str
    external_grant_record_digest: str
    prediction_record_digest: str
    prediction_source_sha256: str
    pixel_precommit_record_digest: str
    pixel_precommit_source_sha256: str
    target_authorization_record_digest: str
    target_authorization_source_sha256: str
    checkpoint_state_dict_sha256: str
    config_digest: str
    joint_q: float
    rows: tuple[SupportPanelPrediction, ...]

    def __post_init__(self) -> None:
        _task_id(self.task_id)
        for label, value in (
            ("batch protocol", self.protocol_address),
            ("batch population grant", self.population_grant_address),
            ("batch external grant", self.external_grant_record_digest),
            ("support prediction record", self.prediction_record_digest),
            ("support prediction source", self.prediction_source_sha256),
            ("support pixel precommit record", self.pixel_precommit_record_digest),
            ("support pixel precommit source", self.pixel_precommit_source_sha256),
            ("target authorization record", self.target_authorization_record_digest),
            ("target authorization source", self.target_authorization_source_sha256),
            ("batch checkpoint state", self.checkpoint_state_dict_sha256),
            ("batch config", self.config_digest),
        ):
            _address(value, label)
        _q(self.joint_q)
        if type(self.rows) is not tuple or len(self.rows) != 12:
            raise CNNToTypedAxisError("support prediction batch needs exactly twelve rows")
        if any(type(row) is not SupportPanelPrediction for row in self.rows):
            raise TypeError("support batch rows need exact SupportPanelPrediction")
        expected_roles = (
            tuple((SupportSide.PRIMARY, ordinal) for ordinal in SUPPORT_ORDINALS)
            + tuple((SupportSide.CONTRAST, ordinal) for ordinal in SUPPORT_ORDINALS)
        )
        if tuple((row.side, row.ordinal) for row in self.rows) != expected_roles:
            raise CNNToTypedAxisError("support rows are not fixed six-plus-six order")
        expected_ids = tuple(
            f"hd/{self.task_id}/{1 if side is SupportSide.PRIMARY else 0}/{ordinal}.png"
            for side, ordinal in expected_roles
        )
        if tuple(row.panel_id for row in self.rows) != expected_ids:
            raise CNNToTypedAxisError("support panel IDs differ from task/role binding")
        for row in self.rows:
            row.verify_q(self.joint_q)

    @property
    def batch_address(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        panel_ids = [row.panel_id for row in self.rows]
        panel_bytes = [
            {
                "panel_id": row.panel_id,
                "png_sha256": row.png_sha256,
                "png_size_bytes": row.png_size_bytes,
            }
            for row in self.rows
        ]
        return {
            "schema": BATCH_SCHEMA,
            "task_id": self.task_id,
            "protocol_address": self.protocol_address,
            "population_grant_address": self.population_grant_address,
            "external_grant_record_digest": self.external_grant_record_digest,
            "prediction_record_digest": self.prediction_record_digest,
            "prediction_source_sha256": self.prediction_source_sha256,
            "pixel_precommit_record_digest": self.pixel_precommit_record_digest,
            "pixel_precommit_source_sha256": self.pixel_precommit_source_sha256,
            "target_authorization_record_digest": self.target_authorization_record_digest,
            "target_authorization_source_sha256": self.target_authorization_source_sha256,
            "checkpoint_state_dict_sha256": self.checkpoint_state_dict_sha256,
            "config_digest": self.config_digest,
            "joint_q": self.joint_q,
            "joint_q_record_digest": self.external_grant_record_digest,
            "straight_class_order": list(STRAIGHT_CLASS_ORDER),
            "catalog_class_order": list(CATALOG_CLASS_ORDER),
            "panel_ids_digest": "sha256:" + canonical_digest(panel_ids),
            "panel_byte_bindings_digest": "sha256:" + canonical_digest(panel_bytes),
            "rows": [row.to_data() for row in self.rows],
            "row_count": 12,
            "query_ordinal_4_present": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "FrozenSupportPredictionBatch":
        raw = _fields(
            value,
            {
                "schema", "task_id", "protocol_address", "population_grant_address",
                "external_grant_record_digest", "prediction_record_digest",
                "prediction_source_sha256", "pixel_precommit_record_digest",
                "pixel_precommit_source_sha256", "target_authorization_record_digest",
                "target_authorization_source_sha256", "checkpoint_state_dict_sha256",
                "config_digest", "joint_q", "joint_q_record_digest",
                "straight_class_order", "catalog_class_order", "panel_ids_digest",
                "panel_byte_bindings_digest", "rows", "row_count",
                "query_ordinal_4_present",
            },
            "frozen support prediction batch",
        )
        if raw["schema"] != BATCH_SCHEMA or type(raw["rows"]) is not list:
            raise CNNToTypedAxisError("support prediction batch schema differs")
        result = cls(
            raw["task_id"], raw["protocol_address"], raw["population_grant_address"],
            raw["external_grant_record_digest"], raw["prediction_record_digest"],
            raw["prediction_source_sha256"], raw["pixel_precommit_record_digest"],
            raw["pixel_precommit_source_sha256"], raw["target_authorization_record_digest"],
            raw["target_authorization_source_sha256"], raw["checkpoint_state_dict_sha256"],
            raw["config_digest"], raw["joint_q"],
            tuple(SupportPanelPrediction.from_data(row) for row in raw["rows"]),
        )
        _canonical_match(result.to_data(), raw, "frozen support prediction batch")
        return result


def _observer_protocol_digest(
    protocol: CNNObserverProtocol,
    grant: CNNPopulationGrant,
    batch: FrozenSupportPredictionBatch,
) -> str:
    return "sha256:" + canonical_digest(
        {
            "adapter_algorithm_record_digest": adapter_algorithm_record()["record_digest"],
            "protocol_address": protocol.protocol_address,
            "population_grant_address": grant.grant_address,
            "support_prediction_batch_address": batch.batch_address,
        }
    )


def _straight_cell(
    row: SupportPanelPrediction,
    observer_protocol_digest: str,
    grant_address: str,
) -> TypedAxisCell:
    if not row.straight_class_set:
        return TypedAxisCell.error(
            Axis.STRAIGHT_ACTION_COUNT,
            observer_protocol_digest,
            "empty_straight_class_set",
        )
    return TypedAxisCell.calibrated_set(
        Axis.STRAIGHT_ACTION_COUNT,
        row.straight_class_set,
        observer_protocol_digest,
        grant_address,
    )


def _catalog_cell(
    row: SupportPanelPrediction,
    observer_protocol_digest: str,
    grant_address: str,
) -> TypedAxisCell:
    values = row.catalog_class_set
    if not values:
        return TypedAxisCell.error(
            Axis.CATALOG_CONVEXITY,
            observer_protocol_digest,
            "empty_catalog_class_set",
        )
    if 0 in values:
        return TypedAxisCell.gap(
            Axis.CATALOG_CONVEXITY,
            observer_protocol_digest,
            "catalog_set_contains_unresolved",
        )
    mapped = tuple(
        "catalog_nonconvex" if index == 1 else "catalog_convex"
        for index in values
    )
    return TypedAxisCell.calibrated_set(
        Axis.CATALOG_CONVEXITY,
        mapped,
        observer_protocol_digest,
        grant_address,
    )


def _typed_row(
    prediction: SupportPanelPrediction,
    observer_protocol_digest: str,
    grant_address: str,
) -> TypedSupportRow:
    cells: list[TypedAxisCell] = []
    for axis in AXES:
        if axis is Axis.STRAIGHT_ACTION_COUNT:
            cell = _straight_cell(prediction, observer_protocol_digest, grant_address)
        elif axis is Axis.CATALOG_CONVEXITY:
            cell = _catalog_cell(prediction, observer_protocol_digest, grant_address)
        else:
            cell = TypedAxisCell.gap(
                axis,
                observer_protocol_digest,
                "cnn_axis_not_observed",
            )
        cells.append(cell)
    return TypedSupportRow(prediction.panel_id, prediction.side, tuple(cells))


@dataclass(frozen=True, slots=True)
class CNNTypedAxisMatrixArtifact:
    protocol: CNNObserverProtocol
    population_grant: CNNPopulationGrant
    prediction_batch: FrozenSupportPredictionBatch
    matrix: TypedSupportMatrix

    def __post_init__(self) -> None:
        if (
            type(self.protocol) is not CNNObserverProtocol
            or type(self.population_grant) is not CNNPopulationGrant
            or type(self.prediction_batch) is not FrozenSupportPredictionBatch
            or type(self.matrix) is not TypedSupportMatrix
        ):
            raise TypeError("matrix artifact members need exact adapter types")
        self.population_grant.authorize_task(self.prediction_batch.task_id)
        if (
            self.population_grant.protocol_address != self.protocol.protocol_address
            or self.prediction_batch.protocol_address != self.protocol.protocol_address
            or self.prediction_batch.population_grant_address
            != self.population_grant.grant_address
            or self.prediction_batch.external_grant_record_digest
            != self.population_grant.external_grant_record_digest
            or self.prediction_batch.checkpoint_state_dict_sha256
            != self.protocol.checkpoint_state_dict_sha256
            or self.prediction_batch.config_digest != self.protocol.config_digest
            or self.prediction_batch.joint_q != self.population_grant.deployment_joint_q
            or (
                self.population_grant.scope
                is PopulationScope.SAME_FAMILY_CONVEX_FOUR_LINES
                and self.prediction_batch.target_authorization_record_digest
                != self.population_grant.target_release_authorization_address
            )
        ):
            raise CNNToTypedAxisError("protocol/grant/prediction custody differs")
        expected_protocol = _observer_protocol_digest(
            self.protocol, self.population_grant, self.prediction_batch
        )
        expected = TypedSupportMatrix.freeze(
            tuple(
                _typed_row(row, expected_protocol, self.population_grant.grant_address)
                for row in self.prediction_batch.rows
            )
        )
        if self.matrix != expected:
            raise CNNToTypedAxisError("typed matrix differs from CNN projection")

    @property
    def artifact_address(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ARTIFACT_SCHEMA,
            "algorithm": adapter_algorithm_record(),
            "protocol": self.protocol.to_data(),
            "population_grant": self.population_grant.to_data(),
            "prediction_batch": self.prediction_batch.to_data(),
            "matrix": self.matrix.to_data(),
            "matrix_address": self.matrix.matrix_address,
            "support_row_count": 12,
            "query_rows_seen": 0,
            "png_reads_during_adaptation_or_replay": 0,
            "model_calls_during_adaptation_or_replay": 0,
            "label_source_calls_during_adaptation_or_replay": 0,
            "lean_present": False,
            "python_is_canonical_authority": True,
        }

    @classmethod
    def from_data(cls, value: object) -> "CNNTypedAxisMatrixArtifact":
        raw = _fields(
            value,
            {
                "schema", "algorithm", "protocol", "population_grant",
                "prediction_batch", "matrix", "matrix_address", "support_row_count",
                "query_rows_seen", "png_reads_during_adaptation_or_replay",
                "model_calls_during_adaptation_or_replay",
                "label_source_calls_during_adaptation_or_replay", "lean_present",
                "python_is_canonical_authority",
            },
            "CNN typed-axis matrix artifact",
        )
        if raw["schema"] != ARTIFACT_SCHEMA or raw["algorithm"] != adapter_algorithm_record():
            raise CNNToTypedAxisError("matrix artifact algorithm differs")
        result = cls(
            CNNObserverProtocol.from_data(raw["protocol"]),
            CNNPopulationGrant.from_data(raw["population_grant"]),
            FrozenSupportPredictionBatch.from_data(raw["prediction_batch"]),
            TypedSupportMatrix.from_data(raw["matrix"]),
        )
        _canonical_match(result.to_data(), raw, "CNN typed-axis matrix artifact")
        return result


def build_cnn_typed_support_matrix(
    *,
    protocol: CNNObserverProtocol,
    population_grant: CNNPopulationGrant,
    prediction_batch: FrozenSupportPredictionBatch,
) -> CNNTypedAxisMatrixArtifact:
    """Project exactly twelve frozen support predictions; perform no I/O."""

    if type(protocol) is not CNNObserverProtocol:
        raise TypeError("protocol needs exact CNNObserverProtocol")
    if type(population_grant) is not CNNPopulationGrant:
        raise TypeError("population grant needs exact CNNPopulationGrant")
    if type(prediction_batch) is not FrozenSupportPredictionBatch:
        raise TypeError("prediction batch needs exact FrozenSupportPredictionBatch")
    population_grant.authorize_task(prediction_batch.task_id)
    if (
        population_grant.protocol_address != protocol.protocol_address
        or prediction_batch.protocol_address != protocol.protocol_address
        or prediction_batch.population_grant_address != population_grant.grant_address
        or prediction_batch.external_grant_record_digest
        != population_grant.external_grant_record_digest
        or prediction_batch.checkpoint_state_dict_sha256
        != protocol.checkpoint_state_dict_sha256
        or prediction_batch.config_digest != protocol.config_digest
        or prediction_batch.joint_q != population_grant.deployment_joint_q
        or (
            population_grant.scope
            is PopulationScope.SAME_FAMILY_CONVEX_FOUR_LINES
            and prediction_batch.target_authorization_record_digest
            != population_grant.target_release_authorization_address
        )
    ):
        raise CNNToTypedAxisError("protocol/grant/prediction custody differs")
    observer_protocol = _observer_protocol_digest(protocol, population_grant, prediction_batch)
    matrix = TypedSupportMatrix.freeze(
        tuple(
            _typed_row(row, observer_protocol, population_grant.grant_address)
            for row in prediction_batch.rows
        )
    )
    return CNNTypedAxisMatrixArtifact(protocol, population_grant, prediction_batch, matrix)


def cold_replay_cnn_typed_support_matrix(
    artifact: CNNTypedAxisMatrixArtifact, *, expected_artifact_address: str
) -> CNNTypedAxisMatrixArtifact:
    """Rebuild the complete projection from canonical records with zero calls/I/O."""

    if type(artifact) is not CNNTypedAxisMatrixArtifact:
        raise TypeError("cold replay needs exact CNNTypedAxisMatrixArtifact")
    _address(expected_artifact_address, "expected matrix artifact")
    restored = CNNTypedAxisMatrixArtifact.from_data(artifact.to_data())
    if restored.artifact_address != expected_artifact_address:
        raise CNNToTypedAxisError("matrix artifact address differs")
    return restored


__all__ = (
    "ARCHITECTURE_ID",
    "CATALOG_CLASS_ORDER",
    "CNNObserverProtocol",
    "CNNPopulationGrant",
    "CNNTypedAxisMatrixArtifact",
    "CNNToTypedAxisError",
    "FINAL_TRAINER_SOURCE_SHA256",
    "FrozenSupportPredictionBatch",
    "GENERIC_V3_PLAN_RECORD_DIGEST",
    "GENERIC_V3_PLAN_SOURCE_SHA256",
    "PopulationScope",
    "SAME_FAMILY_CALIBRATION_TASK_IDS",
    "SAME_FAMILY_PREREG_RECORD_DIGEST",
    "SAME_FAMILY_PREREG_SOURCE_SHA256",
    "STRAIGHT_CLASS_ORDER",
    "SUPPORT_ORDINALS",
    "SupportPanelPrediction",
    "TARGET_SEMANTIC_KEY",
    "TARGET_TASK_ID",
    "adapter_algorithm_record",
    "adapter_source_digest",
    "architecture_preprocess_address",
    "build_cnn_typed_support_matrix",
    "cold_replay_cnn_typed_support_matrix",
    "observer_protocol_from_fit_artifacts",
    "preprocess_contract_digest",
)
