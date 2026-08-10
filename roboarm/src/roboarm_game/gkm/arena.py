"""Host-only RoboArm connector used behind the deterministic safety FSA.

The coding proposer never imports this module and receives no socket, token, or
live environment handle.  It emits declarative scenarios.  The trusted host
uses this connector for isolated digital-twin preflight and, only with a
single-use in-memory permit minted after deterministic validation, committed
execution.
"""

from __future__ import annotations

import copy
import secrets
from dataclasses import dataclass
from typing import Callable, Sequence

from ..environment import make_env
from ..observation import (
    FRAME_ENCODING,
    FRAME_SHAPE,
    OBSERVATION_SCHEMA_VERSION,
    SENSOR_CONTRACT_ID,
    camera_model,
    frame_record,
    validated_operational_telemetry,
)
from ..protocol import Environment
from .scenario import canonical_sha256


class ConnectorViolation(RuntimeError):
    """The trusted connector rejected an invalid or unauthorized operation."""


def _camera_record(frame: object) -> tuple[str, str]:
    try:
        return frame_record(frame)  # type: ignore[arg-type]
    except ValueError as error:
        raise ConnectorViolation(
            f"environment returned an invalid RGB camera frame: {error}"
        ) from error


def _telemetry_record(
    environment: Environment,
) -> tuple[dict[str, object], str]:
    try:
        telemetry = validated_operational_telemetry(
            environment.telemetry()
        )
    except ValueError as error:
        raise ConnectorViolation(
            f"environment returned invalid public telemetry: {error}"
        ) from error
    return telemetry, canonical_sha256(telemetry)


def _sensor_pair_record(
    frame_sha256: str,
    telemetry: dict[str, object],
    telemetry_sha256: str,
) -> dict[str, object]:
    """Bind independently clocked arm and camera products at host receipt."""

    sample = telemetry.get("sample")
    camera = telemetry.get("camera")
    controller = telemetry.get("controller")
    if not isinstance(sample, dict) or not isinstance(camera, dict) or not isinstance(controller, dict):
        raise ConnectorViolation("sensor packet is missing I/O pairing metadata")
    if sample.get("camera_capture_time_s") != camera.get("capture_time_s"):
        raise ConnectorViolation("C920s timestamp disagrees across sensor packet")
    skew = sample.get("sensor_skew_ms")
    if isinstance(skew, bool) or not isinstance(skew, (int, float)) or not 0.0 <= float(skew) < 34.0:
        raise ConnectorViolation("arm/C920s sample skew exceeds one camera period")
    value: dict[str, object] = {
        "pairing": "latest_camera_frame_at_arm_feedback_receipt",
        "frame_sha256": frame_sha256,
        "telemetry_sha256": telemetry_sha256,
        "sample_sequence": sample.get("sequence"),
        "camera_sequence": camera.get("sequence"),
        "sensor_skew_ms": skew,
        "arm_feedback_type": (
            telemetry.get("arm", {}).get("feedback", {}).get("T")
            if isinstance(telemetry.get("arm"), dict)
            and isinstance(telemetry.get("arm", {}).get("feedback"), dict)
            else None
        ),
        "command_interlocked": controller.get("interlocked"),
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return value


@dataclass(frozen=True, slots=True)
class _CommitPermit:
    """Opaque one-use capability; never serialized or exposed to a proposer."""

    authority: object
    nonce: str
    preflight_receipt_sha256: str
    actions: tuple[int, ...]
    expected_frame_sha256: tuple[str, ...]
    expected_telemetry_sha256: tuple[str, ...]
    safety_receipt_sha256: str


class RoboArmConnector:
    """Authoritative simulator adapter with no proposer-facing actuation API."""

    def __init__(
        self,
        *,
        seed: int = 0,
        scenario: str = "round-1",
        max_committed_actions: int = 2_000,
        max_preflight_actions: int = 12_000,
        environment_factory: Callable[[], Environment] | None = None,
    ) -> None:
        if max_committed_actions <= 0 or max_preflight_actions <= 0:
            raise ValueError("connector action budgets must be positive")
        self.seed = int(seed)
        self.scenario = scenario
        self.max_committed_actions = int(max_committed_actions)
        self.max_preflight_actions = int(max_preflight_actions)
        self._factory = environment_factory or (
            lambda: make_env(
                "rb01-v1",
                seed=self.seed,
                scenario=self.scenario,
            )
        )
        self._authority = object()
        self._live_permits: set[str] = set()
        self._preflight_actions = 0
        self._committed_actions = 0
        self._preflights = 0
        self._commits = 0
        self._attempt_summaries: list[dict[str, object]] = []

    @property
    def preflight_actions(self) -> int:
        return self._preflight_actions

    @property
    def committed_actions(self) -> int:
        return self._committed_actions

    def _new_environment(self) -> Environment:
        environment = self._factory()
        if tuple(environment.actions) != (1, 2, 3, 4, 5, 6):
            raise ConnectorViolation(
                "authoritative environment action contract changed"
            )
        return environment

    @staticmethod
    def _snapshot(environment: Environment) -> dict[str, object] | None:
        snapshot_method = getattr(environment, "snapshot", None)
        if snapshot_method is None:
            return None
        value = snapshot_method()
        if not isinstance(value, dict):
            raise ConnectorViolation(
                "authoritative visual snapshot is not an object"
            )
        return copy.deepcopy(value)

    @staticmethod
    def _validated_actions(actions: Sequence[int]) -> tuple[int, ...]:
        if not isinstance(actions, (list, tuple)) or not actions:
            raise ConnectorViolation("action sequence must be nonempty")
        result: list[int] = []
        for value in actions:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or int(value) not in (1, 2, 3, 4, 5, 6)
            ):
                raise ConnectorViolation(
                    "action sequence left the public six-action contract"
                )
            result.append(int(value))
        return tuple(result)

    def initial_observation(self) -> dict[str, object]:
        """Return a public reset observation without exposing a live handle."""

        environment = self._new_environment()
        frame = environment.reset()
        frame_sha256, frame_b64 = _camera_record(frame)
        telemetry, telemetry_sha256 = _telemetry_record(environment)
        sensor_pair = _sensor_pair_record(
            frame_sha256,
            telemetry,
            telemetry_sha256,
        )
        observation = {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "game_id": "rb01-v1",
            "scenario": self.scenario,
            "seed": self.seed,
            "sensor_contract_id": SENSOR_CONTRACT_ID,
            "frame_encoding": FRAME_ENCODING,
            "frame_shape": list(FRAME_SHAPE),
            "camera_model": camera_model(),
            "frame_sha256": frame_sha256,
            "frame_b64": frame_b64,
            "telemetry_sha256": telemetry_sha256,
            "telemetry": telemetry,
            "sensor_pair": sensor_pair,
            "levels_completed": int(environment.levels_completed),
            "terminal": bool(environment.terminal()),
        }
        observation["receipt_sha256"] = canonical_sha256(observation)
        return observation

    def _execute_fresh(
        self,
        actions: tuple[int, ...],
        *,
        attempt_id: str,
        role: str,
        committed: bool,
        expected_frame_sha256: tuple[str, ...] | None = None,
        expected_telemetry_sha256: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        environment = self._new_environment()
        initial = environment.reset()
        initial_sha256, initial_b64 = _camera_record(initial)
        initial_telemetry, initial_telemetry_sha256 = _telemetry_record(
            environment
        )
        initial_visual_state = self._snapshot(environment)
        steps: list[dict[str, object]] = []

        for action in actions:
            if environment.terminal():
                break
            if committed:
                if self._committed_actions >= self.max_committed_actions:
                    raise ConnectorViolation(
                        "committed-action campaign budget exhausted"
                    )
            elif self._preflight_actions >= self.max_preflight_actions:
                raise ConnectorViolation(
                    "preflight-action campaign budget exhausted"
                )

            before = environment.frame()
            before_sha256, _ = _camera_record(before)
            _, before_telemetry_sha256 = _telemetry_record(environment)
            if committed:
                # Every live transition gets a just-in-time digital-twin check
                # before the authoritative state is allowed to mutate.
                guard = environment.clone()
                guard_after = guard.step(action)
                guard_sha256, _ = _camera_record(guard_after)
                _, guard_telemetry_sha256 = _telemetry_record(guard)
                expected_index = len(steps)
                if (
                    expected_frame_sha256 is None
                    or expected_telemetry_sha256 is None
                    or expected_index >= len(expected_frame_sha256)
                    or expected_index >= len(expected_telemetry_sha256)
                    or guard_sha256
                    != expected_frame_sha256[expected_index]
                    or guard_telemetry_sha256
                    != expected_telemetry_sha256[expected_index]
                ):
                    raise ConnectorViolation(
                        "commit interlock diverged from admitted preflight"
                    )

            after = environment.step(action)
            frame_sha256, frame_b64 = _camera_record(after)
            telemetry, telemetry_sha256 = _telemetry_record(environment)
            sensor_pair = _sensor_pair_record(
                frame_sha256,
                telemetry,
                telemetry_sha256,
            )
            if committed:
                self._committed_actions += 1
            else:
                self._preflight_actions += 1
            step = {
                "turn": len(steps) + 1,
                "role": role,
                "action": action,
                "before_frame_sha256": before_sha256,
                "before_telemetry_sha256": before_telemetry_sha256,
                "frame_sha256": frame_sha256,
                "frame_b64": frame_b64,
                "telemetry_sha256": telemetry_sha256,
                "telemetry": telemetry,
                "sensor_pair": sensor_pair,
                "levels_completed": int(environment.levels_completed),
                "terminal": bool(environment.terminal()),
                # Private host/browser evidence.  The public-evidence projector
                # removes this before a later proposer generation can see it.
                "visual_state": self._snapshot(environment),
            }
            steps.append(step)

        trace: dict[str, object] = {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "game_id": "rb01-v1",
            "scenario": self.scenario,
            "seed": self.seed,
            "role": role,
            "sensor_contract_id": SENSOR_CONTRACT_ID,
            "frame_encoding": FRAME_ENCODING,
            "frame_shape": list(FRAME_SHAPE),
            "camera_model": camera_model(),
            "initial_frame_sha256": initial_sha256,
            "initial_frame_b64": initial_b64,
            "initial_telemetry_sha256": initial_telemetry_sha256,
            "initial_telemetry": initial_telemetry,
            "initial_visual_state": initial_visual_state,
            "actions_requested": list(actions),
            "actions": [
                int(step["action"]) for step in steps
            ],
            "levels_completed": int(environment.levels_completed),
            "terminal": bool(environment.terminal()),
            "steps": steps,
        }
        trace["receipt_sha256"] = canonical_sha256(trace)
        self._attempt_summaries.append(
            {
                "attempt_id": attempt_id,
                "role": role,
                "actions": len(steps),
                "levels_completed": trace["levels_completed"],
                "terminal": trace["terminal"],
                "receipt_sha256": trace["receipt_sha256"],
            }
        )
        return trace

    def preflight(
        self,
        actions: Sequence[int],
        *,
        attempt_id: str,
    ) -> dict[str, object]:
        """Run one isolated authoritative digital-twin experiment."""

        validated = self._validated_actions(actions)
        self._preflights += 1
        return self._execute_fresh(
            validated,
            attempt_id=attempt_id,
            role="fsa_preflight",
            committed=False,
        )

    def _mint_permit(
        self,
        *,
        actions: Sequence[int],
        preflight: dict[str, object],
        safety_receipt_sha256: str,
    ) -> _CommitPermit:
        """Mint a one-use capability after the FSA has verified preflight."""

        validated = self._validated_actions(actions)
        preflight_actions = preflight.get("actions")
        steps = preflight.get("steps")
        receipt = preflight.get("receipt_sha256")
        if (
            preflight_actions != list(validated)
            or not isinstance(steps, list)
            or len(steps) != len(validated)
            or not isinstance(receipt, str)
            or not isinstance(safety_receipt_sha256, str)
            or len(safety_receipt_sha256) != 64
        ):
            raise ConnectorViolation(
                "permit request is not bound to a complete preflight"
            )
        expected: list[str] = []
        expected_telemetry: list[str] = []
        for step in steps:
            if (
                not isinstance(step, dict)
                or not isinstance(step.get("frame_sha256"), str)
                or not isinstance(step.get("telemetry_sha256"), str)
            ):
                raise ConnectorViolation(
                    "permit preflight has malformed sensor evidence"
                )
            expected.append(str(step["frame_sha256"]))
            expected_telemetry.append(str(step["telemetry_sha256"]))
        nonce = secrets.token_hex(24)
        self._live_permits.add(nonce)
        return _CommitPermit(
            authority=self._authority,
            nonce=nonce,
            preflight_receipt_sha256=receipt,
            actions=validated,
            expected_frame_sha256=tuple(expected),
            expected_telemetry_sha256=tuple(expected_telemetry),
            safety_receipt_sha256=safety_receipt_sha256,
        )

    def _commit_authorized(
        self,
        permit: _CommitPermit,
        *,
        attempt_id: str,
    ) -> dict[str, object]:
        """Execute only a valid, unspent, FSA-minted in-memory permit."""

        if (
            not isinstance(permit, _CommitPermit)
            or permit.authority is not self._authority
            or permit.nonce not in self._live_permits
        ):
            raise ConnectorViolation(
                "committed execution lacks an authentic safety permit"
            )
        self._live_permits.remove(permit.nonce)
        self._commits += 1
        return self._execute_fresh(
            permit.actions,
            attempt_id=attempt_id,
            role="fsa_committed",
            committed=True,
            expected_frame_sha256=permit.expected_frame_sha256,
            expected_telemetry_sha256=permit.expected_telemetry_sha256,
        )

    def evidence(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "game_id": "rb01-v1",
            "scenario": self.scenario,
            "seed": self.seed,
            "sensor_contract_id": SENSOR_CONTRACT_ID,
            "preflights": self._preflights,
            "commits": self._commits,
            "preflight_actions": self._preflight_actions,
            "committed_actions": self._committed_actions,
            "max_preflight_actions": self.max_preflight_actions,
            "max_committed_actions": self.max_committed_actions,
            "attempts": copy.deepcopy(self._attempt_summaries),
            "live_permits": len(self._live_permits),
        }
        value["receipt_sha256"] = canonical_sha256(value)
        return value


__all__ = [
    "ConnectorViolation",
    "RoboArmConnector",
]
