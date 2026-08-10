"""Replay-check and export admitted GKM evidence for the browser viewer."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

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
from .accounting import canonical_json_sha256
from .lineage import campaign_lineage_profile
from .replay import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIEWER_DESTINATION = PROJECT_ROOT / "web" / "public" / "campaign"


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or linked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    return _object(value, label)


def _telemetry_record(
    environment: object,
) -> tuple[dict[str, object], str]:
    telemetry_method = getattr(environment, "telemetry", None)
    if telemetry_method is None:
        raise ValueError("authoritative environment has no public telemetry")
    telemetry = validated_operational_telemetry(telemetry_method())
    return telemetry, canonical_json_sha256(telemetry)


def _actions(value: object) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("attempt actions must be a nonempty list")
    result: list[int] = []
    for action in value:
        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or action not in range(1, 7)
        ):
            raise ValueError("attempt contains an invalid public action")
        result.append(int(action))
    if len(result) > 160:
        raise ValueError("attempt exceeds the round action budget")
    return result


def _optional_text(
    source: dict[str, Any],
    key: str,
    *,
    limit: int = 4_000,
) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or len(value.encode("utf-8")) > limit
    ):
        raise ValueError(f"browser attempt {key} is invalid")
    return value


def _validated_attempt(
    source: dict[str, Any],
    *,
    campaign_id: str,
) -> dict[str, Any]:
    if (
        source.get("schema_version") != OBSERVATION_SCHEMA_VERSION
        or source.get("attempt_kind") != "gkm"
        or source.get("disposition") not in {"failed", "promoted"}
        or source.get("game_id") != "rb01-v1"
    ):
        raise ValueError("browser attempt has an unsupported schema")
    seed = source.get("seed")
    scenario = source.get("scenario")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("browser attempt seed is invalid")
    if scenario != "round-1":
        raise ValueError("browser attempt scenario is not round-1")
    if (
        source.get("sensor_contract_id") != SENSOR_CONTRACT_ID
        or source.get("frame_encoding") != FRAME_ENCODING
        or source.get("frame_shape") != list(FRAME_SHAPE)
        or source.get("camera_model") != camera_model()
    ):
        raise ValueError("browser attempt sensor contract is obsolete")
    actions = _actions(source.get("actions"))
    recorded_steps = source.get("steps")
    if not isinstance(recorded_steps, list) or len(recorded_steps) != len(actions):
        raise ValueError("browser attempt action/step lengths differ")

    env = make_env("rb01-v1", seed=seed, scenario=scenario)
    initial = env.reset()
    initial_sha, initial_b64 = frame_record(initial)
    initial_telemetry, initial_telemetry_sha = _telemetry_record(env)
    if source.get("initial_frame_sha256") != initial_sha:
        raise ValueError("browser attempt initial frame hash failed replay")
    if source.get("initial_frame_b64") != initial_b64:
        raise ValueError("browser attempt initial frame bytes failed replay")
    if source.get("initial_telemetry") != initial_telemetry:
        raise ValueError("browser attempt initial telemetry failed replay")
    if source.get("initial_telemetry_sha256") != initial_telemetry_sha:
        raise ValueError("browser attempt initial telemetry hash failed replay")
    snapshot_method = getattr(env, "snapshot", None)
    if snapshot_method is None:
        raise ValueError("environment has no browser visual snapshot")
    initial_visual_state = snapshot_method()

    checked_steps: list[dict[str, object]] = []
    for turn, (action, recorded_value) in enumerate(
        zip(actions, recorded_steps, strict=True),
        1,
    ):
        recorded = _object(recorded_value, f"browser step {turn}")
        if recorded.get("action") != action:
            raise ValueError(f"browser step {turn} action failed replay")
        before_sha, _ = frame_record(env.frame())
        _, before_telemetry_sha = _telemetry_record(env)
        after = env.step(action)
        frame_sha, frame_b64 = frame_record(after)
        telemetry, telemetry_sha = _telemetry_record(env)
        if recorded.get("frame_sha256") != frame_sha:
            raise ValueError(f"browser step {turn} frame hash failed replay")
        if recorded.get("frame_b64") != frame_b64:
            raise ValueError(f"browser step {turn} frame bytes failed replay")
        if recorded.get("before_frame_sha256") not in {None, before_sha}:
            raise ValueError(f"browser step {turn} before-frame hash failed replay")
        if recorded.get("telemetry") != telemetry:
            raise ValueError(f"browser step {turn} telemetry failed replay")
        if recorded.get("telemetry_sha256") != telemetry_sha:
            raise ValueError(f"browser step {turn} telemetry hash failed replay")
        if recorded.get("before_telemetry_sha256") not in {
            None,
            before_telemetry_sha,
        }:
            raise ValueError(
                f"browser step {turn} before-telemetry hash failed replay"
            )
        if int(recorded.get("levels_completed", -1)) != env.levels_completed:
            raise ValueError(f"browser step {turn} reward failed replay")
        if bool(recorded.get("terminal")) != env.terminal():
            raise ValueError(f"browser step {turn} terminal state failed replay")
        role = recorded.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"browser step {turn} has no trace role")
        checked_steps.append(
            {
                "turn": turn,
                "role": role,
                "action": action,
                "before_frame_sha256": before_sha,
                "before_telemetry_sha256": before_telemetry_sha,
                "frame_sha256": frame_sha,
                "frame_b64": frame_b64,
                "telemetry_sha256": telemetry_sha,
                "telemetry": telemetry,
                "levels_completed": int(env.levels_completed),
                "terminal": bool(env.terminal()),
                "visual_state": snapshot_method(),
            }
        )

    disposition = str(source["disposition"])
    completed = env.levels_completed >= 1
    if disposition == "failed" and completed:
        raise ValueError("failed browser attempt completed the round on replay")
    if disposition == "promoted" and not completed:
        raise ValueError("promoted browser attempt failed the round on replay")

    admitted = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "attempt_kind": "gkm",
        "disposition": disposition,
        "trace_role": source.get("trace_role"),
        "campaign_id": campaign_id,
        "attempt_id": _optional_text(source, "attempt_id"),
        "scenario_id": _optional_text(source, "scenario_id"),
        "hypothesis": _optional_text(source, "hypothesis"),
        "expected_observation": _optional_text(
            source,
            "expected_observation",
        ),
        "replay_stage": _optional_text(source, "replay_stage"),
        "fsa_receipt_sha256": _optional_text(
            source,
            "fsa_receipt_sha256",
        ),
        "game_id": "rb01-v1",
        "scenario": scenario,
        "seed": seed,
        "source_tree_sha256": source.get("source_tree_sha256"),
        "promotion_receipt_sha256": source.get("promotion_receipt_sha256"),
        "sensor_contract_id": SENSOR_CONTRACT_ID,
        "frame_encoding": FRAME_ENCODING,
        "frame_shape": list(FRAME_SHAPE),
        "camera_model": camera_model(),
        "initial_frame_sha256": initial_sha,
        "initial_frame_b64": initial_b64,
        "initial_telemetry_sha256": initial_telemetry_sha,
        "initial_telemetry": initial_telemetry,
        "initial_visual_state": initial_visual_state,
        "actions": actions,
        "steps": checked_steps,
    }
    generation = source.get("generation")
    if generation is not None and (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation <= 0
    ):
        raise ValueError("browser attempt generation is invalid")
    admitted["generation"] = generation
    raw_failure_evidence = source.get("observed_failure_evidence", [])
    if (
        not isinstance(raw_failure_evidence, list)
        or any(
            not isinstance(value, str) or not value
            for value in raw_failure_evidence
        )
    ):
        raise ValueError("browser failure evidence is invalid")
    admitted["observed_failure_evidence"] = list(raw_failure_evidence)
    if disposition == "failed" and not raw_failure_evidence:
        raise ValueError("failed browser attempt has no operational failure")
    if disposition == "promoted" and raw_failure_evidence:
        raise ValueError("promoted browser attempt carries failure evidence")
    admitted["replay_receipt_sha256"] = canonical_json_sha256(admitted)
    return admitted


def export_campaign_viewer(
    campaign_root: Path,
    destination: Path = DEFAULT_VIEWER_DESTINATION,
    *,
    replace: bool = False,
) -> dict[str, object]:
    """Export replay-checked failures and successes from one promotion."""

    project_root = PROJECT_ROOT.resolve(strict=True)
    campaign = campaign_root.resolve(strict=True)
    if not campaign.is_relative_to(project_root / "artifacts"):
        raise ValueError("campaign root must stay below roboarm/artifacts")
    target = destination.resolve(strict=False)
    if not target.is_relative_to(project_root):
        raise ValueError("viewer export must stay below roboarm")

    result = _read_json(campaign / "campaign_result.json", "campaign result")
    if result.get("promoted") is not True:
        raise ValueError("campaign has no admitted replay-gated promotion")
    if result.get("genuine_failed_attempt") is not True:
        raise ValueError("campaign has no genuine nonempty failed attempt")
    campaign_id = str(result.get("campaign_id", ""))
    if not campaign_id:
        raise ValueError("campaign result has no campaign id")

    manifest_source = _read_json(
        campaign / "browser" / "manifest.json",
        "browser manifest",
    )
    filenames = manifest_source.get("attempts")
    if (
        not isinstance(filenames, list)
        or not 2 <= len(filenames) <= 12
        or len(filenames) != len(set(filenames))
        or any(
            not isinstance(filename, str)
            or not filename.endswith(".json")
            or not filename.replace("_", "").replace("-", "").replace(
                ".", ""
            ).isalnum()
            for filename in filenames
        )
    ):
        raise ValueError("browser manifest has invalid attempt filenames")

    attempts: dict[str, dict[str, Any]] = {}
    for filename in filenames:
        source = _read_json(campaign / "browser" / filename, filename)
        attempts[filename] = _validated_attempt(
            source,
            campaign_id=campaign_id,
        )
    failures = [
        attempt
        for attempt in attempts.values()
        if attempt["disposition"] == "failed"
    ]
    successes = [
        attempt
        for attempt in attempts.values()
        if attempt["disposition"] == "promoted"
    ]
    if not failures or not successes:
        raise ValueError("browser manifest must contain failure and success")
    promoted_source = successes[-1]
    lineage = campaign_lineage_profile(campaign)

    manifest: dict[str, object] = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "attempts": list(filenames),
        "failure_replays": len(failures),
        "success_replays": len(successes),
        "source_tree_sha256": promoted_source.get("source_tree_sha256"),
        "promotion_receipt_sha256": promoted_source.get(
            "promotion_receipt_sha256"
        ),
        "sensor_contract_id": SENSOR_CONTRACT_ID,
        "frame_encoding": FRAME_ENCODING,
        "frame_shape": list(FRAME_SHAPE),
        "camera_model": camera_model(),
        "export_kind": "replay-validated-gkm-evidence",
        "lineage_profile": "lineage_profile.json",
        "lineage_profile_receipt_sha256": lineage[
            "profile_receipt_sha256"
        ],
    }
    manifest["export_receipt_sha256"] = canonical_json_sha256(
        {
            "manifest": manifest,
            "attempt_receipts": {
                name: value["replay_receipt_sha256"]
                for name, value in attempts.items()
            },
        }
    )

    staging = target.parent / f".{target.name}-staging-{uuid.uuid4().hex}"
    retired: Path | None = None
    staging.mkdir(parents=True, exist_ok=False)
    try:
        write_json(staging / "manifest.json", manifest)
        write_json(staging / "lineage_profile.json", lineage)
        for filename, attempt in attempts.items():
            write_json(staging / filename, attempt)
        if target.exists() or target.is_symlink():
            if not replace:
                raise FileExistsError(
                    "viewer export already exists; pass replace=True explicitly"
                )
            retired = target.parent / (
                f".{target.name}-retired-{uuid.uuid4().hex}"
            )
            shutil.move(str(target), str(retired))
        shutil.move(str(staging), str(target))
        if retired is not None:
            if retired.is_symlink() or retired.is_file():
                retired.unlink()
            else:
                shutil.rmtree(retired)
            retired = None
    except Exception:
        if retired is not None and not target.exists():
            shutil.move(str(retired), str(target))
            retired = None
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if retired is not None:
            if retired.is_symlink() or retired.is_file():
                retired.unlink()
            else:
                shutil.rmtree(retired)
    return copy.deepcopy(manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay-check and export a promoted campaign to the viewer"
    )
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_VIEWER_DESTINATION,
    )
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = export_campaign_viewer(
        arguments.campaign_root,
        arguments.destination,
        replace=arguments.replace,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_VIEWER_DESTINATION",
    "export_campaign_viewer",
    "main",
]
