"""Clean proposal-only workspace construction for RoboArm GKM rounds."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..source_guard import materialize_public_sources

PROMOTED_SOURCE_FILES = ("legs.py", "players.py", "solve.py")

SCENARIO_CONTRACT_SOURCE = '''\
"""Public authoring contract for untrusted RoboArm scenario proposals.

This module only validates proposal shape.  It cannot observe or actuate the
RoboArm and it cannot issue a safety verdict.
"""

from __future__ import annotations

ACTIONS = (1, 2, 3, 4, 5, 6)
KINDS = ("experiment", "candidate")


def scenario(scenario_id, kind, hypothesis, expected_observation, actions):
    if (
        not isinstance(scenario_id, str)
        or not scenario_id
        or len(scenario_id) > 64
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in scenario_id
        )
    ):
        raise ValueError("invalid scenario_id")
    if kind not in KINDS:
        raise ValueError("kind must be experiment or candidate")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("hypothesis must be nonempty text")
    if (
        not isinstance(expected_observation, str)
        or not expected_observation.strip()
    ):
        raise ValueError("expected_observation must be nonempty text")
    if (
        not isinstance(actions, (list, tuple))
        or not actions
        or len(actions) > 160
        or any(
            isinstance(action, bool)
            or not isinstance(action, int)
            or action not in ACTIONS
            for action in actions
        )
    ):
        raise ValueError("actions must contain 1..160 public action IDs")
    return {
        "scenario_id": scenario_id,
        "kind": kind,
        "hypothesis": hypothesis.strip(),
        "expected_observation": expected_observation.strip(),
        "actions": [int(action) for action in actions],
    }


def proposal_bundle(evidence, scenarios):
    generation = evidence.get("generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation <= 0
    ):
        raise ValueError("evidence has no valid generation")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("propose() must return a nonempty scenario list")
    if len(scenarios) > 8:
        raise ValueError("one generation may propose at most 8 scenarios")
    identifiers = [item.get("scenario_id") for item in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("scenario_id values must be unique")
    required = {
        "scenario_id",
        "kind",
        "hypothesis",
        "expected_observation",
        "actions",
    }
    for item in scenarios:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("scenario fields differ from the closed contract")
        # Reconstruct to validate every field and strip object subclasses.
        checked = scenario(
            item["scenario_id"],
            item["kind"],
            item["hypothesis"],
            item["expected_observation"],
            item["actions"],
        )
        if checked != item:
            raise ValueError("scenario is not in canonical public form")
    return {
        "schema_version": 1,
        "kind": "roboarm_scenario_proposals",
        "game_id": "rb01-v1",
        "round_id": "rb01-round-1",
        "generation": generation,
        "scenarios": scenarios,
    }
'''

PERCEPTION_SOURCE = '''\
"""Generic RGB-camera and telemetry helpers over host-sealed evidence."""

from __future__ import annotations

import base64
from collections import deque

import numpy as np

FRAME_HEIGHT = 72
FRAME_WIDTH = 128
FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)


def decode_frame(frame_b64):
    raw = base64.b64decode(frame_b64.encode("ascii"), validate=True)
    if len(raw) != FRAME_HEIGHT * FRAME_WIDTH * 3:
        raise ValueError("expected a 128x72x3 RGB8 frame")
    return np.frombuffer(raw, dtype=np.uint8).reshape(FRAME_SHAPE).copy()


def initial_frame(evidence):
    return decode_frame(evidence["initial_observation"]["frame_b64"])


def initial_telemetry(evidence):
    packet = evidence["initial_observation"]["telemetry"]
    if not isinstance(packet, dict):
        raise ValueError("initial telemetry is not an object")
    return packet


def attempt_by_id(evidence, attempt_id):
    for attempt in evidence.get("attempts", []):
        if attempt.get("attempt_id") == attempt_id:
            return attempt
    raise KeyError(attempt_id)


def observed_frames(attempt, phase="preflight"):
    trace = attempt.get(phase)
    if not isinstance(trace, dict):
        return []
    return [
        decode_frame(step["frame_b64"])
        for step in trace.get("steps", [])
    ]


def observed_telemetry(attempt, phase="preflight"):
    trace = attempt.get(phase)
    if not isinstance(trace, dict):
        return []
    return [
        step["telemetry"]
        for step in trace.get("steps", [])
        if isinstance(step.get("telemetry"), dict)
    ]


def latest_frame(evidence):
    for attempt in reversed(evidence.get("attempts", [])):
        for phase in ("commit", "preflight"):
            frames = observed_frames(attempt, phase)
            if frames:
                return frames[-1]
    return initial_frame(evidence)


def latest_telemetry(evidence):
    for attempt in reversed(evidence.get("attempts", [])):
        for phase in ("commit", "preflight"):
            packets = observed_telemetry(attempt, phase)
            if packets:
                return packets[-1]
    return initial_telemetry(evidence)


def color_summary(frame, *, bin_size=32):
    """Return a generic quantized RGB histogram with no semantic labels."""
    data = np.asarray(frame, dtype=np.uint8)
    if data.shape != FRAME_SHAPE:
        raise ValueError("expected a 128x72x3 RGB frame")
    if bin_size not in (8, 16, 32, 64):
        raise ValueError("bin_size must be 8, 16, 32, or 64")
    quantized = data // bin_size
    colors, counts = np.unique(
        quantized.reshape((-1, 3)),
        axis=0,
        return_counts=True,
    )
    order = np.argsort(-counts)
    return [
        {
            "rgb_bin": tuple(int(value) for value in colors[index]),
            "pixels": int(counts[index]),
        }
        for index in order
    ]


def connected_components(frame, *, bin_size=32, min_pixels=2):
    """Return generic 4-connected components of quantized RGB pixels."""
    data = np.asarray(frame, dtype=np.uint8)
    if data.shape != FRAME_SHAPE:
        raise ValueError("expected a 128x72x3 RGB frame")
    quantized = data // bin_size
    labels = (
        quantized[:, :, 0].astype(np.int32) * 4096
        + quantized[:, :, 1].astype(np.int32) * 64
        + quantized[:, :, 2].astype(np.int32)
    )
    seen = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=bool)
    result = []
    for row in range(FRAME_HEIGHT):
        for column in range(FRAME_WIDTH):
            label = int(labels[row, column])
            if seen[row, column]:
                continue
            queue = deque([(row, column)])
            seen[row, column] = True
            cells = []
            while queue:
                current_row, current_column = queue.popleft()
                cells.append((current_row, current_column))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = current_row + dr, current_column + dc
                    if (
                        0 <= rr < FRAME_HEIGHT
                        and 0 <= cc < FRAME_WIDTH
                        and not seen[rr, cc]
                        and int(labels[rr, cc]) == label
                    ):
                        seen[rr, cc] = True
                        queue.append((rr, cc))
            if len(cells) < min_pixels:
                continue
            rows = [cell[0] for cell in cells]
            columns = [cell[1] for cell in cells]
            pixels = np.asarray(
                [data[cell[0], cell[1]] for cell in cells],
                dtype=np.float64,
            )
            result.append({
                "area": len(cells),
                "mean_rgb": tuple(
                    round(float(value), 1)
                    for value in pixels.mean(axis=0)
                ),
                "bbox": (min(columns), min(rows), max(columns), max(rows)),
                "centroid": (
                    round(sum(columns) / len(columns), 2),
                    round(sum(rows) / len(rows), 2),
                ),
            })
    return sorted(
        result,
        key=lambda item: (-item["area"], item["bbox"]),
    )


def frame_delta(before, after):
    first = np.asarray(before, dtype=np.uint8)
    second = np.asarray(after, dtype=np.uint8)
    if first.shape != FRAME_SHAPE or second.shape != FRAME_SHAPE:
        raise ValueError("expected two 128x72x3 RGB frames")
    absolute = np.abs(first.astype(np.int16) - second.astype(np.int16))
    changed = np.argwhere(np.any(absolute != 0, axis=2))
    if len(changed) == 0:
        return {"pixels": 0, "bbox": None, "mean_abs_rgb": (0.0, 0.0, 0.0)}
    return {
        "pixels": int(len(changed)),
        "bbox": (
            int(changed[:, 1].min()),
            int(changed[:, 0].min()),
            int(changed[:, 1].max()),
            int(changed[:, 0].max()),
        ),
        "mean_abs_rgb": tuple(
            round(float(value), 3)
            for value in absolute[np.any(absolute != 0, axis=2)].mean(axis=0)
        ),
    }


def compact_scene(frame):
    return connected_components(frame, bin_size=32, min_pixels=3)
'''

GKM_PROPOSE_SOURCE = '''\
"""Run retained source offline and emit untrusted declarative scenarios."""

from __future__ import annotations

import json
import os
import stat
import traceback
from pathlib import Path

from scenario_contract import proposal_bundle
from solve import propose


def safe_output(path, text):
    target = Path(path)
    if target.is_symlink():
        raise ValueError("proposal output cannot be a symlink")
    if target.exists():
        metadata = target.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("proposal output must be one unaliased file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o644)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("proposal output became aliased")
        raw = text.encode("utf-8")
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("proposal output write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    evidence = json.loads(Path("evidence.json").read_text(encoding="utf-8"))
    error = None
    try:
        scenarios = propose(evidence)
        bundle = proposal_bundle(evidence, scenarios)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        print("PROPOSER_SOURCE_ERROR", json.dumps({"error": error}))
        return 2
    payload = json.dumps(
        bundle,
        sort_keys=True,
        separators=(",", ":"),
    )
    safe_output("scenario_proposals.json", payload + "\\n")
    print("SCENARIO_PROPOSALS", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

LEGS_SEED = '''\
"""Retained reusable RoboArm hypothesis and scenario-construction skills.

Add a named leg only after host-sealed public observations support it.
"""
'''

PLAYERS_SEED = '''\
"""Thin per-round compositions of retained proposal legs."""

from legs import *  # noqa: F403


def propose_level_1(evidence):
    """Initial zero-seed state: no scenario hypothesis yet."""
    return []
'''

SOLVE_SEED = '''\
"""Stable dispatcher for retained RoboArm proposal source."""

from players import propose_level_1


def propose(evidence):
    return propose_level_1(evidence)
'''

ROUND_DESCRIPTION = """\
# rb01 Round 1

This is a scored ARC-style RoboArm manipulation round solved by a
Godel-Kolmogorov machine under a host-owned safety gate.

- The six action meanings and physical step sizes are fully disclosed.
- The current `evidence.json` contains exact host-sealed 128×72×3 RGB camera
  frames, synchronized public controller telemetry, action results, sparse
  `levels_completed`, terminal state, and safety-gate disposition from earlier
  generations.
- RGB pixels are a calibrated perspective camera observation. Controller
  telemetry is a separate packet; no HUD is painted into the image.
- Write hypotheses and declarative action scenarios. You have no live robot,
  simulator handle, connector socket, or actuation token.
- `experiment` scenarios always run in an isolated authoritative digital twin.
- `candidate` scenarios are also preflighted. Only a safe candidate that
  reaches the sparse goal, after an earlier observed failed hypothesis, may
  receive a one-use host commit permit.
- A rejected motion or failed grasp is real observed evidence. Revise the next
  generation from returned RGB frames and telemetry; do not invent
  observations.
- Clone/preflight success alone is never promotion evidence.

Object response, useful contact geometry, grasp conditions, clearance, release
behavior, and the exact success relation must be learned from the sealed public
evidence. There is no dense reward and no private-state query.
"""


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    root: Path
    transcript: Path
    last_message: Path


def _safe_write(path: Path, text: str) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked workspace output: {path.name}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)


def materialize_workspace(
    root: Path,
    *,
    write_root: Path,
    public_evidence: dict[str, object],
    generation: int,
    parent_source: Path | None = None,
) -> WorkspaceLayout:
    """Create one source-minimal, proposal-only generation workspace."""

    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation <= 0
        or public_evidence.get("generation") != generation
    ):
        raise ValueError("workspace evidence/generation binding is invalid")
    resolved_write_root = write_root.resolve(strict=True)
    prospective = root.resolve(strict=False)
    if not prospective.is_relative_to(resolved_write_root):
        raise ValueError("workspace escaped campaign write root")
    root.mkdir(parents=True, exist_ok=False)
    resolved = root.resolve(strict=True)
    if not resolved.is_relative_to(resolved_write_root):
        raise ValueError("workspace escaped campaign write root")

    materialize_public_sources(resolved, write_root=resolved_write_root)
    _safe_write(
        resolved / "scenario_contract.py",
        SCENARIO_CONTRACT_SOURCE,
    )
    _safe_write(resolved / "perception.py", PERCEPTION_SOURCE)
    _safe_write(resolved / "gkm_propose.py", GKM_PROPOSE_SOURCE)
    _safe_write(resolved / "ROUND.md", ROUND_DESCRIPTION)
    _safe_write(
        resolved / "evidence.json",
        json.dumps(public_evidence, sort_keys=True) + "\n",
    )

    if parent_source is None:
        _safe_write(resolved / "legs.py", LEGS_SEED)
        _safe_write(resolved / "players.py", PLAYERS_SEED)
        _safe_write(resolved / "solve.py", SOLVE_SEED)
    else:
        parent = parent_source.resolve(strict=True)
        if not parent.is_relative_to(resolved_write_root):
            raise ValueError("parent source escaped campaign write root")
        for name in PROMOTED_SOURCE_FILES:
            source = parent / name
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"missing clean parent source: {name}")
            shutil.copyfile(source, resolved / name)

    _safe_write(
        resolved / "solver_index.md",
        """\
# Proposal source index

- `evidence.json`: immutable host-sealed public observations from prior rounds.
- `legs.py`: retained reusable perception/hypothesis/scenario skills.
- `players.py`: `propose_level_1(evidence)` must thinly compose named legs.
- `solve.py`: stable `propose(evidence)` dispatcher.
- `perception.py`: generic frame decoding, components, deltas, and evidence lookup.
- `scenario_contract.py`: proposal authoring schema only; no execution or verdict.
- `gkm_propose.py`: offline harness that emits `scenario_proposals.json`.
- `ROUND.md`: sparse objective, generational feedback, and safety rules.

There is deliberately no `Arena`, connector client, socket, token, clone
handle, or direct `step()` capability in this workspace.
""",
    )
    return WorkspaceLayout(
        root=resolved,
        transcript=resolved / "proposer.jsonl",
        last_message=resolved / "proposer_last.md",
    )


__all__ = [
    "PROMOTED_SOURCE_FILES",
    "WorkspaceLayout",
    "materialize_workspace",
]
