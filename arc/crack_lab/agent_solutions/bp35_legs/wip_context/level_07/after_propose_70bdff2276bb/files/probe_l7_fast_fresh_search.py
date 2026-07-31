"""Parallel fresh-replay search with only one-step clone successors."""

import hashlib
import heapq
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from legs import COL_ANCHORS, ROW_ANCHORS
from probe_l7_fresh_graph import (
    available_actions,
    object_identity,
    signed_origin_delta,
)
from probe_l7_raw_search import SEED, avatar_position, target_path_distance


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def avatar_world_row(frame, origin):
    avatar = avatar_position(frame)
    if avatar is None:
        return None
    ai = min(range(10), key=lambda row: abs(ROW_ANCHORS[row] - avatar[1]))
    return origin + ai


def node_summary(
    env,
    origin,
    selected,
    top_entries=0,
    target_returns=0,
    empty_returns=0,
    tick=0,
    camera_delta=0,
):
    if env.terminal() or avatar_position(env.frame()) is None:
        return {
            "terminal": True,
            "level": int(env.levels_completed),
        }
    frame = np.asarray(env.frame())
    avatar = avatar_position(frame)
    ai = min(range(10), key=lambda row: abs(ROW_ANCHORS[row] - avatar[1]))
    aj = min(
        range(8), key=lambda column: abs(COL_ANCHORS[column] - avatar[0])
    )
    return {
        "terminal": False,
        "level": int(env.levels_completed),
        "frame": hashlib.blake2b(
            frame[:63].tobytes(), digest_size=16
        ).hexdigest(),
        "origin": origin,
        "selected": selected,
        "avatar": [ai, aj],
        "world_row": origin + ai,
        "distance": target_path_distance(frame),
        "target": bool(np.any(frame[:63] == 7)),
        "top_entries": top_entries,
        "target_returns": target_returns,
        "empty_returns": empty_returns,
        "tick": tick,
        "camera_delta": camera_delta,
    }


def worker(path):
    result = {}

    def program(env):
        with open("checkpoint.json") as stream:
            prefix = json.load(stream)["final_path"]
        for action in prefix:
            env.step(action)
        base_level = int(env.levels_completed)
        origin = 0
        selected = None
        for action in SEED:
            if env.terminal():
                break
            before = np.asarray(env.frame()).copy()
            identity = object_identity(before, origin, action)
            env.step(*action)
            if action[0] == 6:
                selected = (
                    list(identity) if identity is not None else ["empty"]
                )
            if env.terminal() or env.levels_completed > base_level:
                break
            origin += signed_origin_delta(before, env.frame())

        previous_row = (
            None if env.terminal() else avatar_world_row(env.frame(), origin)
        )
        previous_target = (
            False
            if env.terminal()
            else bool(np.any(np.asarray(env.frame())[:63] == 7))
        )
        top_entries = 0
        target_returns = 0
        empty_returns = 0
        last_delta = 0
        for action in path:
            if env.terminal():
                break
            before = np.asarray(env.frame()).copy()
            identity = object_identity(before, origin, action)
            env.step(*action)
            if action[0] == 6:
                selected = (
                    list(identity) if identity is not None else ["empty"]
                )
            if env.terminal() or env.levels_completed > base_level:
                break
            last_delta = signed_origin_delta(before, env.frame())
            origin += last_delta
            current_row = avatar_world_row(env.frame(), origin)
            current_target = bool(
                np.any(np.asarray(env.frame())[:63] == 7)
            )
            if (
                previous_row is not None
                and current_row is not None
                and previous_row > 2
                and current_row <= 2
            ):
                top_entries += 1
            if (
                current_target
                and not previous_target
                and target_returns < top_entries
            ):
                target_returns += 1
                if selected and selected[0] == "empty":
                    empty_returns += 1
            previous_row = current_row
            previous_target = current_target

        result["parent"] = node_summary(
            env,
            origin,
            selected,
            top_entries,
            target_returns,
            empty_returns,
            len(path) % 6,
            last_delta,
        )
        result["base_level"] = base_level
        if (
            env.terminal()
            or env.levels_completed > base_level
            or avatar_position(env.frame()) is None
        ):
            result["children"] = []
            return

        before = np.asarray(env.frame()).copy()
        children = []
        actions = list(
            dict.fromkeys([*available_actions(before), (6, 9, 3)])
        )
        for action in actions:
            child = env.clone()
            child.step(*action)
            identity = object_identity(before, origin, action)
            child_selected = (
                (
                    list(identity)
                    if identity is not None
                    else ["empty"]
                )
                if action[0] == 6
                else selected
            )
            child_origin = origin
            child_delta = 0
            if not child.terminal():
                child_delta = signed_origin_delta(before, child.frame())
                child_origin += child_delta
            child_top_entries = top_entries
            child_target_returns = target_returns
            child_empty_returns = empty_returns
            child_row = (
                None
                if child.terminal()
                else avatar_world_row(child.frame(), child_origin)
            )
            child_target = (
                False
                if child.terminal()
                else bool(np.any(np.asarray(child.frame())[:63] == 7))
            )
            if (
                previous_row is not None
                and child_row is not None
                and previous_row > 2
                and child_row <= 2
            ):
                child_top_entries += 1
            if (
                child_target
                and not previous_target
                and child_target_returns < child_top_entries
            ):
                child_target_returns += 1
                if child_selected and child_selected[0] == "empty":
                    child_empty_returns += 1
            summary = node_summary(
                child,
                child_origin,
                child_selected,
                child_top_entries,
                child_target_returns,
                child_empty_returns,
                (len(path) + 1) % 6,
                child_delta,
            )
            summary["action"] = list(action)
            children.append(summary)
        result["children"] = children

    levels, replay, error = arena.run_program("bp35", program)
    result["runner"] = [levels, len(replay), error]
    print(json.dumps(result, separators=(",", ":")), flush=True)


def evaluate_many(paths, workers=4):
    script = os.path.abspath(__file__)
    process_env = dict(os.environ)
    process_env["MPLCONFIGDIR"] = "/tmp/mpl-l7-fast-fresh"
    out = []
    for start in range(0, len(paths), workers):
        batch = paths[start:start + workers]
        processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    script,
                    "--worker",
                    json.dumps(path, separators=(",", ":")),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=process_env,
            )
            for path in batch
        ]
        for path, process in zip(batch, processes):
            stdout, _ = process.communicate()
            line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
            out.append(
                (
                    path,
                    json.loads(line)
                    if line
                    else {"runner": [0, 0, "empty worker"]},
                )
            )
    return out


def priority(path, state):
    distance = state.get("distance")
    world_row = state.get("world_row", -99)
    avatar = state.get("avatar") or [0, 0]
    column = avatar[1]
    if distance is not None and distance < 18:
        return 0, distance, len(path), abs(column - 1)
    if world_row >= 9:
        return 1, len(path), -world_row, abs(column - 1)
    returns = state.get("target_returns", 0)
    entries = state.get("top_entries", 0)
    selected = state.get("selected") or []
    support_selected = bool(selected and selected[0] == "support")
    empty_selected = bool(selected and selected[0] == "empty")
    transition = 0 if state.get("camera_delta") else 1
    if state.get("empty_returns", 0):
        return 2, transition, len(path), abs(column - 6)
    if returns and empty_selected:
        return 3, transition, len(path), abs(column - 6)
    if entries and empty_selected:
        return 4, transition, len(path), abs(column - 6)
    if returns and support_selected:
        return 5, transition, len(path), abs(column - 6)
    if entries and support_selected:
        return 6, transition, len(path), abs(column - 6)
    if returns:
        return 7, len(path), abs(column - 6)
    if entries:
        return 8, len(path), abs(column - 6)
    if world_row <= 2 or 5 <= world_row <= 8:
        return 9, len(path), abs(column - 6)
    if distance is not None:
        return 10, len(path), distance
    return 11, len(path), abs(column - 6)


def state_key(state):
    return (
        state.get("frame"),
        state.get("origin"),
        tuple(state.get("selected") or ()),
        state.get("top_entries", 0) % 3,
        state.get("target_returns", 0) % 3,
        state.get("empty_returns", 0) % 3,
        state.get("tick", 0),
    )


def search():
    max_expansions = int(os.environ.get("MAX_EXPANSIONS", "500"))
    max_depth = int(os.environ.get("MAX_DEPTH", "28"))
    workers = int(os.environ.get("WORKERS", "4"))
    if os.environ.get("START_MODE") == "empty_return":
        start_paths = [[
            [7], [7], [6, 3, 47], [4], [4], [6, 9, 3],
            [7], [7], [7], [7],
        ]]
    elif os.environ.get("START_MODE") in (
        "staged_supports",
        "staged_pairs",
        "upper_pairs",
        "pair_tops",
        "pair_returns",
    ):
        stage = [
            [7], [7], [6, 3, 47], [4], [4], [3],
        ]
        cycle = [[7], [7], [7], [4], [7], [7], [7]]
        supports = (
            (45, 21), (39, 39), (39, 45), (39, 51), (39, 57)
        )
        if os.environ.get("START_MODE") in (
            "staged_pairs",
            "upper_pairs",
            "pair_tops",
            "pair_returns",
        ):
            start_paths = [
                [
                    *stage,
                    [6, x1, y1],
                    [6, x2, y2],
                    *(
                        []
                        if os.environ.get("START_MODE") == "upper_pairs"
                        else (
                            (
                                [[7], [7], [7]]
                                if os.environ.get("START_MODE") == "pair_tops"
                                else [[7], [7], [7], [7], [7], [7]]
                            )
                            if os.environ.get("START_MODE") in (
                                "pair_tops", "pair_returns"
                            )
                            else cycle
                        )
                    ),
                ]
                for x1, y1 in supports
                for x2, y2 in supports
                if (x1, y1) != (x2, y2)
            ]
        else:
            start_paths = [
                [*stage, [6, x, y], *cycle]
                for x, y in supports
            ]
    else:
        start_paths = [[[7], [7]]]
    roots = [
        (path, result["parent"])
        for path, result in evaluate_many(
            start_paths, workers=min(workers, len(start_paths))
        )
        if result.get("parent", {}).get("frame") is not None
    ]
    if not roots:
        print("FAST_FRESH_NO_ROOTS", flush=True)
        return
    counter = itertools.count()
    queue = [
        (priority(path, root), next(counter), path, root)
        for path, root in roots
    ]
    heapq.heapify(queue)
    seen = {state_key(root) for _, root in roots}
    expanded = 0
    best = min(
        (priority(path, root), path, root) for path, root in roots
    )
    print(
        "FAST_FRESH_ROOTS",
        len(roots),
        len(seen),
        best[0],
        flush=True,
    )
    started = time.monotonic()

    while queue and expanded < max_expansions:
        batch = []
        while queue and len(batch) < workers:
            _, _, path, _ = heapq.heappop(queue)
            if len(path) < max_depth:
                batch.append(path)
        if not batch:
            break
        for path, result in evaluate_many(batch, workers=workers):
            expanded += 1
            parent = result.get("parent", {})
            if parent.get("level", 0) > 6:
                print("FAST_FRESH_WIN", path, expanded, flush=True)
                return
            if parent.get("terminal"):
                continue
            for child in result.get("children", []):
                child_path = [*path, child["action"]]
                if child.get("level", 0) > 6:
                    verification = evaluate_many([child_path], workers=1)[0][1]
                    if verification.get("parent", {}).get("level", 0) > 6:
                        print(
                            "FAST_FRESH_WIN",
                            child_path,
                            expanded,
                            flush=True,
                        )
                        return
                if child.get("terminal") or child.get("frame") is None:
                    continue
                key = state_key(child)
                if key in seen:
                    continue
                seen.add(key)
                child_priority = priority(child_path, child)
                if child_priority < best[0]:
                    best = (child_priority, child_path, child)
                    print(
                        "FAST_FRESH_PROGRESS",
                        expanded,
                        child_priority,
                        child.get("avatar"),
                        child.get("world_row"),
                        child.get("distance"),
                        child_path,
                        round(time.monotonic() - started, 1),
                        flush=True,
                    )
                heapq.heappush(
                    queue,
                    (
                        child_priority,
                        next(counter),
                        child_path,
                        child,
                    ),
                )
            if expanded >= max_expansions:
                break
        if expanded and expanded % 20 < workers:
            print(
                "FAST_FRESH_SEARCH",
                expanded,
                len(queue),
                len(seen),
                best[0],
                round(time.monotonic() - started, 1),
                flush=True,
            )
    print(
        "FAST_FRESH_DONE",
        expanded,
        len(queue),
        len(seen),
        best,
        flush=True,
    )


if __name__ == "__main__" and "--worker" in sys.argv:
    worker(json.loads(sys.argv[2]))
elif __name__ == "__main__":
    search()
