"""Fresh-process graph search for level 7's selected-object gravity maze."""

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
import gkm_legs as G

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action,
)
from perception import connected_components
from probe_l7_raw_search import SEED, avatar_position, target_path_distance


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


GATE_ROOT = [
    (6, 27, 15), (6, 27, 27), (6, 39, 27), (6, 33, 9),
    (4,), (4,), (4,), (6, 39, 51), (6, 3, 3), (4,),
]
STATIC = {3: "#", 5: "#", 10: ".", 0: "v"}


def lattice(frame):
    return [
        [STATIC.get(int(frame[y][x])) for x in COL_ANCHORS]
        for y in ROW_ANCHORS
    ]


def alignment(world, frame_grid, previous):
    candidates = []
    for origin in range(previous - 10, previous + 11):
        matches = mismatches = overlap = 0
        for i, row in enumerate(frame_grid):
            for j, value in enumerate(row):
                known = world.get((origin + i, j))
                if value is None or known is None:
                    continue
                overlap += 1
                if value == known:
                    matches += 1
                else:
                    mismatches += 1
        candidates.append((
            matches - 3 * mismatches, matches, overlap,
            -abs(origin - previous), -abs(origin), origin,
        ))
    return max(candidates)[-1]


def merge(world, frame_grid, origin):
    for i, row in enumerate(frame_grid):
        for j, value in enumerate(row):
            if value is not None:
                world.setdefault((origin + i, j), value)


def signed_origin_delta(before, after):
    """Camera-origin change from adjacent raw frames, including descents."""
    old = [
        tuple(int(value) for value in before[row])
        for row in range(63)
    ]
    new = [
        tuple(int(value) for value in after[row])
        for row in range(63)
    ]
    candidates = []
    for bands in range(-9, 10):
        pixels = 6 * bands
        start = max(0, -pixels)
        stop = min(63, 63 - pixels)
        hits = sum(
            new[row] == old[row + pixels]
            for row in range(start, stop)
        )
        candidates.append((hits, -abs(bands), bands))
    return max(candidates)[-1]


def object_identity(frame, origin, action):
    if action[0] != 6:
        return None
    _, x, y = action
    i = min(range(10), key=lambda row: abs(ROW_ANCHORS[row] - y))
    if x <= 5:
        color = int(frame[y][x]) if 0 <= y < 63 else -1
        return ("gravity", origin + i) if color == 8 else None
    j = min(range(8), key=lambda col: abs(COL_ANCHORS[col] - x))
    color = int(frame[y][x]) if 0 <= y < 63 and 0 <= x < 64 else -1
    if color in (12, 14):
        return ("support", origin + i, j)
    return None


def available_actions(frame):
    avatar = avatar_position(frame)
    if avatar is None:
        return []
    ax, ay = avatar
    ai = min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - ay))
    aj = min(range(8), key=lambda j: abs(COL_ANCHORS[j] - ax))
    out = [(3,), (4,), (7,)]
    crosses = [
        blob
        for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[1] <= 5 and blob.bbox[0] < 63
    ]
    crosses.sort(key=lambda blob: abs(blob.centroid[0] - ay))
    chosen = []
    for blob in (
        crosses[:1]
        + ([] if not crosses else [min(crosses, key=lambda b: b.centroid[0])])
        + ([] if not crosses else [max(crosses, key=lambda b: b.centroid[0])])
    ):
        y, x = blob.centroid
        action = (6, int(round(x)), int(round(y)))
        if action not in chosen:
            chosen.append(action)
    out.extend(chosen)
    row_radius = 5 if os.environ.get("WIDE_SUPPORTS") == "1" else 3
    col_radius = 3 if os.environ.get("WIDE_SUPPORTS") == "1" else 1
    for i in range(max(0, ai - row_radius), min(10, ai + row_radius + 1)):
        for j in range(max(0, aj - col_radius), min(8, aj + col_radius + 1)):
            if _cell_shape(frame, i, j)[0] in (12, 14):
                out.append(click_action(i, j))
    return list(dict.fromkeys(out))


def worker(path):
    observation = {}

    def program(env):
        with open("checkpoint.json") as stream:
            checkpoint = json.load(stream)
        for action in checkpoint["final_path"]:
            env.step(action)
        base_level = int(env.levels_completed)
        world = {}
        origin = 0
        merge(world, lattice(env.frame()), origin)
        selected = None
        root_route = SEED if os.environ.get("ROOT_MODE") == "seed" else GATE_ROOT
        for action in [*root_route, *path]:
            if env.terminal():
                break
            before = np.asarray(env.frame()).copy()
            identity = object_identity(before, origin, action)
            env.step(*action)
            if identity is not None:
                selected = identity
            if env.levels_completed > base_level:
                observation["win"] = True
                observation["origin"] = origin
                return
            if env.terminal():
                break
            if os.environ.get("SIGNED_ORIGIN") == "1":
                origin += signed_origin_delta(before, env.frame())
            else:
                origin = alignment(world, lattice(env.frame()), origin)
            merge(world, lattice(env.frame()), origin)
        observation["win"] = False
        observation["terminal"] = bool(env.terminal())
        observation["origin"] = origin
        observation["selected"] = selected
        if env.terminal():
            return
        frame = np.asarray(env.frame())
        avatar = avatar_position(frame)
        ai = min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - avatar[1]))
        aj = min(range(8), key=lambda j: abs(COL_ANCHORS[j] - avatar[0]))
        observation.update(
            frame=hashlib.blake2b(
                frame[:63].tobytes(), digest_size=16
            ).hexdigest(),
            avatar=(ai, aj),
            world_row=origin + ai,
            target_distance=target_path_distance(frame),
            target=bool(np.any(frame[:63] == 7)),
            actions=available_actions(frame),
        )
        if os.environ.get("TRACE_WORLD") == "1":
            rows = sorted({row for row, _ in world})
            observation["world"] = [
                [
                    row,
                    "".join(world.get((row, column), "?") for column in range(8)),
                ]
                for row in rows
            ]
            observation["objects"] = [
                [
                    blob.color,
                    origin + min(
                        range(10),
                        key=lambda row: abs(
                            ROW_ANCHORS[row] - blob.centroid[0]
                        ),
                    ),
                    min(
                        range(8),
                        key=lambda column: abs(
                            COL_ANCHORS[column] - blob.centroid[1]
                        ),
                    ),
                    blob.area,
                ]
                for blob in connected_components(
                    frame, colors=(7, 12, 14, 15), min_area=2
                )
                if blob.bbox[0] < 63
            ]

    levels, replay, error = arena.run_program("bp35", program)
    observation["levels"] = levels
    observation["moves"] = len(replay)
    observation["error"] = error
    print(json.dumps(observation, separators=(",", ":")), flush=True)


def evaluate_many(paths, workers=4):
    results = []
    script = os.path.abspath(__file__)
    process_env = dict(os.environ)
    process_env["MPLCONFIGDIR"] = "/tmp/mpl-l7-direct"
    for start in range(0, len(paths), workers):
        batch = paths[start:start + workers]
        processes = [
            subprocess.Popen(
                [
                    sys.executable, script, "--worker",
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
            if not line:
                results.append((path, {"error": "worker produced no result"}))
                continue
            results.append((path, json.loads(line)))
    return results


def priority(path, observation):
    avatar = observation["avatar"]
    gap = abs(12 - observation["world_row"])
    column_gap = abs(1 - avatar[1])
    distance = observation["target_distance"]
    if os.environ.get("PRIORITY_MODE") == "topology":
        world_row = observation["world_row"]
        column = avatar[1]
        if distance is not None and distance < 18:
            return 0, distance, len(path)
        if world_row >= 9:
            return 1, -world_row, column_gap, len(path)
        if world_row <= 2 and column >= 5:
            return 2, abs(column - 6), len(path)
        if 5 <= world_row <= 8:
            return 3, abs(column - 6), len(path)
        if distance is not None:
            return 4, distance, column_gap, len(path)
        return 5, gap, column_gap, len(path)
    escape_bonus = (
        4 if path and path[-1] and path[-1][0] == 7 else 0
    )
    return (
        max(0, gap - escape_bonus),
        30 if distance is None else distance,
        column_gap,
        len(path),
    )


def search():
    max_evaluations = int(os.environ.get("MAX_EVALUATIONS", "240"))
    max_depth = int(os.environ.get("MAX_DEPTH", "24"))
    if os.environ.get("ROOT_PATH"):
        root_path = json.loads(os.environ["ROOT_PATH"])
    elif os.environ.get("START_LEFT") == "1":
        root_path = [[3], [3], [3], [3]]
    elif os.environ.get("START_RIGHT") == "1":
        root_path = [[6, 3, 3], [4]]
    else:
        root_path = []
    root = evaluate_many([root_path])[0][1]
    root_route = SEED if os.environ.get("ROOT_MODE") == "seed" else GATE_ROOT
    if root.get("win"):
        print("FRESH_GRAPH_WIN", root_route, flush=True)
        return
    counter = itertools.count()
    frontier = [(priority(root_path, root), next(counter), root_path, root)]
    seen = {
        (root.get("frame"), tuple(root.get("selected") or ()), root["origin"])
    }
    evaluated = 1
    expanded = 0
    best = (priority(root_path, root), root_path, root)
    started = time.monotonic()
    while frontier and evaluated < max_evaluations:
        _, _, path, observation = heapq.heappop(frontier)
        expanded += 1
        if len(path) >= max_depth:
            continue
        children = [
            [*path, action] for action in observation.get("actions", [])
        ]
        children = children[:max_evaluations - evaluated]
        for child_path, child in evaluate_many(children):
            evaluated += 1
            if os.environ.get("TRACE_CHILDREN") == "1":
                print(
                    "FRESH_GRAPH_CHILD",
                    child_path[-1],
                    {
                        key: child.get(key)
                        for key in (
                            "win", "terminal", "origin", "selected",
                            "avatar", "world_row", "target_distance",
                            "target", "actions",
                        )
                    },
                    flush=True,
                )
            if child.get("win") or child.get("levels", 0) > 6:
                route = [*root_route, *child_path]
                print(
                    "FRESH_GRAPH_WIN", evaluated, expanded, route,
                    flush=True,
                )
                return
            if (
                child.get("error")
                or child.get("terminal")
                or child.get("frame") is None
            ):
                continue
            key = (
                child["frame"], tuple(child.get("selected") or ()),
                child["origin"],
            )
            if key in seen:
                continue
            seen.add(key)
            child_priority = priority(child_path, child)
            if child_priority < best[0]:
                best = (child_priority, child_path, child)
                print(
                    "FRESH_GRAPH_PROGRESS", evaluated, expanded,
                    child_priority, child["avatar"], child["world_row"],
                    child["target_distance"], child_path,
                    round(time.monotonic() - started, 1), flush=True,
                )
            heapq.heappush(
                frontier,
                (
                    child_priority, next(counter), child_path, child
                ),
            )
        if expanded % 5 == 0:
            print(
                "FRESH_GRAPH_SEARCH", evaluated, expanded, len(frontier),
                best[0], round(time.monotonic() - started, 1), flush=True,
            )
    print(
        "FRESH_GRAPH_DONE", evaluated, expanded, len(frontier),
        best, flush=True,
    )


if __name__ == "__main__" and "--worker" in sys.argv:
    worker(json.loads(sys.argv[2]))
elif __name__ == "__main__":
    search()
