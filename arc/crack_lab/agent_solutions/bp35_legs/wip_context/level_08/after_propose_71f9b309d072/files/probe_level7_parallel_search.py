import hashlib
import heapq
import itertools
import json
import os
import subprocess
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    GRID_COLS,
    GRID_ROWS,
    ROW_ANCHORS,
    _cell_shape,
    click_action,
    moves_used,
)
from perception import arr
from probe_level7_no_control import PREFIX, advance, avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


with open("checkpoint.json") as stream:
    CHECKPOINT_PATH = json.load(stream)["final_path"]

ROOT_PATH = tuple(PREFIX)


def choices(frame):
    avatar = avatar_cell(frame)
    actions = [(3,), (4,)]
    actions.extend((6, 3, y) for y in controls(frame))
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            color, _ = _cell_shape(frame, i, j)
            if (
                color in (12, 14)
                and avatar is not None
                and abs(i - avatar[0]) <= 2
                and abs(j - avatar[1]) <= 1
            ):
                actions.append(click_action(i, j))
            elif (
                color == 15
                and avatar is not None
                and abs(i - avatar[0]) <= 1
                and abs(j - avatar[1]) <= 1
            ):
                actions.append(click_action(i, j))
    return tuple(dict.fromkeys(actions))


def evaluate(path):
    observation = {}

    def probe(env):
        for action in CHECKPOINT_PATH:
            env.step(action)
        height = advance(env, (*ROOT_PATH, *path))
        if env.levels_completed > 6:
            observation.update(win=True, height=height)
            return
        if env.terminal() or avatar_cell(env.frame()) is None:
            observation.update(win=False, dead=True, height=height)
            return
        frame = arr(env.frame())
        shape_cells = tuple(
            _cell_shape(frame, i, j)
            for i in range(GRID_ROWS)
            for j in range(GRID_COLS)
        )
        expanded = sum(
            color in (12, 14) and area > 8 for color, area in shape_cells
        )
        observation.update(
            win=False,
            dead=False,
            height=height,
            key=(
                height,
                hashlib.blake2b(
                    frame[:63].tobytes(), digest_size=16
                ).hexdigest(),
                moves_used(frame) % 2,
            ),
            avatar=avatar_cell(frame),
            controls=controls(frame),
            expanded=expanded,
            actions=choices(frame),
        )

    A.run_program("bp35", probe)
    return path, observation


def worker_main():
    for line in sys.stdin:
        try:
            path = tuple(tuple(action) for action in json.loads(line))
            child_path, child = evaluate(path)
            payload = {
                "path": [list(action) for action in child_path],
                "observation": child,
            }
            print(json.dumps(payload, separators=(",", ":")), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}"},
                    separators=(",", ":"),
                ),
                flush=True,
            )


class ReplayPool:
    def __init__(self, workers):
        command = [sys.executable, os.path.abspath(__file__), "--worker"]
        self.processes = [
            subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            for _ in range(workers)
        ]

    def map(self, paths):
        results = []
        for start in range(0, len(paths), len(self.processes)):
            batch = paths[start:start + len(self.processes)]
            for process, path in zip(self.processes, batch):
                process.stdin.write(
                    json.dumps([list(action) for action in path]) + "\n"
                )
                process.stdin.flush()
            for process in self.processes[:len(batch)]:
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError("replay worker exited without a result")
                payload = json.loads(line)
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                child_path = tuple(
                    tuple(action) for action in payload["path"]
                )
                child = payload["observation"]
                if "key" in child:
                    child["key"] = tuple(child["key"])
                if "avatar" in child and child["avatar"] is not None:
                    child["avatar"] = tuple(child["avatar"])
                if "controls" in child:
                    child["controls"] = tuple(child["controls"])
                if "actions" in child:
                    child["actions"] = tuple(
                        tuple(action) for action in child["actions"]
                    )
                results.append((child_path, child))
        return results

    def close(self):
        for process in self.processes:
            if process.stdin:
                process.stdin.close()
        for process in self.processes:
            process.wait()


def score(obs, depth):
    avatar = obs["avatar"]
    central = min(avatar[1], 7 - avatar[1], 3)
    return (
        obs["height"] * 100
        + len(obs["controls"]) * 12
        + obs["expanded"]
        + central
        - depth
    )


def search(max_evaluations=900, max_depth=28, workers=8):
    _, root = evaluate(())
    if root.get("win"):
        return ()
    seen = {root["key"]}
    counter = itertools.count()
    frontier = [
        (-score(root, 0), 0, next(counter), (), root),
    ]
    evaluated = 1
    expanded = 0
    best = (root["height"], len(root["controls"]), (), root)
    pool = ReplayPool(workers)
    try:
        while frontier and evaluated < max_evaluations:
            _, _, _, path, obs = heapq.heappop(frontier)
            expanded += 1
            if len(path) >= max_depth:
                continue
            candidates = [(*path, action) for action in obs["actions"]]
            room = max_evaluations - evaluated
            candidates = candidates[:room]
            for child_path, child in pool.map(candidates):
                evaluated += 1
                if child.get("win"):
                    print(
                        "FOUND", evaluated, expanded, child["height"],
                        child_path, flush=True,
                    )
                    return child_path
                if child.get("dead") or child["key"] in seen:
                    continue
                seen.add(child["key"])
                progress = (child["height"], len(child["controls"]))
                if progress > best[:2]:
                    best = (*progress, child_path, child)
                    print(
                        "PROGRESS", evaluated, expanded, progress,
                        "depth", len(child_path), "avatar", child["avatar"],
                        "controls", child["controls"], "path", child_path,
                        flush=True,
                    )
                heapq.heappush(
                    frontier,
                    (
                        -score(child, len(child_path)),
                        len(child_path),
                        next(counter),
                        child_path,
                        child,
                    ),
                )
            if expanded % 10 == 0:
                print(
                    "STATUS", evaluated, expanded, len(frontier),
                    "BEST", best[:2], flush=True,
                )
    finally:
        pool.close()
    print(
        "EXHAUSTED", evaluated, expanded, len(frontier),
        "BEST", best[:3], flush=True,
    )
    return ()


if __name__ == "__main__" and "--worker" in sys.argv:
    worker_main()
elif __name__ == "__main__":
    route = search()
    print("ROUTE", route)
    if route:
        print("WIN", [*ROOT_PATH, *route])
