"""Search horizontal landing plus gravity-control macros from decode root 8."""

import hashlib
import heapq
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import moves_used, run_actions
from probe_l7_decode_matrix import controls, target
from probe_l7_fresh_graph import signed_origin_delta
from probe_l7_root8_local import decoded_route
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)


def digest(frame):
    return hashlib.blake2b(
        np.asarray(frame)[:63].tobytes(), digest_size=12
    ).digest()


def horizontal_landings(node, base_level):
    landings = [(node.clone(), [])]
    for direction in (LEFT, RIGHT):
        walked = node.clone()
        path = []
        previous = None
        unchanged = 0
        for _ in range(7):
            before = np.asarray(walked.frame()).copy()
            walked.step(*direction)
            path.append(direction)
            if walked.levels_completed > base_level:
                return landings, (walked, list(path))
            if walked.terminal() or avatar_cell(walked.frame()) is None:
                break
            terrain = digest(walked.frame())
            landings.append((walked.clone(), list(path)))
            if terrain == previous:
                unchanged += 1
            else:
                unchanged = 0
            previous = terrain
            if unchanged >= 1 and np.array_equal(
                before[:63], np.asarray(walked.frame())[:63]
            ):
                break
    unique = {}
    for landing, path in landings:
        unique.setdefault(digest(landing.frame()), (landing, path))
    return list(unique.values()), None


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    run_actions(env, decoded_route())
    staged_action = None
    if os.environ.get("STAGE_ACTION"):
        staged_action = tuple(json.loads(os.environ["STAGE_ACTION"]))
        env.step(*staged_action)
    base_level = int(env.levels_completed)
    root = env.clone()
    max_states = int(os.environ.get("MAX_STATES", "240"))
    max_macros = int(os.environ.get("MAX_MACROS", "12"))
    counter = itertools.count()

    def key(node, origin):
        frame = node.frame()
        return digest(frame), moves_used(frame) % 2, origin

    def score(node, macro_depth, origin):
        frame = node.frame()
        prize = target(frame)
        avatar = avatar_cell(frame)
        return (
            0 if prize is not None else 1,
            -origin,
            -len(controls(frame)),
            99 if avatar is None else -avatar[1],
            macro_depth,
        )

    frontier = [(score(root, 0, 0), next(counter), root, (), 0, 0)]
    seen = {key(root, 0)}
    evaluated = 0
    expanded = 0
    best = frontier[0][0]
    started = time.monotonic()
    print(
        "ROOT8_MACRO_ROOT", avatar_cell(root.frame()),
        target(root.frame()), tuple(controls(root.frame())),
        staged_action, lattice(root.frame()), flush=True,
    )
    while frontier and evaluated < max_states:
        _priority, _tie, node, path, origin, macro_depth = heapq.heappop(
            frontier
        )
        if macro_depth >= max_macros:
            continue
        expanded += 1
        before_node = np.asarray(node.frame()).copy()
        landings, movement_win = horizontal_landings(node, base_level)
        if movement_win is not None:
            _winner, suffix = movement_win
            route = (*path, *suffix)
            print("ROOT8_MACRO_WIN", route, flush=True)
            print(
                "ROOT8_MACRO_ROUTE",
                [
                    *decoded_route(),
                    *([] if staged_action is None else [staged_action]),
                    *route,
                ],
                flush=True,
            )
            return
        for landing, walk in landings:
            for control in controls(landing.frame()):
                child = landing.clone()
                before = np.asarray(child.frame()).copy()
                child.step(*control)
                evaluated += 1
                child_path = (*path, *walk, control)
                if child.levels_completed > base_level:
                    print(
                        "ROOT8_MACRO_WIN", child_path, evaluated, expanded,
                        flush=True,
                    )
                    print(
                        "ROOT8_MACRO_ROUTE",
                        [
                            *decoded_route(),
                            *(
                                []
                                if staged_action is None
                                else [staged_action]
                            ),
                            *child_path,
                        ],
                        flush=True,
                    )
                    return
                if child.terminal() or avatar_cell(child.frame()) is None:
                    continue
                child_origin = (
                    origin
                    + signed_origin_delta(before_node, before)
                    + signed_origin_delta(before, child.frame())
                )
                child_key = key(child, child_origin)
                if child_key in seen:
                    continue
                seen.add(child_key)
                child_score = score(
                    child, macro_depth + 1, child_origin
                )
                if child_score < best:
                    best = child_score
                    print(
                        "ROOT8_MACRO_PROGRESS", evaluated, expanded,
                        child_score, avatar_cell(child.frame()),
                        target(child.frame()), tuple(controls(child.frame())),
                        child_path, lattice(child.frame()), flush=True,
                    )
                heapq.heappush(
                    frontier,
                    (
                        child_score, next(counter), child, child_path,
                        child_origin, macro_depth + 1,
                    ),
                )
        if evaluated and evaluated % 40 < 12:
            print(
                "ROOT8_MACRO_SEARCH", evaluated, expanded,
                len(frontier), len(seen), best,
                round(time.monotonic() - started, 1), flush=True,
            )
    print(
        "ROOT8_MACRO_DONE", evaluated, expanded, len(frontier),
        len(seen), best, flush=True,
    )


arena.run_program("bp35", probe)
