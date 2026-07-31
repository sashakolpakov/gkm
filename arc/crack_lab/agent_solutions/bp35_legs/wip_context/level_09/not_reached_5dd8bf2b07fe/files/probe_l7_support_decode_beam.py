"""Decode ambiguous support clicks against every actual support on that row."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, _cell_shape, click_action
from probe_l7_decode_matrix import controls, target
from probe_level7_coordinate_decode import AMBIGUOUS, EARLY_SHIFTED
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)


def normalized_route(raw):
    route = []
    for step, item in enumerate(raw, 1):
        action = (item,) if isinstance(item, int) else tuple(item)
        if step in EARLY_SHIFTED and len(action) == 3 and action[1] != 3:
            action = action[0], action[1] + 12, action[2]
        route.append(action)
    return route


def support_candidates(frame, action):
    candidates = [action]
    _kind, raw_x, y = action
    row = min(range(10), key=lambda i: abs(3 + 6 * i - y))
    for x in COL_ANCHORS:
        column = (x - 15) // 6
        if _cell_shape(frame, row, column)[0] in (12, 14):
            candidates.append((6, x, y))
    for x in (raw_x - 6, raw_x + 6, raw_x + 12):
        if 0 <= x < 64:
            candidates.append((6, x, y))
    return list(dict.fromkeys(candidates))


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw = json.load(stream)["staged_prefix_actions"]
    base_level = int(env.levels_completed)
    route = normalized_route(raw)
    branches = [(env.clone(), [])]
    for step, action in enumerate(route, 1):
        if step in AMBIGUOUS:
            children = []
            for node, witness in branches:
                if node.terminal():
                    continue
                candidates = support_candidates(node.frame(), action)
                if step == AMBIGUOUS[0] and os.environ.get("FIRST_X"):
                    wanted_x = int(os.environ["FIRST_X"])
                    candidates = [
                        candidate
                        for candidate in candidates
                        if candidate[1] == wanted_x
                    ]
                for candidate in candidates:
                    child = node.clone()
                    child.step(*candidate)
                    if child.levels_completed > base_level:
                        print(
                            "SUPPORT_DECODE_WIN", step,
                            [*witness, candidate], flush=True,
                        )
                        return
                    if not child.terminal():
                        children.append((child, [*witness, candidate]))
            unique = {}
            for child, witness in children:
                key = np.asarray(child.frame())[:63].tobytes()
                unique.setdefault(key, (child, witness))
            branches = list(unique.values())[:128]
            print(
                "SUPPORT_DECODE_BOUNDARY", step, len(children),
                len(branches), flush=True,
            )
        else:
            live = []
            for node, witness in branches:
                node.step(*action)
                if node.levels_completed > base_level:
                    print(
                        "SUPPORT_DECODE_WIN", step,
                        [*witness, action], flush=True,
                    )
                    return
                if not node.terminal():
                    live.append((node, [*witness, action]))
            branches = live
        if not branches:
            print("SUPPORT_DECODE_ALL_DEAD", step, flush=True)
            return

    roots = {}
    for node, witness in branches:
        key = np.asarray(node.frame())[:63].tobytes()
        roots.setdefault(
            key,
            (
                node, witness, avatar_cell(node.frame()),
                target(node.frame()), tuple(controls(node.frame())),
                lattice(node.frame()),
            ),
        )
    print("SUPPORT_DECODE_ROOTS", len(roots), flush=True)
    for index, item in enumerate(roots.values()):
        node, witness, avatar, prize, switches, grid = item
        ambiguous_actions = tuple(
            (step, witness[step - 1]) for step in AMBIGUOUS
        )
        print(
            "SUPPORT_DECODE_ROOT", index, ambiguous_actions,
            avatar, prize, switches, grid, flush=True,
        )
        if os.environ.get("NO_SUFFIX") == "1":
            continue
        for direction in (LEFT, RIGHT):
            staged = node.clone()
            suffix = [direction] * 4
            for move in suffix:
                staged.step(*move)
                if staged.terminal() or staged.levels_completed > base_level:
                    break
            if staged.levels_completed > base_level:
                print(
                    "SUPPORT_DECODE_WIN",
                    [*witness, *suffix], flush=True,
                )
                return
            if staged.terminal():
                continue
            for control in controls(staged.frame()):
                child = staged.clone()
                child.step(*control)
                if child.levels_completed > base_level:
                    print(
                        "SUPPORT_DECODE_WIN",
                        [*witness, *suffix, control], flush=True,
                    )
                    return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
