"""Finite decode matrix for the preserved 60-action witness."""

import itertools
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import ROW_ANCHORS
from perception import connected_components
from probe_level7_coordinate_decode import AMBIGUOUS, EARLY_SHIFTED
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)


def controls(frame):
    out = []
    for blob in connected_components(frame, colors=(8,), min_area=1):
        if blob.bbox[0] >= 63 or blob.bbox[1] > 5:
            continue
        y, x = blob.centroid
        action = (6, round(x), round(y))
        if action not in out:
            out.append(action)
    return sorted(out, key=lambda action: action[2])


def target(frame):
    ys, xs = np.where(np.asarray(frame)[:63] == 7)
    if not len(xs):
        return None
    return round(float(xs.mean())), round(float(ys.mean()))


def build(raw, flags):
    shifted = EARLY_SHIFTED | {
        step
        for step, flag in zip(AMBIGUOUS, flags)
        if flag
    }
    route = []
    for step, item in enumerate(raw, 1):
        action = (item,) if isinstance(item, int) else tuple(item)
        if step in shifted and len(action) == 3 and action[1] != 3:
            action = action[0], action[1] + 12, action[2]
        route.append(action)
    return route


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw = json.load(stream)["staged_prefix_actions"]
    root = env.clone()
    base_level = int(env.levels_completed)
    outcomes = []
    unique_states = {}
    tested = 0

    for flags in itertools.product((False, True), repeat=len(AMBIGUOUS)):
        for use_bottom in (False, True):
            node = root.clone()
            route = []
            repairs = []
            for step, action in enumerate(build(raw, flags), 1):
                candidate = action
                if (
                    len(action) == 3
                    and action[0] == 6
                    and action[1] <= 5
                    and int(node.frame()[action[2]][action[1]]) != 8
                ):
                    visible = controls(node.frame())
                    if visible:
                        candidate = visible[-1] if use_bottom else visible[0]
                        repairs.append((step, action, candidate))
                node.step(*candidate)
                route.append(candidate)
                if node.levels_completed > base_level:
                    print(
                        "DECODE_MATRIX_WIN", flags, use_bottom, repairs,
                        route, flush=True,
                    )
                    return
                if node.terminal():
                    break
            tested += 1
            if node.terminal():
                continue
            key = np.asarray(node.frame())[:63].tobytes()
            unique_states.setdefault(
                key, (node, flags, use_bottom, repairs, route)
            )
            if tested % 16 == 0:
                print(
                    "DECODE_MATRIX_PROGRESS", tested, len(unique_states),
                    flush=True,
                )

    for node, flags, use_bottom, repairs, route in unique_states.values():
        for direction in (LEFT, RIGHT):
            staged = node.clone()
            suffix = []
            for count in range(7):
                if count:
                    staged.step(*direction)
                    suffix.append(direction)
                    if staged.levels_completed > base_level:
                        print(
                            "DECODE_MATRIX_WIN", flags, use_bottom,
                            repairs, [*route, *suffix], flush=True,
                        )
                        return
                    if staged.terminal():
                        break
                for control in controls(staged.frame()):
                    child = staged.clone()
                    child.step(*control)
                    if child.levels_completed > base_level:
                        print(
                            "DECODE_MATRIX_WIN", flags, use_bottom,
                            repairs,
                            [*route, *suffix, control], flush=True,
                        )
                        return
                outcomes.append(
                    (
                        flags, use_bottom, repairs, direction, count,
                        avatar_cell(staged.frame()), target(staged.frame()),
                        tuple(controls(staged.frame())), lattice(staged.frame()),
                    )
                )
    outcomes.sort(
        key=lambda item: (
            item[6] is None,
            -(item[5] or (0, 0))[1],
            item[5] or (99, 99),
        )
    )
    print(
        "DECODE_MATRIX_DONE", tested, len(unique_states), len(outcomes),
        flush=True,
    )
    for outcome in outcomes[:30]:
        print("DECODE_MATRIX_STATE", outcome, flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
