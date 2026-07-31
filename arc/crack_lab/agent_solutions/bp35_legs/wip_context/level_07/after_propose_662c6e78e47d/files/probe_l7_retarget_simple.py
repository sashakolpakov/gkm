"""Eight deterministic top/bottom repairs for stale decoded gravity clicks."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_level7_decoded_stage import decoded_route
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


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    base_level = int(env.levels_completed)
    root = env.clone()
    outcomes = []
    for mask in range(8):
        node = root.clone()
        repaired = []
        route = []
        stale_index = 0
        for step, action in enumerate(decoded_route(), 1):
            candidate = action
            if (
                len(action) == 3
                and action[0] == 6
                and action[1] <= 5
                and int(node.frame()[action[2]][action[1]]) != 8
            ):
                visible = controls(node.frame())
                if visible:
                    choose_bottom = bool(mask & (1 << stale_index))
                    candidate = visible[-1] if choose_bottom else visible[0]
                    repaired.append((step, action, candidate))
                stale_index += 1
            node.step(*candidate)
            route.append(candidate)
            if node.levels_completed > base_level:
                print("RETARGET_SIMPLE_WIN", mask, repaired, route, flush=True)
                return
            if node.terminal():
                break
        if node.terminal():
            outcomes.append((mask, repaired, "dead"))
            continue
        for direction in (LEFT, RIGHT):
            staged = node.clone()
            suffix = []
            for _ in range(4):
                staged.step(*direction)
                suffix.append(direction)
                if staged.levels_completed > base_level:
                    print(
                        "RETARGET_SIMPLE_WIN", mask, repaired,
                        [*route, *suffix], flush=True,
                    )
                    return
                if staged.terminal():
                    break
            if staged.terminal():
                continue
            for control in controls(staged.frame()):
                child = staged.clone()
                child.step(*control)
                if child.levels_completed > base_level:
                    print(
                        "RETARGET_SIMPLE_WIN", mask, repaired,
                        [*route, *suffix, control], flush=True,
                    )
                    return
            outcomes.append(
                (
                    mask, repaired, direction,
                    avatar_cell(staged.frame()), target(staged.frame()),
                    tuple(controls(staged.frame())), lattice(staged.frame()),
                )
            )
    print("RETARGET_SIMPLE_DONE", len(outcomes), flush=True)
    for outcome in outcomes:
        print("RETARGET_SIMPLE_STATE", outcome, flush=True)


arena.run_program("bp35", probe)
