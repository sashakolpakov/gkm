"""Retarget only stale gravity coordinates in the decoded route."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)


def control_actions(frame):
    actions = []
    for blob in connected_components(frame, colors=(8,), min_area=1):
        if blob.bbox[0] >= 63 or blob.bbox[1] > 5:
            continue
        y, x = blob.centroid
        action = (6, round(x), round(y))
        if action not in actions:
            actions.append(action)
    return actions


def target_cell(frame):
    ys, xs = np.where(np.asarray(frame)[:63] == 7)
    if not len(xs):
        return None
    return round(float(xs.mean())), round(float(ys.mean()))


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    base_level = int(env.levels_completed)

    branches = [(env.clone(), [], [])]
    for index, action in enumerate(decoded_route(), 1):
        next_branches = []
        for node, route, replaced in branches:
            candidates = [action]
            if (
                len(action) == 3
                and action[0] == 6
                and action[1] <= 5
                and int(node.frame()[action[2]][action[1]]) != 8
            ):
                visible = control_actions(node.frame())
                candidates = [action, *visible]
            for candidate in dict.fromkeys(candidates):
                child = node.clone()
                child.step(*candidate)
                if child.levels_completed > base_level:
                    print(
                        "RETARGET_WIN_ROUTE", [*route, candidate],
                        [*replaced, (index, action, candidate)], flush=True,
                    )
                    return
                if child.terminal():
                    continue
                change = (
                    []
                    if candidate == action
                    else [(index, action, candidate)]
                )
                next_branches.append(
                    (child, [*route, candidate], [*replaced, *change])
                )
        # Exact frame plus replacement record prevents accidental collapse of
        # distinct held-input histories.
        unique = {}
        for item in next_branches:
            key = (
                np.asarray(item[0].frame())[:63].tobytes(),
                tuple(item[2]),
            )
            unique.setdefault(key, item)
        branches = list(unique.values())
        if len(branches) > 64:
            branches = branches[:64]
        if not branches:
            print("RETARGET_ALL_DEAD", index, flush=True)
            return

    outcomes = []
    for node, route, replaced in branches:
        for walk in (
            [LEFT] * 4,
            [RIGHT] * 4,
            [LEFT] * 3,
            [RIGHT] * 3,
            [],
        ):
            staged = node.clone()
            suffix = []
            for action in walk:
                staged.step(*action)
                suffix.append(action)
                if staged.levels_completed > base_level:
                    print(
                        "RETARGET_WIN_WALK", replaced,
                        [*route, *suffix], flush=True,
                    )
                    return
                if staged.terminal():
                    break
            if staged.terminal():
                continue
            for control in control_actions(staged.frame()):
                child = staged.clone()
                child.step(*control)
                if child.levels_completed > base_level:
                    print(
                        "RETARGET_WIN_CONTROL", replaced,
                        [*route, *suffix, control], flush=True,
                    )
                    return
            outcomes.append(
                (
                    replaced,
                    tuple(walk),
                    avatar_cell(staged.frame()),
                    target_cell(staged.frame()),
                    tuple(control_actions(staged.frame())),
                    lattice(staged.frame()),
                )
            )
    print("RETARGET_DONE", len(branches), len(outcomes), flush=True)
    for outcome in outcomes:
        print("RETARGET_STATE", outcome, flush=True)


arena.run_program("bp35", probe)
