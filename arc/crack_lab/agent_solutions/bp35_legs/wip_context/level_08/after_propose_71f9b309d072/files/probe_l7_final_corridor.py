"""Reproduce and clear the support-packed final corridor decode."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import click_action
from probe_l7_decode_matrix import build, controls, target
from probe_level7_coordinate_decode import AMBIGUOUS
from probe_level7_reward_recovery import avatar_cell, lattice


LEFT, RIGHT = (3,), (4,)
FLAGS = (False, False, True, False, True)


def summary(node):
    return (
        int(node.levels_completed),
        bool(node.terminal()),
        None if node.terminal() else avatar_cell(node.frame()),
        None if node.terminal() else target(node.frame()),
        () if node.terminal() else tuple(controls(node.frame())),
        "" if node.terminal() else lattice(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw = json.load(stream)["staged_prefix_actions"]
    base_level = int(env.levels_completed)
    route = []
    repairs = []
    for step, action in enumerate(build(raw, FLAGS), 1):
        candidate = action
        if (
            len(action) == 3
            and action[0] == 6
            and action[1] <= 5
            and int(env.frame()[action[2]][action[1]]) != 8
        ):
            visible = controls(env.frame())
            if visible:
                candidate = visible[0]
                repairs.append((step, action, candidate))
        env.step(*candidate)
        route.append(candidate)
        if env.terminal() or env.levels_completed > base_level:
            print("FINAL_CORRIDOR_EARLY", step, candidate, summary(env))
            return
    print(
        "FINAL_CORRIDOR_ROOT", FLAGS, tuple(AMBIGUOUS), repairs,
        summary(env), flush=True,
    )

    clean_root = env.clone()
    clean_variants = {
        "roundtrip": [RIGHT] * 3 + [LEFT] * 4,
        "right_release_left": [RIGHT] * 3 + [(7,)] + [LEFT] * 4,
        "right_two_release_left": (
            [RIGHT] * 3 + [(7,), (7,)] + [LEFT] * 4
        ),
        "right_left_release_left": (
            [RIGHT] * 3 + [LEFT, (7,)] + [LEFT] * 4
        ),
        "right_wait_left": (
            [RIGHT] * 3 + [(6, 9, 3)] + [LEFT] * 4
        ),
    }
    for name, suffix in clean_variants.items():
        node = clean_root.clone()
        for action in suffix:
            node.step(*action)
            if node.terminal() or node.levels_completed > base_level:
                break
        print(
            "FINAL_CORRIDOR_CLEAN", name, suffix, summary(node), flush=True,
        )
        if node.levels_completed > base_level:
            print("FINAL_CORRIDOR_WIN", [*route, *suffix], flush=True)
            return

    release_states = {}
    for rights in (0, 3):
        for releases in range(31):
            node = clean_root.clone()
            suffix = [RIGHT] * rights + [(7,)] * releases + [LEFT] * 4
            for action in suffix:
                node.step(*action)
                if node.terminal() or node.levels_completed > base_level:
                    break
            if node.levels_completed > base_level:
                print(
                    "FINAL_CORRIDOR_WIN", [*route, *suffix], flush=True,
                )
                return
            if node.terminal():
                continue
            for control in controls(node.frame()):
                child = node.clone()
                child.step(*control)
                if child.levels_completed > base_level:
                    print(
                        "FINAL_CORRIDOR_WIN",
                        [*route, *suffix, control], flush=True,
                    )
                    return
            state = summary(node)
            release_states.setdefault(state, (rights, releases))
    print(
        "FINAL_CORRIDOR_RELEASES", len(release_states), flush=True,
    )
    for state, witness in release_states.items():
        print("FINAL_CORRIDOR_RELEASE_STATE", witness, state, flush=True)

    for index, action in enumerate(
        [
            RIGHT, RIGHT, RIGHT,
            click_action(6, 4), LEFT,
            click_action(6, 3), LEFT,
            click_action(6, 2), LEFT,
            LEFT, LEFT,
        ],
        1,
    ):
        env.step(*action)
        route.append(action)
        print("FINAL_CORRIDOR_STEP", index, action, summary(env), flush=True)
        if env.terminal() or env.levels_completed > base_level:
            if env.levels_completed > base_level:
                print("FINAL_CORRIDOR_WIN", route, flush=True)
            return

    frame = np.asarray(env.frame())
    local = [LEFT, RIGHT, (7,), *controls(frame)]
    for row in range(10):
        for col in range(8):
            color = int(frame[3 + 6 * row][15 + 6 * col])
            if color in (12, 14, 15):
                local.append(click_action(row, col))
    for action in dict.fromkeys(local):
        child = env.clone()
        child.step(*action)
        if child.levels_completed > base_level:
            print("FINAL_CORRIDOR_WIN", [*route, action], flush=True)
            return
        if not child.terminal() and (
            avatar_cell(child.frame()) != avatar_cell(env.frame())
            or target(child.frame()) is not None
            or controls(child.frame())
        ):
            print(
                "FINAL_CORRIDOR_OPTION", action, summary(child), flush=True,
            )


arena.run_program("bp35", probe)
