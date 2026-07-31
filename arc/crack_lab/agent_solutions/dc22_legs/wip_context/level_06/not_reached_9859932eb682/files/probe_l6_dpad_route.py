"""Map remote direction selection to legal cargo-ring moves."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import (
    avatar_position,
    enter_right,
    movement_reach,
)


UP = (6, 50, 34)
LEFT = (6, 46, 36)
RIGHT = (6, 54, 36)
DOWN = (6, 50, 40)


def ring(env):
    return [
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    ]


def observe(env):
    solve.solve(env)
    base = enter_right(env, 3)
    reached, _ = movement_reach(base)
    for action in reached[(56, 34)]:
        base.step(action)
    base.step(*UP)
    print("RING_AT_JUNCTION", ring(base), avatar_position(base))
    branches = (
        ("LEFT", (2, 3), LEFT),
        ("RIGHT", (2, 4), RIGHT),
        ("DOWN", (2, 2), DOWN),
    )
    for name, selector_path, control in branches:
        branch = base.clone()
        for action in selector_path:
            branch.step(action)
        before = ring(branch)
        branch.step(*control)
        print(
            "DIRECTION", name, avatar_position(branch),
            before, ring(branch), branch.levels_completed,
        )
    left_chain = base.clone()
    left_chain.step(2)
    left_chain.step(3)
    for phase in range(1, 9):
        before = ring(left_chain)
        left_chain.step(*LEFT)
        after = ring(left_chain)
        print(
            "LEFT_CHAIN", phase, before, after,
            left_chain.levels_completed,
        )
    docked = base.clone()
    docked.step(2)
    docked.step(3)
    docked.step(*LEFT)
    docked.step(6, 50, 26)
    print("DOCKED_MAIN", ring(docked), avatar_position(docked))
    docked_branches = (
        ("LEFT", (), LEFT),
        ("UP", (4, 1), UP),
        ("RIGHT", (4, 4), RIGHT),
        ("DOWN", (4, 2), DOWN),
    )
    for name, selector_path, control in docked_branches:
        branch = docked.clone()
        for action in selector_path:
            branch.step(action)
        before = ring(branch)
        branch.step(*control)
        print(
            "DOCKED_DIRECTION", name, avatar_position(branch),
            before, ring(branch), branch.levels_completed,
        )
    turned = docked.clone()
    turned.step(4)
    turned.step(2)
    turned.step(*DOWN)
    for restore_main in (False, True):
        branch = turned.clone()
        branch.step(1)
        branch.step(3)
        if restore_main:
            branch.step(6, 50, 26)
        before = ring(branch)
        branch.step(*LEFT)
        print(
            "TURNED_LEFT", restore_main, avatar_position(branch),
            before, ring(branch), branch.levels_completed,
        )
    dock_chain = turned.clone()
    dock_chain.step(1)
    dock_chain.step(3)
    for phase in range(1, 7):
        before = ring(dock_chain)
        dock_chain.step(*LEFT)
        print(
            "INNER_DOCK_CHAIN", phase, before, ring(dock_chain),
            dock_chain.levels_completed,
        )
    inner_docked = turned.clone()
    inner_docked.step(1)
    inner_docked.step(3)
    for _ in range(3):
        inner_docked.step(*LEFT)
    inner_branches = (
        ("LEFT", (), LEFT),
        ("UP", (4, 1), UP),
        ("RIGHT", (4, 4), RIGHT),
        ("DOWN", (4, 2), DOWN),
    )
    for name, selector_path, control in inner_branches:
        branch = inner_docked.clone()
        for action in selector_path:
            branch.step(action)
        before = ring(branch)
        branch.step(*control)
        print(
            "INNER_DOCK_PRE_MAIN_DIRECTION", name,
            avatar_position(branch), before, ring(branch),
            branch.levels_completed,
        )
    inner_ready = inner_docked.clone()
    before_main = ring(inner_docked)
    inner_docked.step(6, 50, 26)
    print(
        "INNER_DOCK_MAIN", before_main, ring(inner_docked),
        avatar_position(inner_docked), inner_docked.levels_completed,
    )
    for name, selector_path, control in inner_branches:
        branch = inner_docked.clone()
        for action in selector_path:
            branch.step(action)
        before = ring(branch)
        branch.step(*control)
        print(
            "INNER_DOCK_DIRECTION", name, avatar_position(branch),
            before, ring(branch), branch.levels_completed,
        )
    state_one = inner_ready.clone()
    state_one.step(4)
    state_one.step(6, 50, 26)
    hub_position = avatar_position(state_one)
    state_one.step(6, 50, 46)
    state_one.step(6, 50, 46)
    state_one.step(6, 50, 26)
    print(
        "INNER_DOCK_STATE1", hub_position, avatar_position(state_one),
        state_one.levels_completed,
    )
    partial = enter_right(env, 3)
    partial.step(3)
    partial.step(*LEFT)
    for _ in range(2):
        partial.step(4)
        partial.step(3)
        partial.step(*LEFT)
    partial_branches = (
        ("LEFT", (), LEFT),
        ("UP", (4, 1), UP),
        ("RIGHT", (4, 4), RIGHT),
        ("DOWN", (4, 2), DOWN),
    )
    for rotated in (False, True):
        root = partial.clone()
        if rotated:
            root.step(6, 50, 26)
        for name, selector_path, control in partial_branches:
            branch = root.clone()
            for action in selector_path:
                branch.step(action)
            before = ring(branch)
            branch.step(*control)
            print(
                "PARTIAL_DOCK_DIRECTION", rotated, name,
                before, ring(branch), branch.levels_completed,
            )
    partial_state_one = partial.clone()
    partial_state_one.step(4)
    partial_state_one.step(6, 50, 26)
    hub_position = avatar_position(partial_state_one)
    partial_state_one.step(6, 50, 46)
    partial_state_one.step(6, 50, 46)
    partial_state_one.step(6, 50, 26)
    print(
        "PARTIAL_DOCK_STATE1", hub_position,
        avatar_position(partial_state_one),
        partial_state_one.levels_completed,
    )


arena.run_program("dc22", observe)
