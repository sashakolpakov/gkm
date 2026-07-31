import importlib.util
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import connected_components


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

A_CONTROL = (6, 56, 8)
B_CONTROL = (6, 51, 25)
S_CONTROL = (6, 51, 48)
D_CONTROLS = {
    "u": (1, (6, 50, 32), 2),
    "d": (2, (6, 50, 40), 1),
    "l": (3, (6, 46, 36), 4),
    "r": (4, (6, 54, 36), 3),
}

REMOTE = [
    3, 3, 3, 3, 3,
    A_CONTROL, A_CONTROL, A_CONTROL, A_CONTROL,
    2, 2, 3, 3, 3, 2, 3, A_CONTROL, 1, A_CONTROL, 1, 1,
    B_CONTROL,
] + [1] * 17 + [3] + [2] * 11

HUB = (
    REMOTE
    + [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
    + [S_CONTROL, S_CONTROL, S_CONTROL, 3, B_CONTROL]
)

TOP_PATH = ["u", "r", "u", "u", "l", "l", "u", "u", "u"]

REVERSE_TO_ROOT = [
    4,
    2, 2, 2, 3, 2, 3, 3, 2, 3, 3, 1, A_CONTROL, 1,
] + [1] * 7 + [3]

ROOT_TO_SELECTOR = (
    [2] * 8
    + [4, 4, A_CONTROL, 4, A_CONTROL, 1]
    + [A_CONTROL, 4] * 3
    + [1, 1, 1]
)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, path):
    for action in path:
        step(env, action)


def move_ring(env, labels):
    for label in labels:
        outward, control, inward = D_CONTROLS[label]
        apply(env, [outward, control, inward])


def avatar(env):
    blobs = connected_components(env.frame(), colors=(14,), min_area=2)
    candidates = [blob for blob in blobs if blob.area >= 2]
    if not candidates:
        return None
    blob = max(candidates, key=lambda item: item.area)
    return blob.bbox[0] // 2, blob.bbox[1] // 2, blob.area


def color8(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(8,), min_area=4)
        if blob.bbox[1] < 40
    ]


def movers(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(1, 8, 12), min_area=4
        )
        if blob.bbox[1] < 40
    )


def movement_reach(root):
    queue = deque([(root.clone(), [])])
    seen = {root.frame()[:63].tobytes()}
    positions = {avatar(root): []}
    while queue and len(seen) < 180:
        node, path = queue.popleft()
        for direction in (1, 2, 3, 4):
            child = node.clone()
            child.step(direction)
            child_path = path + [direction]
            if child.levels_completed > 5:
                return positions, child_path
            key = child.frame()[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            position = avatar(child)
            positions.setdefault(position, child_path)
            queue.append((child, child_path))
    return positions, None


def phase1(root, label):
    child = root.clone()
    apply(child, [B_CONTROL, S_CONTROL, S_CONTROL, B_CONTROL])
    print(
        label, "avatar", avatar(child), "level", child.levels_completed,
        "color8", color8(child), flush=True,
    )
    positions, win = movement_reach(child)
    print(
        label, "reach", sorted(position for position in positions if position),
        "win", win, flush=True,
    )


def phase1_result(root):
    child = root.clone()
    apply(child, [B_CONTROL, S_CONTROL, S_CONTROL, B_CONTROL])
    return avatar(child), child.levels_completed


def center_controls(root, label):
    before = root.frame()[:63].copy()
    for point in ((50, 36), (51, 35), (51, 36)):
        child = root.clone()
        child.step(6, *point)
        changed = int(np.count_nonzero(before != child.frame()[:63]))
        if changed or child.levels_completed > 5:
            print(
                "CENTER", label, point, "changed", changed,
                "avatar", avatar(child), "level", child.levels_completed,
                "color8", color8(child), flush=True,
            )


def run(env):
    solver.solve(env)
    apply(env, HUB)
    print("HUB", avatar(env), color8(env), flush=True)
    center_controls(env, "HUB")
    hub_root = env.clone()

    left_terminal = hub_root.clone()
    move_ring(left_terminal, ["l", "l", "l", "l"])
    for push in range(1, 9):
        before = left_terminal.frame()[:63].copy()
        move_ring(left_terminal, ["l"])
        changed = int(np.count_nonzero(before != left_terminal.frame()[:63]))
        print(
            "LEFT_PUSH", push, "changed", changed,
            "level", left_terminal.levels_completed,
            "phase1", phase1_result(left_terminal), flush=True,
        )

    move_ring(env, ["l", "l", "l"])
    print("DOCKED", avatar(env), color8(env), env.levels_completed, flush=True)
    center_controls(env, "DOCKED")
    docked = env.clone()
    phase1(env, "DOCKED_PHASE1")

    move_ring(env, ["r", "r", "r"])
    print("RETURNED", avatar(env), color8(env), env.levels_completed, flush=True)
    phase1(env, "RETURNED_PHASE1")

    move_ring(env, TOP_PATH)
    print("TOP", avatar(env), color8(env), env.levels_completed, flush=True)
    center_controls(env, "TOP")
    north_switch = env.clone()
    apply(north_switch, [B_CONTROL] + REVERSE_TO_ROOT + [1] * 12)
    print(
        "NORTH_SWITCH_APPROACH", avatar(north_switch),
        north_switch.levels_completed, flush=True,
    )
    before_switch = north_switch.frame()[:63].copy()
    north_switch.step(2)
    print(
        "NORTH_SWITCH_ENTER", avatar(north_switch),
        "changed", int(np.count_nonzero(
            before_switch != north_switch.frame()[:63]
        )),
        "level", north_switch.levels_completed, flush=True,
    )
    switch_positions, switch_win = movement_reach(north_switch)
    print(
        "NORTH_SWITCH_REACH",
        sorted(position for position in switch_positions if position),
        "win", switch_win, flush=True,
    )
    phase1(north_switch, "NORTH_SWITCH_PHASE1")
    for push in range(1, 9):
        before = env.frame()[:63].copy()
        move_ring(env, ["u"])
        changed = int(np.count_nonzero(before != env.frame()[:63]))
        print(
            "TOP_PUSH", push, "changed", changed,
            "level", env.levels_completed,
            "phase1", phase1_result(env), flush=True,
        )
    phase1(env, "TOP_PHASE1")

    northern = env.clone()
    apply(
        northern,
        [B_CONTROL, S_CONTROL, S_CONTROL, S_CONTROL, B_CONTROL],
    )
    positions, _ = movement_reach(northern)
    for position, path in positions.items():
        for name in ("u", "d", "l", "r"):
            child = northern.clone()
            apply(child, path)
            before = color8(child)
            outward, control, inward = D_CONTROLS[name]
            step(child, control)
            after = color8(child)
            if after != before or child.levels_completed > 5:
                print(
                    "NORTH_CONTROL", position, name, before, after,
                    child.levels_completed, flush=True,
                )
    for direction in (1, 2, 3, 4):
        child = northern.clone()
        previous = (avatar(child), movers(child))
        transitions = []
        for turn in range(1, 31):
            child.step(direction)
            current = (avatar(child), movers(child))
            if current != previous or child.levels_completed > 5:
                transitions.append(
                    (turn, current[0], current[1], child.levels_completed)
                )
                previous = current
            if child.levels_completed > 5:
                break
        print("NORTH_REPEAT", direction, transitions, flush=True)

    coupled = docked.clone()
    apply(coupled, [B_CONTROL] + REVERSE_TO_ROOT + [1] * 5)
    print(
        "COUPLED_BEFORE", avatar(coupled), color8(coupled),
        movers(coupled), coupled.levels_completed, flush=True,
    )
    step(coupled, B_CONTROL)
    print(
        "COUPLED_AFTER", avatar(coupled), color8(coupled),
        movers(coupled), coupled.levels_completed, flush=True,
    )
    apply(
        coupled,
        [2] * 5 + ROOT_TO_SELECTOR + [3, B_CONTROL],
    )
    print(
        "COUPLED_HUB", avatar(coupled), color8(coupled),
        movers(coupled), coupled.levels_completed, flush=True,
    )
    phase1(coupled, "COUPLED_PHASE1")
    move_ring(coupled, ["r", "r", "r"] + TOP_PATH)
    print(
        "COUPLED_TOP", avatar(coupled), color8(coupled),
        movers(coupled), coupled.levels_completed, flush=True,
    )
    phase1(coupled, "COUPLED_TOP_PHASE1")

    attached = hub_root.clone()
    move_ring(attached, ["l", "l", "l", "l"])
    apply(attached, [B_CONTROL] + REVERSE_TO_ROOT + [2] * 8)
    print(
        "ATTACH_BEFORE", avatar(attached), color8(attached),
        movers(attached), attached.levels_completed, flush=True,
    )
    step(attached, B_CONTROL)
    print(
        "ATTACH_AFTER", avatar(attached), color8(attached),
        movers(attached), attached.levels_completed, flush=True,
    )
    selector_suffix = (
        [4, 4, A_CONTROL, 4, A_CONTROL, 1]
        + [A_CONTROL, 4] * 3
        + [1, 1, 1, 3, B_CONTROL]
    )
    apply(attached, selector_suffix)
    print(
        "ATTACH_HUB", avatar(attached), color8(attached),
        movers(attached), attached.levels_completed, flush=True,
    )
    move_ring(attached, ["r", "r", "r", "r"])
    print(
        "ATTACH_RETURN", avatar(attached), color8(attached),
        movers(attached), attached.levels_completed, flush=True,
    )
    move_ring(attached, TOP_PATH)
    print(
        "ATTACH_TOP", avatar(attached), color8(attached),
        movers(attached), attached.levels_completed, flush=True,
    )
    phase1(attached, "ATTACH_TOP_PHASE1")

    carried = hub_root.clone()
    move_ring(carried, ["l", "l", "l", "l"])
    apply(carried, [B_CONTROL] + REVERSE_TO_ROOT + [2] * 8)
    apply(carried, [B_CONTROL, B_CONTROL] + selector_suffix)
    print(
        "CARRY_HUB", avatar(carried), color8(carried),
        movers(carried), carried.levels_completed, flush=True,
    )
    center_controls(carried, "CARRY_HUB")
    phase1(carried, "CARRY_HUB_PHASE1")
    move_ring(carried, ["r", "r", "r", "r"])
    print(
        "CARRY_RETURN", avatar(carried), color8(carried),
        movers(carried), carried.levels_completed, flush=True,
    )
    move_ring(carried, TOP_PATH)
    print(
        "CARRY_TOP", avatar(carried), color8(carried),
        movers(carried), carried.levels_completed, flush=True,
    )
    phase1(carried, "CARRY_TOP_PHASE1")

    routed = hub_root.clone()
    apply(routed, [S_CONTROL, S_CONTROL])
    move_ring(routed, TOP_PATH)
    print(
        "ROUTED_TOP", avatar(routed), color8(routed),
        movers(routed), routed.levels_completed, flush=True,
    )
    apply(
        routed,
        [S_CONTROL, S_CONTROL, B_CONTROL, S_CONTROL, S_CONTROL, B_CONTROL],
    )
    print(
        "ROUTED_PHASE1", avatar(routed), color8(routed),
        movers(routed), routed.levels_completed, flush=True,
    )
    routed_positions, routed_win = movement_reach(routed)
    print(
        "ROUTED_REACH",
        sorted(position for position in routed_positions if position),
        "win", routed_win, flush=True,
    )


levels, path, error = A.run_program("dc22", run)
print("HARNESS", levels, len(path), error, flush=True)
