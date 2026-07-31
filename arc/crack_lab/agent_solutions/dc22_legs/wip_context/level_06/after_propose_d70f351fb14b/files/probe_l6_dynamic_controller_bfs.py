"""Interleave ring motion and global phases while staying on the controller."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import MAIN, SELECTOR, TOP, avatar_position, enter_right


MACROS = (
    ("U", (1, (6, 50, 34), 2)),
    ("D", (2, (6, 50, 40), 1)),
    ("L", (3, (6, 46, 36), 4)),
    ("R", (4, (6, 54, 36), 3)),
    ("B", (1, MAIN, 2)),
    ("A", (TOP,)),
    ("S", (SELECTOR,)),
)


def apply(env, actions):
    for action in actions:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def key(env):
    return perception.arr(env.frame())[:63].tobytes()


def rings(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 40
    )


def exits(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(11,), min_area=4
        )
        if blob.area == 4 and blob.size == (2, 2) and blob.bbox[1] < 40
    )


def phase1_destination(node, phase):
    probe = node.clone()
    probe.step(*MAIN)
    hub = avatar_position(probe)
    offset = (1 - phase) % 4
    apply(probe, [SELECTOR] * offset)
    before = avatar_position(probe)
    probe.step(*MAIN)
    return hub, before, avatar_position(probe), probe.levels_completed


def observe(env):
    solve.solve(env)
    root = enter_right(env, 3)
    base_level = root.levels_completed
    queue = deque([(root.clone(), [], 3)])
    seen = {key(root)}
    placements = {rings(root)}
    while queue and len(seen) < 5000:
        node, path, phase = queue.popleft()
        if len(path) >= 40:
            continue
        for label, actions in MACROS:
            child = node.clone()
            apply(child, actions)
            child_path = path + [label]
            child_phase = (phase + 1) % 4 if label == "S" else phase
            if child.levels_completed > base_level:
                print(
                    "DYNAMIC_CONTROLLER_WIN", child_path,
                    "states", len(seen), flush=True,
                )
                return
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            ring = rings(child)
            if ring not in placements:
                placements.add(ring)
                print(
                    "DYNAMIC_CONTROLLER_PLACEMENT", len(placements),
                    ring, child_path, flush=True,
                )
            visible = exits(child)
            if visible:
                print(
                    "DYNAMIC_CONTROLLER_EXIT", child_path,
                    visible, flush=True,
                )
                return
            hub, before, after, level = phase1_destination(
                child, child_phase
            )
            if after != before or level > base_level:
                print(
                    "DYNAMIC_CONTROLLER_PHASE1", child_path,
                    "phase", child_phase,
                    "hub", hub, "before", before, "after", after,
                    "level", level, "states", len(seen), flush=True,
                )
                return
            if len(seen) % 100 == 0:
                print(
                    "DYNAMIC_CONTROLLER_PROGRESS", len(seen),
                    "queue", len(queue), "depth", len(child_path),
                    "placements", len(placements), flush=True,
                )
            queue.append((child, child_path, child_phase))
    print(
        "DYNAMIC_CONTROLLER_DONE", len(seen), len(queue),
        "placements", len(placements), flush=True,
    )


arena.run_program("dc22", observe)
