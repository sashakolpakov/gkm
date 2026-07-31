"""Look for activation of the selector's missing (13, 8) destination."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_exact_crossings import (
    horizontal_entry,
    placements_with_paths,
)
from probe_l6_right import MAIN, enter_right
from probe_l6_right import SELECTOR


def exact_closure(root):
    queue = deque([(root.clone(), [])])
    seen = {perception.arr(root.frame())[:63].tobytes()}
    out = []
    while queue:
        node, path = queue.popleft()
        out.append((node, path))
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = perception.arr(child.frame())[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, path + [action]))
    return out


def marker(env, color):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(color,), min_area=1
        )
        if blob.bbox[1] < 40
    )


def avatar(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(14,), min_area=1
        )
        if blob.bbox[1] < 40
    )


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    placements = placements_with_paths(enter_right(env, 3))
    for index in range(len(placements)):
        root = horizontal_entry(placements[index][0])
        for node, path in exact_closure(root):
            before_avatar = avatar(node)
            phase_one = node.clone()
            phase_one.step(*SELECTOR)
            phase_one.step(*SELECTOR)
            phase_one.step(*MAIN)
            after_avatar = avatar(phase_one)
            hub_after = tuple(
                item for item in after_avatar
                if item[0][0] >= 44
                and 18 <= item[0][1] < 40
                and item[1] == 4
            )
            if hub_after or phase_one.levels_completed > base_level:
                print(
                    "PHASE1_TELEPORT", index, path, before_avatar,
                    "AFTER_AVATAR", after_avatar,
                    "LEVEL", phase_one.levels_completed,
                )
                return
        print("PHASE1_CONTEXT_DONE", index)


arena.run_program("dc22", observe)
