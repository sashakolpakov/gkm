"""Test every ring transition under all bridge/main/selector phases."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import MAIN, SELECTOR, TOP, enter_right


MOVES = (
    (1, (6, 50, 34), 2),
    (2, (6, 50, 40), 1),
    (3, (6, 46, 36), 4),
    (4, (6, 54, 36), 3),
)


def ring_key(env):
    return perception.arr(env.frame())[6:42, 6:34].tobytes()


def exit_tiles(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(11,), min_area=4
        )
        if blob.area == 4
        and blob.size == (2, 2)
        and blob.bbox[1] < 40
    )


def ring_bfs(root, base_level):
    queue = deque([(root.clone(), [])])
    seen = {ring_key(root)}
    while queue:
        node, path = queue.popleft()
        for label, (outward, control, inward) in enumerate(MOVES):
            child = node.clone()
            child.step(outward)
            child.step(*control)
            child.step(inward)
            child_path = path + [label]
            if (
                child.levels_completed > base_level
                or exit_tiles(child)
            ):
                return child_path, exit_tiles(child), child.levels_completed
            key = ring_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, child_path))
    return None, (), base_level


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    entry = enter_right(env, 3)
    for top_phase in range(6):
        for main_phase in range(2):
            for selector_offset in range(4):
                root = entry.clone()
                for _ in range(top_phase):
                    root.step(*TOP)
                for _ in range(selector_offset):
                    root.step(*SELECTOR)
                if main_phase:
                    root.step(1)
                    root.step(*MAIN)
                    root.step(2)
                path, tiles, level = ring_bfs(root, base_level)
                print(
                    "SYNC_RING_CONTEXT",
                    top_phase, main_phase, selector_offset, path, tiles,
                    level - base_level,
                )
                if path is not None:
                    return


arena.run_program("dc22", observe)
