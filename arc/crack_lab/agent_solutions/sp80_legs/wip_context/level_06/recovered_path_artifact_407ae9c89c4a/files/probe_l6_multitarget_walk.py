"""Achieved-state walk for the bar-serves-both-left-sockets hypothesis."""
from collections import defaultdict, deque
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_central_stack import moves, placements


REVERSE = {1: 2, 2: 1, 3: 4, 4: 3}


def selected_pos(env):
    pixels = np.asarray(env.frame())
    rows, cols = np.where(pixels == 9)
    return int(rows.min()), int(cols.min())


def apply(env, actions):
    for action in actions:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def d_tree(env, clock, started):
    root = env.clone()
    root.step(6, 31, 46)
    apply(root, [1] * 11 + [4] * 8)
    root_key = selected_pos(root)
    queue = deque([root])
    seen = {root_key}
    children = defaultdict(list)
    paths = {root_key: []}
    while queue:
        node = queue.popleft()
        parent = selected_pos(node)
        for action in (1, 2, 3, 4):
            child = node.clone()
            try:
                child.step(action)
            except IndexError:
                continue
            clock[0] += 1
            key = selected_pos(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append(child)
            children[parent].append((action, key))
            paths[key] = paths[parent] + [action]
            delay = clock[0] / 280.0 - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)

    walk = []

    def visit(parent):
        for action, child in children[parent]:
            walk.append((action, child, True))
            visit(child)
            walk.append((REVERSE[action], parent, False))

    visit(root_key)
    return root_key, walk, paths


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    started = time.monotonic()
    clock = [0]
    root_key, walk, d_paths = d_tree(env, clock, started)
    print("WALK_PHASE", "TREE", root_key, len(d_paths))

    staged = env.clone()
    prefix = (
        [(6, 31, 46)] + [1] * 11 + [4] * 8
        + [(6, 30, 19)] + [2] * 4 + [4] * 5
    )
    apply(staged, prefix)
    clock[0] += len(prefix)
    print("WALK_PHASE", "STAGED")

    tested = 0
    errors = 0
    c_tops = tuple(range(11, 48, 3))
    b_lefts = tuple(range(14, 48, 3))
    for c_node, c in placements(
        staged, (25, 33), 32, 23, c_tops, (23,), clock,
    ):
        print("WALK_PHASE", "C", c)
        for b_node, b in placements(
            c_node, (45, 18), 14, 44, (26,), b_lefts, clock,
        ):
            print("WALK_PHASE", "B", b)
            work = b_node.clone()
            work.step(6, 57, 13)
            clock[0] += 1

            def check(position):
                nonlocal tested, errors
                test = work.clone()
                try:
                    test.step(5)
                except IndexError:
                    errors += 1
                    return False
                clock[0] += 1
                tested += 1
                if test.levels_completed <= env.levels_completed:
                    return False
                path = (
                    prefix
                    + [(6, 25, 33)]
                    + moves(32, c[0], 1, 2)
                    + [(6, 45, 18)]
                    + moves(14, 26, 1, 2)
                    + moves(44, b[1], 3, 4)
                    + [(6, 57, 13)]
                    + d_paths[position]
                    + [5]
                )
                print(
                    "MULTITARGET_WALK_WIN",
                    {"A": (29, 44), "B": b, "C": c, "D": position},
                    "TESTED", tested, "STEPS", clock[0], "PATH", path,
                )
                return True

            if check(root_key):
                return
            for action, position, first_visit in walk:
                try:
                    work.step(action)
                except IndexError:
                    errors += 1
                    break
                clock[0] += 1
                if first_visit and check(position):
                    return
                if first_visit:
                    delay = clock[0] / 280.0 - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(delay)
    print(
        "MULTITARGET_WALK_NONE", "TESTED", tested,
        "ERRORS", errors, "STEPS", clock[0],
    )


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
