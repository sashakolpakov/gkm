"""Bounded symbolic clone search from a reproduced level-6 frontier."""
from collections import deque

import gkm_try as harness
import legs


PREFIX = [
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
    legs.click_action(5, 4),
    (legs.LEFT,), (legs.LEFT,), (legs.LEFT,),
    legs.click_action(4, 2),
    legs.click_action(7, 0),
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
    (legs.LEFT,), (legs.LEFT,),
    (legs.RIGHT,), (legs.RIGHT,),
    (legs.LEFT,), (legs.LEFT,),
    legs.click_action(5, 3), (legs.LEFT,),
    legs.click_action(5, 2), (legs.LEFT,),
    legs.click_action(5, 2),
    (legs.RIGHT,), (legs.RIGHT,), (legs.RIGHT,),
]


def avatar_cell(frame):
    for i, y in enumerate(legs.ROW_ANCHORS):
        for j, x in enumerate(legs.COL_ANCHORS):
            if int(frame[y][x]) in legs.AVATAR_COLORS:
                return i, j
    return None


def state_key(node, height):
    frame = node.frame()
    return (
        height,
        legs.moves_used(frame) % 2,
        avatar_cell(frame),
        tuple(
            legs._cell_shape(frame, i, j)
            for i in range(legs.GRID_ROWS)
            for j in range(legs.GRID_COLS)
        ),
    )


def choices(node):
    frame = node.frame()
    avatar = avatar_cell(frame)
    out = [(legs.LEFT,), (legs.RIGHT,)]
    if avatar is None:
        return out
    ai, aj = avatar
    for i, y in enumerate(legs.ROW_ANCHORS):
        for j, x in enumerate(legs.COL_ANCHORS):
            color = int(frame[y][x])
            if color in (7, 8, 12, 14):
                out.append((legs.CLICK, x, y))
            elif color == 15 and abs(i - ai) <= 2 and abs(j - aj) <= 2:
                out.append((legs.CLICK, x, y))
    return out


def search(root, max_states=5000, max_depth=50):
    base_level = root.levels_completed
    queue = deque([((), 0)])
    seen = {state_key(root, 0)}
    expanded = 0
    best = 0
    while queue and expanded < max_states:
        path, height = queue.popleft()
        node = root.clone()
        legs.run_actions(node, path)
        expanded += 1
        if expanded % 500 == 0:
            print("SEARCH", expanded, len(queue), len(path), best, flush=True)
        if len(path) >= max_depth or node.terminal():
            continue
        local_keys = set()
        for action in choices(node):
            child = node.clone()
            child.step(*action)
            if child.levels_completed > base_level:
                return list(path + (action,)), expanded, len(seen), best
            if child.terminal() or avatar_cell(child.frame()) is None:
                continue
            gained = legs.band_shift(node.frame(), child.frame())
            new_height = height + gained
            key = state_key(child, new_height)
            if key in seen or key in local_keys:
                continue
            local_keys.add(key)
            seen.add(key)
            best = max(best, new_height)
            queue.append((path + (action,), new_height))
    return [], expanded, len(seen), best


def probe(env):
    harness.resumed_solve(env)
    root = env.clone()
    legs.run_actions(root, PREFIX)
    print(
        "ROOT",
        legs.moves_used(root.frame()),
        avatar_cell(root.frame()),
        tuple("".join(row) for row in legs.band_grid(root.frame())),
        flush=True,
    )
    plan, expanded, seen, best = search(root)
    result = root.clone()
    legs.run_actions(result, plan)
    print(
        "FOUND",
        plan,
        "expanded",
        expanded,
        "seen",
        seen,
        "best",
        best,
        "level",
        result.levels_completed,
        "terminal",
        result.terminal(),
        flush=True,
    )


levels, path, error = harness.A.run_program("bp35", probe)
print("PROBE_RESULT", levels, len(path), error)
