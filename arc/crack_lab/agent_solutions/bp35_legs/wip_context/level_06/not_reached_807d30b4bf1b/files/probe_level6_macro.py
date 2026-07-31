"""Best-first room search retaining no-delta interaction identity."""
import heapq
import itertools
from collections import deque

import gkm_try as harness
import legs


def avatar_cell(frame):
    for i, y in enumerate(legs.ROW_ANCHORS):
        for j, x in enumerate(legs.COL_ANCHORS):
            if int(frame[y][x]) in legs.AVATAR_COLORS:
                return i, j
    return None


def cells(frame):
    return tuple(
        legs._cell_shape(frame, i, j)
        for i in range(legs.GRID_ROWS)
        for j in range(legs.GRID_COLS)
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
            if color in (7, 8):
                out.append((legs.CLICK, x, y))
            elif color in (12, 14, 15):
                if abs(i - ai) <= 2 and abs(j - aj) <= 2:
                    out.append((legs.CLICK, x, y))
    return out


def search(env, max_states=12000, max_depth=80):
    base_level = env.levels_completed
    root = env.clone()
    counter = itertools.count()

    def key(node, latent):
        frame = node.frame()
        return (
            legs.moves_used(frame) % 2,
            avatar_cell(frame),
            cells(frame),
            latent,
        )

    def reconstruct(path):
        node = root.clone()
        legs.run_actions(node, path)
        return node

    frontier = [(0, 0, next(counter), (), None)]
    seen_progress = {key(root, None)}
    expanded = 0
    best = 0
    while frontier and expanded < max_states:
        neg_height, _, _, prefix, prefix_latent = heapq.heappop(frontier)
        stage = reconstruct(prefix)
        local_queue = deque([((), prefix_latent)])
        local_seen = {key(stage, prefix_latent)}
        while local_queue and expanded < max_states:
            suffix, latent = local_queue.popleft()
            path = prefix + suffix
            node = stage.clone()
            legs.run_actions(node, suffix)
            expanded += 1
            if expanded % 1000 == 0:
                print(
                    "SEARCH", expanded, len(frontier), len(local_queue),
                    -neg_height, len(path), flush=True,
                )
            if len(path) >= max_depth or node.terminal():
                continue
            before_cells = cells(node.frame())
            before_avatar = avatar_cell(node.frame())
            for action in choices(node):
                child = node.clone()
                child.step(*action)
                new_path = path + (action,)
                if child.levels_completed > base_level:
                    return list(new_path), expanded, best
                if child.terminal() or avatar_cell(child.frame()) is None:
                    continue
                gain = max(0, legs.band_shift(node.frame(), child.frame()))
                child_cells = cells(child.frame())
                child_avatar = avatar_cell(child.frame())
                new_latent = latent
                if (
                    action[0] == legs.CLICK
                    and child_cells == before_cells
                    and child_avatar == before_avatar
                    and gain == 0
                ):
                    i = min(
                        range(legs.GRID_ROWS),
                        key=lambda r: abs(legs.ROW_ANCHORS[r] - action[2]),
                    )
                    j = min(
                        range(legs.GRID_COLS),
                        key=lambda c: abs(legs.COL_ANCHORS[c] - action[1]),
                    )
                    color = int(node.frame()[action[2]][action[1]])
                    new_latent = (color, i, j)
                state = key(child, new_latent)
                new_suffix = suffix + (action,)
                if gain:
                    if state in seen_progress:
                        continue
                    seen_progress.add(state)
                    height = -neg_height + gain
                    if height > best:
                        best = height
                        print(
                            "BEST", best, len(new_path), new_path,
                            tuple("".join(row)
                                  for row in legs.band_grid(child.frame())),
                            flush=True,
                        )
                    heapq.heappush(
                        frontier,
                        (-height, len(new_path), next(counter),
                         new_path, new_latent),
                    )
                elif len(new_suffix) < 16 and state not in local_seen:
                    local_seen.add(state)
                    local_queue.append((new_suffix, new_latent))
    return [], expanded, best


def probe(env):
    harness.resumed_solve(env)
    plan, expanded, best = search(env)
    result = env.clone()
    legs.run_actions(result, plan)
    print(
        "FOUND", plan, "expanded", expanded, "best", best,
        "level", result.levels_completed, "terminal", result.terminal(),
        flush=True,
    )


levels, path, error = harness.A.run_program("bp35", probe)
print("PROBE_RESULT", levels, len(path), error)
