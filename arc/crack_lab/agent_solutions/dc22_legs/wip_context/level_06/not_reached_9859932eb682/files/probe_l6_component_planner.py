"""Shortest-path search over level-6 walking components and visible controls."""
from collections import deque
import heapq
import itertools
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import enter_right


A = (6, 56, 8)
B = (6, 50, 26)
S = (6, 50, 46)
DPAD = {
    (56, 34): ("U", (6, 50, 34)),
    (60, 34): ("D", (6, 50, 40)),
    (58, 32): ("L", (6, 46, 36)),
    (58, 36): ("R", (6, 54, 36)),
}
PORTALS = {
    (4, 4),
    (48, 18),
    (52, 32),
    (58, 34),
}
PHYSICAL_ENTRY = (
    [3] * 5
    + [A] * 4
    + [2, 2, 3, 3, 3, 2, 3, A, 1, A, 1, 1, B]
    + [1] * 17
    + [3]
    + [2] * 11
)
PHYSICAL_TO_HUB = [
    A, A,
    2, 2, 2, 2, 2, 2,
    4, 2, 2, A, 2, A,
]
FULL_AVATAR_CONTEXTS = False


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def avatar(env):
    frame = perception.arr(env.frame())[:62, :40]
    mask = frame == 14
    squares = np.argwhere(
        mask[:-1, :-1]
        & mask[1:, :-1]
        & mask[:-1, 1:]
        & mask[1:, 1:]
    )
    if len(squares):
        return tuple(int(value) for value in squares[0])
    return None


def visible_key(env):
    return perception.arr(env.frame())[:63].tobytes()


def solid_exits(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(11,), min_area=4
        )
        if blob.area == 4
        and blob.size == (2, 2)
        and blob.bbox[1] < 40
    )


def interaction_region(position):
    if position is None:
        return ("hidden",)
    if position in PORTALS:
        return ("portal", position)
    row, col = position
    if row >= 54 and col >= 30:
        return ("controller", position)
    if row < 14:
        return ("top",)
    if 24 <= row < 44:
        return ("rotator", position)
    if row >= 44 and col < 18:
        return ("phased_bridge", position)
    if row >= 44 and col >= 18:
        return ("hub",)
    return ("upper",)


def outcome_signature(before_frame, child):
    """Ignore avatar redraws while retaining real control-state changes."""
    after_frame = perception.arr(child.frame())[:63]
    changed = before_frame != after_frame
    changed &= before_frame != 14
    changed &= after_frame != 14
    rows, cols = np.where(changed)
    delta = tuple(
        (int(row), int(col), int(after_frame[row, col]))
        for row, col in zip(rows, cols)
    )
    after_avatar = avatar(child)
    context = (
        (
            "component",
            tuple(sorted(
                position for position in walk_closure(child)
                if position is not None
            )),
        )
        if FULL_AVATAR_CONTEXTS
        else interaction_region(after_avatar)
    )
    return delta, context


def walk_closure(root):
    """Infer bounded two-pixel walking paths from the visible support cells."""
    frame = perception.arr(root.frame())
    start = avatar(root)
    if start is None:
        return {None: []}
    queue = deque([start])
    by_position = {start: []}
    directions = (
        (1, -2, 0),
        (2, 2, 0),
        (3, 0, -2),
        (4, 0, 2),
    )
    while queue and len(by_position) < 120:
        row, col = queue.popleft()
        for action, dr, dc in directions:
            target = row + dr, col + dc
            nr, nc = target
            if target in by_position or not (0 <= nr < 62 and 0 <= nc < 40):
                continue
            block = frame[nr:nr + 2, nc:nc + 2]
            support = sum(
                int(value) not in {0, 4, 5, 15}
                for value in block.flat
            )
            if support < 2:
                continue
            by_position[target] = by_position[(row, col)] + [action]
            queue.append(target)
    return by_position


def replay_walk(root, path, base_level):
    child = root.clone()
    for index, direction in enumerate(path):
        child.step(direction)
        if child.levels_completed > base_level:
            return child, path[:index + 1]
    return child, None


def control_sites(by_position):
    """Yield the contexts that can differ under each visible control."""
    positioned = [
        (position, path)
        for position, path in by_position.items()
        if position is not None
    ]
    if not positioned:
        return

    if FULL_AVATAR_CONTEXTS:
        for position, path in positioned:
            yield "A", position, path, A
            yield "B", position, path, B
        canonical_position, canonical = min(positioned)
        yield "S", canonical_position, canonical, S
        return

    # A changes the phased bridges globally, and can also move an avatar
    # occupying the lower incremental bridge.
    canonical_position, canonical = min(positioned)
    yield "A", canonical_position, canonical, A
    for position, path in positioned:
        row, col = position
        if row >= 54 and col < 22:
            yield "A", position, path, A

    # S advances the destination globally.  Retain portal occupancy because
    # the selected endpoint can move beneath the avatar.
    yield "S", canonical_position, canonical, S
    for position, path in positioned:
        if position in PORTALS:
            yield "S", position, path, S

    # Away from a portal, the reproduced context matrix has one global B
    # outcome (some occupied cells merely block it).  Keep a few ordered
    # representatives so at least one is an active cell, plus every portal.
    for position, path in sorted(positioned)[:4]:
        yield "B", position, path, B
    for position, path in positioned:
        if position in PORTALS:
            yield "B", position, path, B

    # Ring motion is enabled only at the matching arm of the remote D-pad.
    for position, path in positioned:
        if position in DPAD:
            label, point = DPAD[position]
            yield label, position, path, point


def observe(env):
    global FULL_AVATAR_CONTEXTS
    solve.solve(env)
    base_level = env.levels_completed
    mode = sys.argv[1] if sys.argv[1:] else "initial"
    if mode == "controller":
        root = enter_right(env, 3)
    elif mode in {"physical_full", "hub_full"}:
        FULL_AVATAR_CONTEXTS = True
        root = env.clone()
        for action in PHYSICAL_ENTRY:
            step(root, action)
        if mode == "hub_full":
            for action in PHYSICAL_TO_HUB:
                step(root, action)
    else:
        root = env.clone()
    counter = itertools.count()
    heap = [(0, next(counter), root, [])]
    seen_components = set()
    seen_region_sets = set()
    enqueued_frames = {visible_key(root)}
    best = (10**9, -1, -1)

    while heap and len(seen_components) < 5000:
        _, _, root, root_path = heapq.heappop(heap)
        by_position = walk_closure(root)
        canonical_position = min(
            position for position in by_position if position is not None
        ) if any(position is not None for position in by_position) else None
        canonical, winning_walk = replay_walk(
            root, by_position[canonical_position], base_level
        )
        if winning_walk is not None:
            print("PLANNER_WIN", root_path + winning_walk, flush=True)
            return
        component_key = visible_key(canonical)
        if component_key in seen_components:
            continue
        seen_components.add(component_key)
        if len(seen_components) <= 10 or len(by_position) > 70:
            print(
                "PLANNER_NODE", len(seen_components), len(heap),
                "positions", len(by_position),
                "path_len", len(root_path), flush=True,
            )

        exits = sorted(solid_exits(root))
        positions = [
            position for position in by_position
            if position is not None
        ]
        region_set = tuple(sorted({
            interaction_region(position)[0]
            for position in positions
        }))
        if region_set not in seen_region_sets:
            seen_region_sets.add(region_set)
            print(
                "PLANNER_REGIONS", region_set,
                "positions", len(positions),
                "path_len", len(root_path), "path", root_path, flush=True,
            )
        metric = (
            min(position[0] for position in positions),
            max(position[1] for position in positions),
            len(positions),
        )
        if exits:
            print(
                "PLANNER_EXIT", root_path, exits, metric,
                "states", len(seen_components), flush=True,
            )
            for r0, c0, _, _ in exits:
                goal_path = by_position.get((r0, c0))
                if goal_path is None:
                    continue
                _, winning_walk = replay_walk(root, goal_path, base_level)
                if winning_walk is not None:
                    print(
                        "PLANNER_WIN", root_path + winning_walk,
                        "visible_exit", (r0, c0), flush=True,
                    )
                    return
        improved = (
            metric[0] < best[0]
            or metric[1] > best[1]
            or metric[2] > best[2]
        )
        if improved:
            best = (
                min(best[0], metric[0]),
                max(best[1], metric[1]),
                max(best[2], metric[2]),
            )
            print(
                "PLANNER_PROGRESS", len(seen_components), len(heap),
                "metric", metric, "best", best,
                "path_len", len(root_path), "path", root_path, flush=True,
            )

        local_outcomes = set()
        for label, target, walk_path, control in control_sites(by_position):
            child_path = root_path + walk_path + [control]
            if len(child_path) > 295:
                continue
            walked, winning_walk = replay_walk(root, walk_path, base_level)
            if winning_walk is not None:
                print("PLANNER_WIN", root_path + winning_walk, flush=True)
                return
            if avatar(walked) != target:
                continue
            before_frame = perception.arr(walked.frame())[:63].copy()
            before = before_frame.tobytes()
            child = walked.clone()
            step(child, control)
            if child.levels_completed > base_level:
                print(
                    "PLANNER_WIN", child_path,
                    "control", label,
                    "states", len(seen_components), flush=True,
                )
                return
            after = visible_key(child)
            signature = outcome_signature(before_frame, child)
            if after == before or signature in local_outcomes:
                continue
            local_outcomes.add(signature)
            if after in enqueued_frames:
                continue
            enqueued_frames.add(after)
            heapq.heappush(
                heap,
                (len(child_path), next(counter), child, child_path),
            )

        if len(seen_components) % 50 == 0:
            print(
                "PLANNER_STATES", len(seen_components), len(heap),
                "path_len", len(root_path), "best", best, flush=True,
            )

    print(
        "PLANNER_DONE", len(seen_components), len(heap),
        "best", best, flush=True,
    )


arena.run_program("dc22", observe)
