"""Bounded clean-room macro search over level-6 movement components."""
from collections import deque
import hashlib
import heapq
import itertools
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve


A = (6, 56, 8)
B = (6, 50, 26)
S = (6, 50, 46)
DPAD = {
    (56, 34): (6, 50, 34),
    (60, 34): (6, 50, 40),
    (58, 32): (6, 46, 36),
    (58, 36): (6, 54, 36),
}
PORTAL_AND_PIVOT_SITES = {
    (4, 4),
    (18, 8),
    (32, 8),
    (48, 18),
    (48, 20),
    (52, 32),
    (58, 34),
}
A_INTERACTION_SITES = {
    (52, 18),
    (54, 14),
    (54, 16),
    (54, 18),
    (56, 10),
    (56, 12),
    (56, 14),
    (58, 10),
}
TO_BRIDGE = [3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3]
TO_REMOTE_PAD = [4, 1, 4, 1, 4, 4, 1, 1, 1]
PHYSICAL_ENTRY = (
    [3] * 5
    + [A] * 4
    + [2, 2, 3, 3, 3, 2, 3, A, 1, A, 1, 1, B]
    + [1] * 17
    + [3]
    + [2] * 11
)


def entry_path():
    path = (
        [A] * 4
        + TO_BRIDGE
        + [A, 1, 3, A]
        + [1] * 6
        + [B]
        + [1] * 13
        + [S] * 3
        + [2] * 19
        + [A, A, 4, A, 2, A]
        + TO_REMOTE_PAD
        + [B]
    )
    return path


def step(env, action):
    if isinstance(action, tuple):
        env.step(*action)
    else:
        env.step(action)


def apply(env, path):
    for action in path:
        step(env, action)


def frame_key(env):
    return np.asarray(env.frame())[:63].tobytes()


def normalized_world(env):
    frame = np.asarray(env.frame())[:63].copy()
    position = avatar_position(env)
    if position is not None:
        row, col = position
        frame[row:row + 2, col:col + 2] = 2
    return frame.tobytes()


def avatar_position(env):
    frame = np.asarray(env.frame())
    for row in range(0, 62, 2):
        for col in range(0, 64, 2):
            if np.all(frame[row:row + 2, col:col + 2] == 14):
                return row, col
    return None


def interaction_region(env):
    position = avatar_position(env)
    if position is None:
        return ("hidden",)
    if position in PORTAL_AND_PIVOT_SITES:
        return ("site", position)
    row, col = position
    if row < 14 and col >= 32:
        return ("goal_platform",)
    if col >= 32:
        return ("right0",) if row < 54 else ("right3",)
    if row < 14:
        return ("top",)
    if row < 24:
        return ("upper",)
    if 24 <= row < 44:
        return ("rotator",)
    if row >= 54 and col < 22:
        return ("phased_bridge", row // 4, col // 4)
    if row >= 44 and col >= 18:
        return ("hub",)
    return ("lower",)


def queued_key(env):
    return normalized_world(env), interaction_region(env)


def solid_exit_count(env):
    frame = np.asarray(env.frame())
    return sum(
        bool(np.all(frame[row:row + 2, col:col + 2] == 11))
        for row in range(0, 62, 2)
        for col in range(0, 64, 2)
    )


def movement_closure(root, base_level):
    """Infer ordinary walking paths from pixels; replay interaction paths."""
    frame = np.asarray(root.frame())
    start = avatar_position(root)
    if start is None:
        return {None: []}, None
    queue = deque([start])
    by_position = {start: []}
    directions = ((1, -2, 0), (2, 2, 0), (3, 0, -2), (4, 0, 2))
    while queue:
        row, col = queue.popleft()
        for action, dr, dc in directions:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < 62 and 0 <= nc < 64):
                continue
            block = frame[nr:nr + 2, nc:nc + 2]
            support = sum(
                int(value) not in {0, 4, 5, 15}
                for value in block.flat
            )
            patterned_dpad = (
                nr in {56, 58, 60} and nc in {32, 34, 36}
            )
            if support < 2 and not patterned_dpad:
                continue
            position = (nr, nc)
            if position in by_position:
                continue
            by_position[position] = by_position[(row, col)] + [action]
            queue.append(position)

    for position, path in by_position.items():
        row, col = position
        if not np.all(frame[row:row + 2, col:col + 2] == 11):
            continue
        probe = root.clone()
        apply(probe, path)
        if probe.levels_completed > base_level:
            return by_position, path
    return by_position, None


def component_digest(root, positions):
    digest = hashlib.sha1()
    digest.update(normalized_world(root))
    digest.update(repr(tuple(sorted(positions, key=repr))).encode())
    return digest.digest()


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    prefix = (
        PHYSICAL_ENTRY
        if len(sys.argv) > 1 and sys.argv[1] == "physical"
        else entry_path()
    )
    apply(env, prefix)
    print(
        "SEARCH_ROOT", "suffix", len(prefix),
        "avatar", avatar_position(env),
        "exit", solid_exit_count(env),
        flush=True,
    )

    serial = itertools.count()
    heap = [(0, len(prefix), next(serial), env.clone(), prefix)]
    seen_components = set()
    queued_exact = {queued_key(env)}
    best = (99, -1, 0, 0)
    while heap and len(seen_components) < 1400:
        depth, _, _, node, path = heapq.heappop(heap)
        by_position, winning_walk = movement_closure(node, base_level)
        positions = set(by_position)
        component = component_digest(node, positions)
        if component in seen_components:
            continue
        seen_components.add(component)
        if winning_walk is not None:
            winner = path + winning_walk
            print("COMPONENT_WIN", len(winner), winner, flush=True)
            return

        visible = [position for position in positions if position is not None]
        metric = (
            min((row for row, _ in visible), default=99),
            max((col for _, col in visible), default=-1),
            len(visible),
            solid_exit_count(node),
        )
        improved = (
            metric[0] < best[0]
            or metric[1] > best[1]
            or metric[2] > best[2]
            or metric[3] > best[3]
        )
        if improved:
            best = (
                min(best[0], metric[0]),
                max(best[1], metric[1]),
                max(best[2], metric[2]),
                max(best[3], metric[3]),
            )
            print(
                "COMPONENT_PROGRESS", len(seen_components),
                "depth", depth, "path_len", len(path),
                "metric", metric, "best", best, "path", path,
                flush=True,
            )
        if depth >= 24:
            continue

        interesting_walks = {tuple()}
        for position, walk in by_position.items():
            if position in PORTAL_AND_PIVOT_SITES:
                interesting_walks.add(tuple(walk))
            if position in A_INTERACTION_SITES:
                interesting_walks.add(tuple(walk))
            if position in DPAD:
                interesting_walks.add(tuple(walk))

        outcomes = {}
        tested = set()
        for walk_tuple in interesting_walks:
            walk = list(walk_tuple)
            walked = node.clone()
            apply(walked, walk)
            if walked.levels_completed > base_level:
                print(
                    "COMPONENT_WIN", len(path + walk), path + walk,
                    flush=True,
                )
                return
            actual = avatar_position(walked)
            controls = []
            if not walk:
                controls.extend((A, B, S))
            if actual in PORTAL_AND_PIVOT_SITES:
                controls.extend((A, B, S))
            if actual in A_INTERACTION_SITES:
                controls.append(A)
            if actual in DPAD:
                controls.append(DPAD[actual])
            for control in controls:
                child_path = path + walk + [control]
                if len(child_path) > 295:
                    continue
                exact_test = frame_key(walked), control
                if exact_test in tested:
                    continue
                tested.add(exact_test)
                child = walked.clone()
                before = exact_test[0]
                step(child, control)
                if child.levels_completed > base_level:
                    print(
                        "COMPONENT_WIN", len(child_path), child_path,
                        flush=True,
                    )
                    return
                after = frame_key(child)
                if after == before:
                    continue
                previous = outcomes.get(after)
                edge_cost = 0 if control in set(DPAD.values()) else 1
                if previous is None or len(child_path) < len(previous[1]):
                    outcomes[after] = (child, child_path, edge_cost)

        for _, (child, child_path, edge_cost) in outcomes.items():
            normalized = queued_key(child)
            if normalized in queued_exact:
                continue
            queued_exact.add(normalized)
            heapq.heappush(
                heap,
                (
                    depth + edge_cost,
                    len(child_path),
                    next(serial),
                    child,
                    child_path,
                ),
            )
        if len(seen_components) % 25 == 0:
            print(
                "COMPONENT_STATES", len(seen_components),
                "queue", len(heap), "depth", depth,
                "path_len", len(path), "best", best,
                flush=True,
            )
    print(
        "COMPONENT_DONE", len(seen_components), len(heap), best,
        flush=True,
    )


arena.run_program("dc22", observe)
