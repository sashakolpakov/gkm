"""Macro BFS over movement components with normalized avatar pixels."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import (
    MAIN,
    SELECTOR,
    TOP,
    avatar_position,
    enter_right,
)
from probe_l6_exact_crossings import placements_with_paths
from probe_l6_reach_by_ring import vertical_entry


DPAD_BY_POSITION = {
    (56, 34): (6, 50, 34),
    (60, 34): (6, 50, 40),
    (58, 32): (6, 46, 36),
    (58, 36): (6, 54, 36),
}


def normalized_world(env):
    frame = perception.arr(env.frame())[:63].copy()
    for blob in perception.connected_components(
        frame, colors=(14,), min_area=1
    ):
        if blob.area == 4 and blob.bbox[1] < 40:
            r0, c0, r1, c1 = blob.bbox
            frame[r0:r1 + 1, c0:c1 + 1] = 2
            break
    return frame.tobytes()


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


def interaction_region(env):
    position = avatar_position(env)
    if position is None:
        return ("hidden",)
    row, col = position
    portal_entries = {
        (4, 4): "top_portal",
        (48, 18): "hub_portal",
        (52, 32): "right0_portal",
        (58, 34): "right3_portal",
    }
    if position in portal_entries:
        return (portal_entries[position],)
    if col >= 32:
        return ("right0",) if row < 54 else ("right3",)
    if row < 14:
        return ("top",)
    if row >= 54 and col < 22:
        return ("phased_bridge", row, col)
    if 24 <= row < 44:
        return ("rotator", row, col)
    if row >= 44 and col >= 18:
        return ("hub",)
    if row < 24:
        return ("upper",)
    return ("lower",)


def movement_closure(root, base_level):
    start = avatar_position(root)
    queue = deque([(root.clone(), [])])
    seen = {start}
    states = []
    while queue:
        node, path = queue.popleft()
        states.append((node, path))
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_path = path + [action]
            if child.levels_completed > base_level:
                return states, child_path
            position = avatar_position(child)
            if position in seen:
                continue
            seen.add(position)
            queue.append((child, child_path))
    return states, None


def observe(env):
    solve.solve(env)
    if len(sys.argv) > 1 and sys.argv[1] == "physical":
        placement = placements_with_paths(enter_right(env, 3))[0][0]
        root = vertical_entry(placement)
    else:
        root = enter_right(env, 3)
    base_level = root.levels_completed
    queue = deque([(root.clone(), [], 3, 0)])
    seen = set()
    queued = {
        (normalized_world(root), interaction_region(root), 3)
    }
    print("SYMBOLIC_MACRO_START", avatar_position(root))
    while queue and len(seen) < 500:
        node, path, phase, depth = queue.popleft()
        walk_states, win_walk = movement_closure(node, base_level)
        component = tuple(
            sorted(
                position for position in (
                    avatar_position(walked)
                    for walked, _ in walk_states
                )
                if position is not None
            )
        )
        state_key = normalized_world(node), component, phase
        if state_key in seen:
            continue
        seen.add(state_key)
        if win_walk is not None:
            print("SYMBOLIC_MACRO_WIN", path + win_walk)
            return
        if exit_tiles(node):
            print("SYMBOLIC_MACRO_EXIT", path, exit_tiles(node))
            return
        if depth >= 16 or len(path) >= 220:
            continue

        candidates = []
        representative, representative_path = walk_states[0]
        candidates.append(
            (representative, representative_path, SELECTOR, (phase + 1) % 4)
        )
        candidates.append((representative, representative_path, MAIN, phase))
        candidates.append((representative, representative_path, TOP, phase))
        for walked, walk_path in walk_states:
            position = avatar_position(walked)
            if position in {
                (4, 4), (32, 8), (48, 18), (52, 32), (58, 34)
            }:
                candidates.append((walked, walk_path, MAIN, phase))
            if (
                position is not None
                and position[0] >= 54
                and position[1] < 22
            ):
                candidates.append((walked, walk_path, TOP, phase))

        local = set()
        for walked, walk_path, control, child_phase in candidates:
            child = walked.clone()
            child.step(*control)
            child_path = path + walk_path + [control]
            if child.levels_completed > base_level:
                print("SYMBOLIC_MACRO_WIN", child_path)
                return
            child_key = (
                normalized_world(child),
                interaction_region(child),
                child_phase,
            )
            if child_key in local or child_key in queued:
                continue
            local.add(child_key)
            queued.add(child_key)
            queue.append((child, child_path, child_phase, depth + 1))
        if len(seen) % 5 == 0:
            print(
                "SYMBOLIC_MACRO_PROGRESS", len(seen), len(queue),
                depth, len(path), len(walk_states),
            )
    print("SYMBOLIC_MACRO_DONE", len(seen), len(queue))


arena.run_program("dc22", observe)
