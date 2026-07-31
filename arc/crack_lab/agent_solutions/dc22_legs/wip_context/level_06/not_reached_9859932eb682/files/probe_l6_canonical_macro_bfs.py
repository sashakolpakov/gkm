"""Movement-component BFS targeting the hidden 2x2 exit tile."""
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


DPAD_BY_POSITION = {
    (56, 34): (6, 50, 34),
    (60, 34): (6, 50, 40),
    (58, 32): (6, 46, 36),
    (58, 36): (6, 54, 36),
}


def frame_key(env):
    return perception.arr(env.frame())[:63].tobytes()


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


def movement_closure(root, base_level):
    queue = deque([(root.clone(), [])])
    seen = {frame_key(root)}
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
            key = frame_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, child_path))
    return states, None


def canonical_closure(root, base_level):
    states, win = movement_closure(root, base_level)
    ranked = sorted((frame_key(node), node, path) for node, path in states)
    key, representative, representative_path = ranked[0]
    revealed = next(
        (
            (path, exit_tiles(node))
            for node, path in states
            if exit_tiles(node)
        ),
        None,
    )
    return key, representative, representative_path, states, win, revealed


def observe(env):
    solve.solve(env)
    root = enter_right(env, 3)
    base_level = root.levels_completed
    queue = deque([(root.clone(), [], 3, 0)])
    seen = set()
    enqueued = set()
    print("CANONICAL_START", avatar_position(root), exit_tiles(root))
    while queue and len(seen) < 1200:
        node, path, phase, depth = queue.popleft()
        (
            component_key,
            representative,
            representative_path,
            walk_states,
            win_walk,
            revealed,
        ) = canonical_closure(node, base_level)
        state_key = component_key, phase
        if state_key in seen:
            continue
        seen.add(state_key)
        if win_walk is not None:
            print("CANONICAL_WIN", path + win_walk)
            return
        if revealed is not None:
            reveal_path, tiles = revealed
            print("CANONICAL_EXIT_REVEALED", path + reveal_path, tiles)
            return
        if depth >= 18 or len(path) >= 210:
            continue

        candidates = []
        # Selector state is position-independent; use one component anchor.
        candidates.append(
            (representative, representative_path, SELECTOR, (phase + 1) % 4)
        )
        # Rotators and the phased bridge can carry or teleport the avatar.
        for walked, walk_path in walk_states:
            candidates.append((walked, walk_path, MAIN, phase))
            candidates.append((walked, walk_path, TOP, phase))
            dpad = DPAD_BY_POSITION.get(avatar_position(walked))
            if dpad is not None:
                candidates.append((walked, walk_path, dpad, phase))

        local = set()
        for walked, walk_path, control, child_phase in candidates:
            child = walked.clone()
            child.step(*control)
            child_path = path + walk_path + [control]
            if child.levels_completed > base_level:
                print("CANONICAL_WIN", child_path)
                return
            exact = frame_key(child), child_phase
            if exact in local or exact in enqueued:
                continue
            local.add(exact)
            enqueued.add(exact)
            queue.append((child, child_path, child_phase, depth + 1))
        if len(seen) % 25 == 0:
            print(
                "CANONICAL_PROGRESS", len(seen), len(queue),
                depth, len(path), avatar_position(representative),
            )
    print("CANONICAL_DONE", len(seen), len(queue))


arena.run_program("dc22", observe)
