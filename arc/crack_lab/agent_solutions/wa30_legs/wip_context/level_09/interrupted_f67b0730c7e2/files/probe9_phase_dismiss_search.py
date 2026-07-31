"""Search the one-turn-advanced thief collision after the compact prefix."""

import gkm_try

from perception import arr
from probe9_prefix_shortcuts import direct_short_prefix, reach_level_9
from probe9_verify import boxes, target_state


def avatar_cell(frame):
    avatar = boxes(frame, 14)
    if not avatar:
        return None
    row0, col0, row1, col1 = avatar[0]
    return ((row0 + row1) // 8, (col0 + col1) // 8)


def local_cargo_absent(frame):
    grid = arr(frame)
    return 4 not in grid[32:36, 4:8]


def phase_prefix():
    return direct_short_prefix() + [2] * 3


def goal(env):
    return (
        not boxes(env.frame(), 15)
        and avatar_cell(env.frame()) == (8, 2)
        and local_cargo_absent(env.frame())
    )


def search(env, max_depth=8, max_transitions=20000):
    frontier = [(env.clone(), [])]
    transitions = 0
    for depth in range(1, max_depth + 1):
        next_states = {}
        best = None
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                transitions += 1
                score = (
                    not bool(boxes(child.frame(), 15)),
                    local_cargo_absent(child.frame()),
                    avatar_cell(child.frame()) == (8, 2),
                    len(target_state(child.frame())["filled"]),
                )
                if best is None or score > best[0]:
                    best = (
                        score,
                        child_path,
                        avatar_cell(child.frame()),
                        boxes(child.frame(), 15),
                    )
                if goal(child):
                    print(
                        "PHASE_DISMISS_WIN",
                        depth,
                        transitions,
                        child_path,
                        target_state(child.frame()),
                        flush=True,
                    )
                    return child_path
                if not child.terminal():
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, child_path),
                    )
                if transitions >= max_transitions:
                    print("PHASE_DISMISS_LIMIT", transitions, best, flush=True)
                    return None
        frontier = list(next_states.values())
        print(
            "PHASE_DISMISS_DEPTH",
            depth,
            len(frontier),
            transitions,
            best,
            flush=True,
        )
    return None


def inspect(env):
    reach_level_9(env)
    state = env.clone()
    prefix = phase_prefix()
    for action in prefix:
        state.step(action)
    print(
        "PHASE_DISMISS_START",
        len(prefix),
        avatar_cell(state.frame()),
        boxes(state.frame(), 15),
        flush=True,
    )
    print("PHASE_DISMISS_RESULT", search(state), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
