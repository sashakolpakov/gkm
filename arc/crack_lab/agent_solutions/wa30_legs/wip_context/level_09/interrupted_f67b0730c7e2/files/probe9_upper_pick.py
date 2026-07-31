"""Search for a faster dismissal plus pickup of the upper local cargo."""

import gkm_try

from perception import arr
from probe9_candidate import direct_second_prefix
from probe9_picksearch import avatar_cell
from probe9_verify import ReachedLevel9, StopAtLevel9, boxes, target_state


def goal(env):
    grid = arr(env.frame())
    upper_cell = set(int(value) for value in grid[28:32, 4:8].flat)
    return (
        (5, 6) in target_state(env.frame())["filled"]
        and not boxes(env.frame(), 15)
        and avatar_cell(env.frame()) == (7, 2)
        and 4 not in upper_cell
        and 0 in upper_cell
    )


def search(env, max_depth=9, max_transitions=25000):
    frontier = [(env.clone(), [])]
    transitions = 0
    for depth in range(1, max_depth + 1):
        next_states = {}
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                transitions += 1
                if goal(child):
                    print(
                        "UPPER_PICK_WIN",
                        depth,
                        transitions,
                        child_path,
                        flush=True,
                    )
                    return child_path
                if not child.terminal():
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, child_path),
                    )
                if transitions >= max_transitions:
                    print("UPPER_PICK_LIMIT", transitions, flush=True)
                    return None
        frontier = list(next_states.values())
        print(
            "UPPER_PICK_DEPTH",
            depth,
            len(frontier),
            transitions,
            flush=True,
        )
    return None


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    state = env.clone()
    for action in direct_second_prefix():
        state.step(action)
    print("UPPER_PICK_RESULT", search(state), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
