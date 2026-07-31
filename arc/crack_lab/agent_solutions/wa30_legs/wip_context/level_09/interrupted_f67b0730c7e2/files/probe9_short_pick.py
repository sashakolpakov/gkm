"""Find the shortest lower-local pickup after the fast level-9 dismissal."""

import gkm_try

from perception import arr
from probe9_prefix_shortcuts import direct_short_prefix, reach_level_9
from probe9_short_tail_search import avatar_cell
from probe9_verify import boxes, target_state


DISMISS = [3] * 5 + [5]


def lower_local_absent(frame):
    grid = arr(frame)
    return 4 not in grid[32:36, 4:8]


def search_pick(env, max_depth=8, max_transitions=20000):
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
                if lower_local_absent(child.frame()):
                    print(
                        "SHORT_PICK_WIN",
                        depth,
                        transitions,
                        child_path,
                        avatar_cell(child.frame()),
                        boxes(child.frame(), 4),
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
                    print("SHORT_PICK_LIMIT", transitions, flush=True)
                    return None
        frontier = list(next_states.values())
        print("SHORT_PICK_DEPTH", depth, len(frontier), transitions, flush=True)
    return None


def inspect(env):
    reach_level_9(env)
    state = env.clone()
    prefix = direct_short_prefix() + DISMISS
    for action in prefix:
        state.step(action)
    print(
        "SHORT_PICK_START",
        len(prefix),
        avatar_cell(state.frame()),
        boxes(state.frame(), 4),
        flush=True,
    )
    print("SHORT_PICK_RESULT", search_pick(state), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
