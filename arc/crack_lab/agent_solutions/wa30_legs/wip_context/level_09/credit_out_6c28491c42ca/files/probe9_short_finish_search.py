"""Bounded reward search from the fast level-9 carried-block state."""

import gkm_try

from perception import arr
from probe9_prefix_shortcuts import direct_short_prefix, reach_level_9
from probe9_verify import boxes, target_state


DISMISS = [3] * 5 + [5]
PICK_AND_POSITION = [1, 3, 3, 5, 4] + [1] * 6 + [4] * 5


def search(env, max_depth=11, max_transitions=40000):
    base_level = env.levels_completed
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
                if child.levels_completed > base_level:
                    print(
                        "SHORT_FINISH_WIN",
                        depth,
                        transitions,
                        child_path,
                        flush=True,
                    )
                    return child_path
                target = target_state(child.frame())
                score = (len(target["filled"]), -len(target["empty"]))
                if best is None or score > best[0]:
                    best = (
                        score,
                        child_path,
                        target,
                        boxes(child.frame(), 12),
                    )
                if not child.terminal():
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, child_path),
                    )
                if transitions >= max_transitions:
                    print("SHORT_FINISH_LIMIT", transitions, best, flush=True)
                    return None
        frontier = list(next_states.values())
        print(
            "SHORT_FINISH_DEPTH",
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
    prefix = direct_short_prefix() + DISMISS + PICK_AND_POSITION
    for action in prefix:
        state.step(action)
    print(
        "SHORT_FINISH_START",
        len(prefix),
        target_state(state.frame()),
        boxes(state.frame(), 12),
        flush=True,
    )
    print("SHORT_FINISH_RESULT", search(state), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
