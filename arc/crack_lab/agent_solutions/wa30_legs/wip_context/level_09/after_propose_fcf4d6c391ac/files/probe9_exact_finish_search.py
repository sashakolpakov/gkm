"""Exhaust the seven-action finish from the improved turn-62 state."""

import gkm_try

from perception import arr
from probe9_best_mutations import POSITION
from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import boxes, target_state


TO_TURN_60 = (
    direct_second_prefix()
    + COMBINED_DISMISS_PICK
    + [5]
    + POSITION
    + [2]
)


def search(env, max_depth=9, max_transitions=30000):
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
                        "EXACT_FINISH_WIN",
                        depth,
                        transitions,
                        child_path,
                        flush=True,
                    )
                    return child_path
                target = target_state(child.frame())
                score = (
                    len(target["filled"]),
                    -len(target["empty"]),
                )
                if best is None or score > best[0]:
                    best = (
                        score,
                        child_path,
                        boxes(child.frame(), 14),
                        boxes(child.frame(), 12),
                        target,
                    )
                if not child.terminal():
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, child_path),
                    )
                if transitions >= max_transitions:
                    print("EXACT_FINISH_LIMIT", transitions, best, flush=True)
                    return None
        frontier = list(next_states.values())
        print(
            "EXACT_FINISH_DEPTH",
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
    for action in TO_TURN_60:
        state.step(action)
    print(
        "EXACT_FINISH_START",
        len(TO_TURN_60),
        boxes(state.frame(), 14),
        target_state(state.frame()),
        flush=True,
    )
    print("EXACT_FINISH_RESULT", search(state), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
