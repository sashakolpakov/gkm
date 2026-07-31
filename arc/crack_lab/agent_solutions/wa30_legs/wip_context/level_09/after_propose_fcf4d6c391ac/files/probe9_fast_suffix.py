"""Bounded reward search from the optimized turn-60 level-9 placement."""

import gkm_try

from perception import arr
from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_verify import ReachedLevel9, StopAtLevel9, target_state


POSITION_BLOCK = [4] + [1] * 6 + [4] * 5


def search(start, max_depth=10, max_transitions=40000):
    base_level = start.levels_completed
    frontier = [(start.clone(), [])]
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
                        "FAST_SUFFIX_WIN",
                        depth,
                        transitions,
                        child_path,
                        flush=True,
                    )
                    return child_path
                target = target_state(child.frame())
                score = (len(target["filled"]), -len(target["empty"]))
                if best is None or score > best[0]:
                    best = (score, child_path, target)
                if not child.terminal():
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, child_path),
                    )
                if transitions >= max_transitions:
                    print(
                        "FAST_SUFFIX_LIMIT",
                        transitions,
                        best,
                        flush=True,
                    )
                    return None
        frontier = list(next_states.values())
        print(
            "FAST_SUFFIX_DEPTH",
            depth,
            len(frontier),
            transitions,
            best,
            flush=True,
        )
    return None


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    state = env.clone()
    prefix = (
        direct_second_prefix()
        + COMBINED_DISMISS_PICK
        + POSITION_BLOCK
    )
    for action in prefix:
        state.step(action)
    print(
        "FAST_SUFFIX_START",
        len(prefix),
        target_state(state.frame()),
        flush=True,
    )
    print("FAST_SUFFIX_RESULT", search(state), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
