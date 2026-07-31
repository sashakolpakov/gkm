"""Bounded reward search from the final level-9 staging positions."""

import gkm_try

from perception import arr
from probe9_candidate import direct_second_prefix
from probe9_verify import (
    ReachedLevel9,
    StopAtLevel9,
    boxes,
    target_state,
)


def reward_search(env, max_depth, max_transitions=20000):
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
                        "SUFFIX_WIN",
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
                    print("SUFFIX_LIMIT", transitions, best, flush=True)
                    return None
        frontier = list(next_states.values())
        print(
            "SUFFIX_DEPTH",
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

    base = direct_second_prefix() + [2, 2, 3, 5]
    stages = {
        "short_pickup": (
            base + [3] * 3 + [5, 4]
            + [1] * 6 + [4] * 5
        ),
    }
    for name, prefix in stages.items():
        state = env.clone()
        for action in prefix:
            state.step(action)
        print(
            "SUFFIX_STAGE",
            name,
            len(prefix),
            {
                "avatar": boxes(state.frame(), 14),
                "cargo": boxes(state.frame(), 4),
                "target": target_state(state.frame()),
            },
            flush=True,
        )
        suffix = reward_search(
            state, 70 - len(prefix), max_transitions=50000
        )
        print("SUFFIX_RESULT", name, suffix, flush=True)


gkm_try.A.run_program("wa30", inspect)
