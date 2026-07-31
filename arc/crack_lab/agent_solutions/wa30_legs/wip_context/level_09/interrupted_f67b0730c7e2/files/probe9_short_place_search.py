"""Find the shortest settled placement after the fast local pickup."""

import gkm_try

from perception import arr
from probe9_prefix_shortcuts import direct_short_prefix, reach_level_9
from probe9_verify import boxes, target_state


PREFIX = (
    direct_short_prefix()
    + [3] * 5 + [5]
    + [1, 3, 3, 5]
)


def search(env, max_depth=10, max_transitions=20000):
    initial = set(target_state(env.frame())["filled"])
    frontier = [(env.clone(), [])]
    transitions = 0
    for depth in range(1, max_depth + 1):
        next_states = {}
        wins = []
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                transitions += 1
                filled = set(target_state(child.frame())["filled"])
                gained = tuple(sorted(filled - initial))
                if gained:
                    wins.append((
                        child_path,
                        gained,
                        boxes(child.frame(), 14),
                        boxes(child.frame(), 12),
                    ))
                    if len(wins) >= 20:
                        print(
                            "SHORT_PLACE_WIN",
                            depth,
                            transitions,
                            wins,
                            flush=True,
                        )
                        return wins
                if not child.terminal():
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, child_path),
                    )
                if transitions >= max_transitions:
                    print("SHORT_PLACE_LIMIT", transitions, wins, flush=True)
                    return wins or None
        if wins:
            print(
                "SHORT_PLACE_WIN",
                depth,
                transitions,
                wins,
                flush=True,
            )
            return wins
        frontier = list(next_states.values())
        print("SHORT_PLACE_DEPTH", depth, len(frontier), transitions, flush=True)
    return None


def inspect(env):
    reach_level_9(env)
    state = env.clone()
    for action in PREFIX:
        state.step(action)
    print(
        "SHORT_PLACE_START",
        len(PREFIX),
        boxes(state.frame(), 14),
        boxes(state.frame(), 4),
        target_state(state.frame()),
        flush=True,
    )
    print("SHORT_PLACE_RESULT", search(state), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
