"""Test shortest movement orderings for the final level-9 cargo."""

from itertools import combinations

import gkm_try

from perception import arr
from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_verify import ReachedLevel9, StopAtLevel9, boxes, target_state


SUCCESSFUL_NINE = [4, 5, 1, 3, 2, 5, 2, 5, 1]


def finish(base, suffix):
    clone = base.clone()
    base_level = clone.levels_completed
    route = list(suffix) + [5] * max(0, 69 - 60 - len(suffix))
    best = 0
    for action in route:
        if clone.terminal() or clone.levels_completed > base_level:
            break
        clone.step(action)
        best = max(best, len(target_state(clone.frame())["filled"]))
    return clone.levels_completed - base_level, best, target_state(clone.frame())


def reward_search(states, max_depth=9, max_transitions=40000):
    frontier = [
        (state.clone(), path, [])
        for state, path in states
    ]
    base_level = frontier[0][0].levels_completed
    transitions = 0
    for depth in range(1, max_depth + 1):
        next_states = {}
        best = None
        for node, position_path, suffix in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_suffix = suffix + [action]
                transitions += 1
                if child.levels_completed > base_level:
                    print(
                        "POSITION_SEARCH_WIN",
                        {
                            "depth": depth,
                            "transitions": transitions,
                            "path": position_path,
                            "suffix": child_suffix,
                        },
                        flush=True,
                    )
                    return position_path, child_suffix
                target = target_state(child.frame())
                score = (len(target["filled"]), -len(target["empty"]))
                if best is None or score > best[0]:
                    best = (score, position_path, child_suffix, target)
                if not child.terminal():
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, position_path, child_suffix),
                    )
                if transitions >= max_transitions:
                    print(
                        "POSITION_SEARCH_LIMIT",
                        transitions,
                        best,
                        flush=True,
                    )
                    return None
        frontier = list(next_states.values())
        print(
            "POSITION_SEARCH_DEPTH",
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

    base = env.clone()
    prefix = (
        direct_second_prefix()
        + COMBINED_DISMISS_PICK
        + [5, 4]
    )
    for action in prefix:
        base.step(action)

    states = {}
    for up_positions in combinations(range(11), 6):
        up_positions = set(up_positions)
        path = [
            1 if index in up_positions else 4
            for index in range(11)
        ]
        clone = base.clone()
        for action in path:
            clone.step(action)
        states.setdefault(arr(clone.frame()).tobytes(), (clone, path))
    print("POSITION_STATES", len(states), flush=True)

    suffixes = [SUCCESSFUL_NINE]
    suffixes.extend(
        SUCCESSFUL_NINE[:index] + SUCCESSFUL_NINE[index + 1:]
        for index in range(len(SUCCESSFUL_NINE))
    )
    suffixes.extend(
        (
            [5, 1, 3, 2, 5, 2, 5, 1],
            [2, 5, 1],
            [2, 2, 5, 1],
        )
    )
    best = None
    for state, path in states.values():
        for suffix in suffixes:
            reward, filled, target = finish(state, suffix)
            score = (reward, filled)
            if best is None or score > best[0]:
                best = (
                    score,
                    path,
                    suffix,
                    boxes(state.frame(), 14),
                    target,
                )
            if reward:
                print(
                    "POSITION_WIN",
                    {"path": path, "suffix": suffix, "target": target},
                    flush=True,
                )
                return
    print("POSITION_BEST", best, flush=True)
    print(
        "POSITION_SEARCH_RESULT",
        reward_search(list(states.values())),
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
