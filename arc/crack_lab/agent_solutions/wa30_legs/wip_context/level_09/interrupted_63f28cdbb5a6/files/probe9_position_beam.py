"""Beam-search the final level-9 courier/placement ordering from turn 58."""

import gkm_try

from perception import arr
from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_prefix_shortcuts import reach_level_9
from probe9_short_position_orders import positioned_states
from probe9_verify import boxes, target_state


def score(env):
    target = target_state(env.frame())
    occupied = 8 - len(target["empty"])
    target_helpers = sum(
        12 in signature
        for signature in target["signatures"].values()
    )
    target_avatar = sum(
        14 in signature
        for signature in target["signatures"].values()
    )
    return (
        len(target["filled"]),
        target_helpers,
        -target_avatar,
        occupied,
    )


def search(states, max_depth=11, beam_width=400, max_transitions=25000):
    base_level = states[0][0].levels_completed
    frontier = [
        (state.clone(), position, [])
        for state, position in states
    ]
    transitions = 0
    for depth in range(1, max_depth + 1):
        next_states = {}
        best = None
        for node, position, suffix in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_suffix = suffix + [action]
                transitions += 1
                if child.levels_completed > base_level:
                    print(
                        "POSITION_BEAM_WIN",
                        {
                            "depth": depth,
                            "transitions": transitions,
                            "position": position,
                            "suffix": child_suffix,
                        },
                        flush=True,
                    )
                    return position, child_suffix
                child_score = score(child)
                if best is None or child_score > best[0]:
                    best = (
                        child_score,
                        position,
                        child_suffix,
                        target_state(child.frame()),
                        boxes(child.frame(), 12),
                    )
                if not child.terminal():
                    key = arr(child.frame()).tobytes()
                    existing = next_states.get(key)
                    candidate = (child_score, child, position, child_suffix)
                    if existing is None or child_score > existing[0]:
                        next_states[key] = candidate
                if transitions >= max_transitions:
                    print("POSITION_BEAM_LIMIT", transitions, best, flush=True)
                    return None
        ranked = sorted(
            next_states.values(),
            key=lambda item: item[0],
            reverse=True,
        )[:beam_width]
        frontier = [
            (child, position, suffix)
            for _, child, position, suffix in ranked
        ]
        print(
            "POSITION_BEAM_DEPTH",
            depth,
            len(frontier),
            transitions,
            best,
            flush=True,
        )
    return None


def inspect(env):
    reach_level_9(env)
    picked = env.clone()
    pickup_prefix = direct_second_prefix() + COMBINED_DISMISS_PICK + [5]
    for action in pickup_prefix:
        picked.step(action)
    unique = {}
    for up, right in ((6, 4), (5, 5), (4, 6)):
        for state, path in positioned_states(picked, up, right):
            unique.setdefault(arr(state.frame()).tobytes(), (state, path))
    print("POSITION_BEAM_START", len(unique), flush=True)
    print("POSITION_BEAM_RESULT", search(list(unique.values())), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
