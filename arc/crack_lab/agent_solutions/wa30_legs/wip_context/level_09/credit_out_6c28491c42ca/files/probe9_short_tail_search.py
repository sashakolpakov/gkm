"""Bounded dismissal search from the four-action-shorter level-9 prefix."""

import gkm_try

from perception import arr
from probe9_prefix_shortcuts import direct_short_prefix, reach_level_9
from probe9_verify import boxes, target_state, tile_map


def avatar_cell(frame):
    avatar = boxes(frame, 14)
    if not avatar:
        return None
    row0, col0, row1, col1 = avatar[0]
    return ((row0 + row1) // 8, (col0 + col1) // 8)


def search_dismissal(env, max_depth=16, max_transitions=40000):
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
                target = target_state(child.frame())
                score = (len(target["filled"]), -len(target["empty"]))
                if best is None or score > best[0]:
                    best = (
                        score,
                        child_path,
                        avatar_cell(child.frame()),
                        boxes(child.frame(), 15),
                    )
                if not boxes(child.frame(), 15):
                    print(
                        "SHORT_DISMISS_WIN",
                        depth,
                        transitions,
                        child_path,
                        avatar_cell(child.frame()),
                        target,
                        flush=True,
                    )
                    return child_path
                if not child.terminal():
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, child_path),
                    )
                if transitions >= max_transitions:
                    print("SHORT_DISMISS_LIMIT", transitions, best, flush=True)
                    return None
        frontier = list(next_states.values())
        print(
            "SHORT_DISMISS_DEPTH",
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
    prefix = direct_short_prefix()
    for action in prefix:
        state.step(action)
    print(
        "SHORT_DISMISS_START",
        len(prefix),
        avatar_cell(state.frame()),
        boxes(state.frame(), 15),
        target_state(state.frame()),
        flush=True,
    )
    path = search_dismissal(state)
    print("SHORT_DISMISS_RESULT", path, flush=True)
    if path is not None:
        for action in path:
            state.step(action)
        print(
            "SHORT_DISMISS_STATE",
            {
                "avatar": boxes(state.frame(), 14),
                "cargo": boxes(state.frame(), 4),
                "courier": boxes(state.frame(), 12),
                "target": target_state(state.frame()),
            },
            *tile_map(state.frame()),
            sep="\n",
            flush=True,
        )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
