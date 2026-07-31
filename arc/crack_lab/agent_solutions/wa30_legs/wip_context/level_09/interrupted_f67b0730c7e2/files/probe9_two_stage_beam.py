"""Search the shortened hand finish after the actual two-staged handoff."""

import gkm_try

from perception import arr
from probe9_two_staged_trace import DISMISS, TWO_STAGED
from probe9_verify import boxes, target_state


SHORT_POSITION = (
    [3] * 2 + [5, 4]
    + [1] * 6 + [4] * 4
)


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


def search(env, max_depth=11, beam_width=400, max_transitions=25000):
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
                        "TWO_STAGE_BEAM_WIN",
                        depth,
                        transitions,
                        child_path,
                        flush=True,
                    )
                    return child_path
                child_score = score(child)
                if best is None or child_score > best[0]:
                    best = (
                        child_score,
                        child_path,
                        boxes(child.frame(), 14),
                        boxes(child.frame(), 12),
                        target_state(child.frame()),
                    )
                if not child.terminal():
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child_score, child, child_path),
                    )
                if transitions >= max_transitions:
                    print("TWO_STAGE_BEAM_LIMIT", transitions, best, flush=True)
                    return None
        ranked = sorted(
            next_states.values(),
            key=lambda item: item[0],
            reverse=True,
        )[:beam_width]
        frontier = [(child, path) for _, child, path in ranked]
        print(
            "TWO_STAGE_BEAM_DEPTH",
            depth,
            len(frontier),
            transitions,
            best,
            flush=True,
        )
    return None


def inspect(env):
    gkm_try.resumed_solve(env)
    state = env.clone()
    prefix = TWO_STAGED + DISMISS + SHORT_POSITION
    for action in prefix:
        state.step(action)
    print(
        "TWO_STAGE_BEAM_START",
        13 + len(prefix),
        boxes(state.frame(), 14),
        boxes(state.frame(), 12),
        target_state(state.frame()),
        flush=True,
    )
    print("TWO_STAGE_BEAM_RESULT", search(state), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
