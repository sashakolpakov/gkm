"""Beam-decode support coordinates together with the one omitted support."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from legs import COL_ANCHORS, ROW_ANCHORS
from perception import arr
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, controls


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


LEFT = (3,)
BEAM_UNUSED = int(os.environ.get("BEAM_UNUSED", "128"))
BEAM_USED = int(os.environ.get("BEAM_USED", "96"))
PER_INSERTION = int(os.environ.get("PER_INSERTION", "8"))
SHARD = int(os.environ.get("SHARD", "0"))
SHARDS = int(os.environ.get("SHARDS", "1"))
ROUTE = decoded_route()
START_BOUNDARY = int(os.environ.get("START_BOUNDARY", "0"))
END_BOUNDARY = int(os.environ.get("END_BOUNDARY", str(len(ROUTE))))
AMBIGUOUS_STEPS = {19, 21, 22, 23, 41}


@dataclass
class State:
    node: object
    path: tuple
    height: int
    inserted_at: int | None
    deviations: int


def lattice(frame):
    return tuple(
        tuple(int(frame[y][x]) for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def fast_shift_grids(a, b):
    scored = []
    for bands in range(10):
        hits = sum(
            a[i][j] == b[i + bands][j]
            for i in range(10 - bands)
            for j in range(8)
        )
        scored.append((hits, -bands, bands))
    return max(scored)[2]


def click_candidates(frame, colors=(12, 14)):
    return [
        (6, x, y)
        for y in ROW_ANCHORS
        for x in COL_ANCHORS
        if int(frame[y][x]) in colors
    ]


def route_actions(frame, action, step):
    if (
        step not in AMBIGUOUS_STEPS
        or len(action) != 3
        or action[0] != 6
        or action[1] <= 5
    ):
        return [action]
    y = action[2]
    row = min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - y))
    candidates = [action]
    for x in COL_ANCHORS:
        if int(frame[ROW_ANCHORS[row]][x]) in (12, 14, 15):
            candidates.append((6, x, y))
    return list(dict.fromkeys(candidates))


def step_state(state, action, inserted_at=None, deviation=0):
    child = state.node.clone()
    before = lattice(child.frame())
    child.step(*action)
    if child.terminal():
        return None
    return State(
        child,
        state.path + (action,),
        state.height + fast_shift_grids(before, lattice(child.frame())),
        state.inserted_at if inserted_at is None else inserted_at,
        state.deviations + deviation,
    )


def rank(state):
    frame = state.node.frame()
    avatar = avatar_cell(frame) or (99, 99)
    expanded_hint = sum(
        int(frame[y][x]) in (12, 14)
        for y in ROW_ANCHORS
        for x in COL_ANCHORS
    )
    return (
        state.height,
        expanded_hint,
        len(controls(frame)),
        -avatar[1],
        -avatar[0],
    )


def prune(states, used):
    states.sort(key=rank, reverse=True)
    if not used:
        return states[:BEAM_UNUSED]
    groups = {}
    for state in states:
        groups.setdefault(state.inserted_at, []).append(state)
    kept = []
    for boundary_states in groups.values():
        baseline = next(
            (state for state in boundary_states if state.deviations == 0),
            None,
        )
        selected = [] if baseline is None else [baseline]
        selected.extend(
            state
            for state in boundary_states
            if state is not baseline
        )
        kept.extend(selected[:PER_INSERTION])
    kept.sort(key=rank, reverse=True)
    if len(kept) > BEAM_USED:
        baseline_states = [
            state for state in kept if state.deviations == 0
        ]
        others = [
            state for state in kept if state.deviations != 0
        ]
        kept = baseline_states + others[:max(
            0, BEAM_USED - len(baseline_states)
        )]
    return kept


def flatten(root, states):
    out = []
    for state in states:
        node = root.clone()
        for action in state.path:
            node.step(*action)
            if node.terminal():
                break
        if not node.terminal():
            out.append(
                State(
                    node,
                    state.path,
                    state.height,
                    state.inserted_at,
                    state.deviations,
                )
            )
    return out


def check_win(state, label):
    if state.node.levels_completed <= 6:
        return False
    print(
        "COUPLED_WIN", label, len(state.path), list(state.path),
        flush=True,
    )
    return True


def insert_supports(states, boundary):
    if (
        not START_BOUNDARY <= boundary <= END_BOUNDARY
        or boundary % SHARDS != SHARD
    ):
        return []
    out = []
    for state in states:
        for action in click_candidates(state.node.frame()):
            child = step_state(state, action, boundary)
            if child is None:
                continue
            if check_win(child, ("insert", boundary, action)):
                return None
            out.append(child)
    return out


def advance_route(states, action, step):
    out = []
    for state in states:
        candidates = route_actions(state.node.frame(), action, step)
        if len(candidates) == 1:
            candidate = candidates[0]
            before = lattice(state.node.frame())
            state.node.step(*candidate)
            if state.node.terminal():
                continue
            state.path += (candidate,)
            state.height += fast_shift_grids(
                before, lattice(state.node.frame())
            )
            state.deviations += int(candidate != action)
            if check_win(state, ("route", step, candidate)):
                return None
            out.append(state)
            continue
        for candidate in candidates:
            child = step_state(
                state,
                candidate,
                deviation=int(candidate != action),
            )
            if child is None:
                continue
            if check_win(child, ("route", step, candidate)):
                return None
            out.append(child)
    return out


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)

    unused = [State(env.clone(), (), 0, None, 0)]
    used = []
    for boundary, action in enumerate(ROUTE):
        if boundary == START_BOUNDARY and boundary:
            unused = flatten(env, unused)
        if boundary > END_BOUNDARY:
            unused = []
        inserted = insert_supports(unused, boundary)
        if inserted is None:
            return
        used.extend(inserted)
        used = prune(used, used=True)
        next_unused = advance_route(unused, action, boundary + 1)
        next_used = advance_route(used, action, boundary + 1)
        if next_unused is None or next_used is None:
            return
        unused = prune(next_unused, used=False)
        used = prune(next_used, used=True)
        if boundary == END_BOUNDARY:
            used = flatten(env, used)
        if (
            boundary < 4
            or len(action) == 3 and action[1] > 5
            or (boundary + 1) % 10 == 0
        ):
            print(
                "COUPLED_STAGE", boundary + 1, len(unused), len(used),
                max((state.height for state in unused), default=-1),
                max((state.height for state in used), default=-1),
                flush=True,
            )

    inserted = insert_supports(unused, len(ROUTE))
    if inserted is None:
        return
    used.extend(inserted)
    used = prune(used, used=True)

    for suffix_step in range(4):
        next_used = advance_route(used, LEFT, len(ROUTE) + suffix_step + 1)
        if next_used is None:
            return
        used = prune(next_used, used=True)
        print(
            "COUPLED_SUFFIX", suffix_step + 1, len(used),
            max((state.height for state in used), default=-1),
            flush=True,
        )

    for state in used:
        visible = controls(state.node.frame())
        if not visible:
            continue
        final = (6, 3, max(visible))
        child = step_state(state, final)
        if child is not None and check_win(child, ("final", final)):
            return

    print("COUPLED_NONE", len(used), flush=True)
    for state in sorted(used, key=rank, reverse=True)[:30]:
        print(
            "COUPLED_OPTION", rank(state), state.inserted_at,
            avatar_cell(state.node.frame()), tuple(controls(state.node.frame())),
            [
                (index + 1, action)
                for index, action in enumerate(state.path)
                if len(action) == 3 and action[1] > 5
            ],
            flush=True,
        )


if __name__ == "__main__":
    levels, path, error = arena.run_program("bp35", probe)
    print(
        "COUPLED_RESULT", SHARD, levels, len(path), repr(error),
        flush=True,
    )
