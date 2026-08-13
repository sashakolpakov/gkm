"""Optimize carrier keys while preserving a verified coordinate-move route."""

import json
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    _bridge_carrier_state,
    solve_bridge_carrier_peg_solitaire,
    solve_compact_bridge_carrier_peg_solitaire,
    solve_grid_wrapped_bridge_carrier_peg_solitaire,
    solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    solve_repeated_frontier_bridge_carrier_peg_solitaire,
    solve_wrapped_bridge_carrier_peg_solitaire,
)
from perception import arr, connected_components, safe_step


def solve_level_five(env):
    solve_bridge_carrier_peg_solitaire(env, max_align_states=650)


SOLVERS = {
    4: solve_compact_bridge_carrier_peg_solitaire,
    5: solve_level_five,
    6: solve_wrapped_bridge_carrier_peg_solitaire,
    7: solve_parallel_wrapped_bridge_carrier_peg_solitaire,
    8: solve_grid_wrapped_bridge_carrier_peg_solitaire,
    9: solve_repeated_frontier_bridge_carrier_peg_solitaire,
}


class Recorder:
    def __init__(self, inner):
        self.inner = inner
        self.path = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def clone(self):
        return self.inner.clone()

    def step(self, action, *coordinates):
        recorded = ((action,) + coordinates) if coordinates else action
        self.path.append(recorded)
        return self.inner.step(action, *coordinates)


def split(path):
    groups = []
    keys = []
    index = 0
    while index < len(path):
        if isinstance(path[index], int):
            keys.append(path[index])
            index += 1
        else:
            groups.append((tuple(keys), (path[index], path[index + 1])))
            keys = []
            index += 2
    if keys:
        raise ValueError("verified route has trailing keys")
    return tuple(groups)


def frame_key(env):
    return arr(env.frame())[1:, :].tobytes()


def piece_key(frame):
    state = _bridge_carrier_state(frame)
    movable = tuple(sorted(
        (blob.color, blob.top_left)
        for blob in connected_components(frame, colors=(8, 9))
        if blob.size == (4, 4)
    ))
    return state[:4], movable, state[5]


def local_pattern(frame, clicks):
    image = arr(frame)
    first, second = clicks
    source = (first[2] - 1, first[1] - 1)
    destination = (second[2] - 1, second[1] - 1)
    midpoint = (
        (source[0] + destination[0]) // 2,
        (source[1] + destination[1]) // 2,
    )
    return tuple(
        image[row:row + 4, col:col + 4].tobytes()
        for row, col in (source, midpoint, destination)
    )


def optimize(entry, groups, patterns, baseline, max_states, include_reset):
    base_level = int(entry.levels_completed)
    actions = (1, 2, 3, 4, 7) if include_reset else (1, 2, 3, 4)
    serial = 0
    start_key = (0, frame_key(entry))
    best = {start_key: 0}
    queue = [(2 * len(groups), 0, serial, 0, entry.clone(), ())]
    expanded = 0
    stage_counts = {}
    while queue and expanded < max_states:
        _, cost, _, stage, node, path = heappop(queue)
        node_key = (stage, frame_key(node))
        if cost != best.get(node_key):
            continue
        if cost + 2 * (len(groups) - stage) >= baseline:
            continue
        expanded += 1
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if expanded % 100 == 0:
            print("fixed_progress", expanded, len(best), cost, stage,
                  tuple(sorted(stage_counts.items())), flush=True)

        clicks = groups[stage][1]
        if local_pattern(node.frame(), clicks) == patterns[stage]:
            child = node.clone()
            before_piece = piece_key(child.frame())
            for action in clicks:
                safe_step(child, action)
            after_piece = piece_key(child.frame())
            advanced = (
                int(child.levels_completed) > base_level
                or (after_piece != before_piece and after_piece[-1] is None)
            )
        else:
            advanced = False
        if advanced:
            child_cost = cost + 2
            child_path = path + clicks
            if int(child.levels_completed) > base_level:
                return child_path, expanded, stage_counts
            child_stage = stage + 1
            if child_stage == len(groups):
                return child_path, expanded, stage_counts
            child_key = (child_stage, frame_key(child))
            if child_cost < best.get(child_key, baseline):
                best[child_key] = child_cost
                serial += 1
                priority = child_cost + 2 * (len(groups) - child_stage)
                heappush(queue, (
                    priority, child_cost, serial, child_stage,
                    child, child_path,
                ))

        for action in actions:
            child_cost = cost + 1
            if child_cost + 2 * (len(groups) - stage) >= baseline:
                continue
            child = node.clone()
            safe_step(child, action)
            child_frame_key = frame_key(child)
            if child_frame_key == node_key[1]:
                continue
            child_key = (stage, child_frame_key)
            if child_cost >= best.get(child_key, baseline):
                continue
            best[child_key] = child_cost
            serial += 1
            priority = child_cost + 2 * (len(groups) - stage)
            heappush(queue, (
                priority, child_cost, serial, stage,
                child, path + (action,),
            ))
    return None, expanded, stage_counts


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "7"))
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    prior = int(env.levels_completed)
    entry = None
    entry_index = -1
    for index, action in enumerate(campaign):
        safe_step(env, tuple(action) if isinstance(action, list) else action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            entry_index = index
            break
        prior = current

    if desired == 5:
        route_node = entry.clone()
        route = []
        for action in campaign[entry_index + 1:]:
            normalized = tuple(action) if isinstance(action, list) else action
            safe_step(route_node, normalized)
            route.append(normalized)
            if int(route_node.levels_completed) >= desired:
                break
        recorded_path = tuple(route)
    else:
        recorder = Recorder(entry.clone())
        SOLVERS[desired](recorder)
        recorded_path = tuple(recorder.path)
    groups = split(recorded_path)
    start_stage = int(os.environ.get("OPT_START_STAGE", "0"))
    end_stage = min(
        len(groups), int(os.environ.get("OPT_END_STAGE", str(len(groups))))
    )
    root = entry.clone()
    prefix = []
    for keys, clicks in groups[:start_stage]:
        for action in keys + clicks:
            safe_step(root, action)
            prefix.append(action)
    suffix_groups = groups[start_stage:end_stage]
    suffix_baseline = sum(len(keys) + len(clicks)
                          for keys, clicks in suffix_groups)
    print("fixed_setup", desired, start_stage, end_stage,
          len(recorded_path), len(suffix_groups), suffix_baseline,
          tuple(len(keys) for keys, _ in suffix_groups), flush=True)
    reference = root.clone()
    patterns = []
    for keys, clicks in suffix_groups:
        for action in keys:
            safe_step(reference, action)
        patterns.append(local_pattern(reference.frame(), clicks))
        for action in clicks:
            safe_step(reference, action)
    solution, expanded, stage_counts = optimize(
        root,
        suffix_groups,
        tuple(patterns),
        suffix_baseline,
        int(os.environ.get("OPT_STATES", "5000")),
        os.environ.get("OPT_RESET") == "1",
    )
    full_solution = None if solution is None else tuple(prefix) + solution
    print("fixed_result", desired, start_stage, end_stage, len(recorded_path),
          None if full_solution is None else len(full_solution), expanded,
          tuple(sorted(stage_counts.items())), solution, flush=True)


levels, path, error = arena.run_program("lf52", probe)
if error:
    print("fixed_worker_error", repr(error), flush=True)
