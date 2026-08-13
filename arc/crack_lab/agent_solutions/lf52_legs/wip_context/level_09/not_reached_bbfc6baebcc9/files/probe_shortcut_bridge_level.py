"""Search shorter exact milestone paths in bridge/carrier levels."""

import json
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import (
    _bridge_carrier_moves,
    solve_compact_bridge_carrier_peg_solitaire,
)
from perception import arr, connected_components, frame_delta, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


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


def key(env):
    return int(env.levels_completed), arr(env.frame())[1:, :].tobytes()


def pegs(frame):
    return sum(
        blob.color == 14 and blob.size == (4, 4)
        for blob in connected_components(frame, colors=(14,))
    )


def units(path):
    out = []
    index = 0
    while index < len(path):
        if isinstance(path[index], int):
            out.append((path[index],))
            index += 1
        else:
            out.append((path[index], path[index + 1]))
            index += 2
    return tuple(out)


def macros(frame):
    out = [(action,) for action in (1, 2, 3, 4)]
    out.extend(
        ((6, source[1] + 1, source[0] + 1),
         (6, destination[1] + 1, destination[0] + 1))
        for _, source, destination in _bridge_carrier_moves(frame)
    )
    return tuple(out)


def reversible_macros(frame):
    inverse = {1: 2, 2: 1, 3: 4, 4: 3}
    out = [((action,), (inverse[action],), "key")
           for action in (1, 2, 3, 4)]
    for kind, source, destination in _bridge_carrier_moves(frame):
        forward = (
            (6, source[1] + 1, source[0] + 1),
            (6, destination[1] + 1, destination[0] + 1),
        )
        reverse = (
            (6, destination[1] + 1, destination[0] + 1),
            (6, source[1] + 1, source[0] + 1),
        )
        out.append((forward, reverse, kind))
    return tuple(out)


def shortest(root, target, bound, max_states):
    serial = 0
    queue = [(0, serial, root.clone(), ())]
    best = {key(root): 0}
    expanded = 0
    while queue and len(best) <= max_states:
        cost, _, node, path = heappop(queue)
        if cost != best.get(key(node)) or cost >= bound:
            continue
        expanded += 1
        if expanded % 100 == 0:
            print("search_progress", expanded, len(best), cost, flush=True)
        for macro in macros(node.frame()):
            child_cost = cost + len(macro)
            if child_cost >= bound:
                continue
            child = node.clone()
            before = key(child)
            for action in macro:
                safe_step(child, action)
            child_key = key(child)
            if child_key == before:
                continue
            child_path = path + macro
            if child_key == target:
                return child_path, len(best), expanded
            if child_cost >= best.get(child_key, 10 ** 9):
                continue
            best[child_key] = child_cost
            serial += 1
            heappush(queue, (child_cost, serial, child, child_path))
    return None, len(best), expanded


def shortest_replay(root, target, bound, max_states):
    def reconstruct(path):
        node = root.clone()
        for action in path:
            safe_step(node, action)
        return node

    serial = 0
    queue = [(0, serial, ())]
    best = {key(root): 0}
    expanded = 0
    while queue and len(best) <= max_states:
        cost, _, path = heappop(queue)
        node = reconstruct(path)
        if cost != best.get(key(node)) or cost >= bound:
            continue
        expanded += 1
        if expanded % 100 == 0:
            print("replay_progress", expanded, len(best), cost, flush=True)
        for macro in macros(node.frame()):
            child_cost = cost + len(macro)
            if child_cost >= bound:
                continue
            child_path = path + macro
            child = reconstruct(child_path)
            child_key = key(child)
            if child_key == key(node):
                continue
            if child_key == target:
                return child_path, len(best), expanded
            if child_cost >= best.get(child_key, 10 ** 9):
                continue
            best[child_key] = child_cost
            serial += 1
            heappush(queue, (child_cost, serial, child_path))
    return None, len(best), expanded


def shortest_undo(root, target, bound, max_states):
    node = root.clone()
    expanded_total = 0

    def search(remaining, path, seen):
        nonlocal expanded_total
        if expanded_total >= max_states:
            return None
        node_key = key(node)
        if node_key == target:
            return path
        known = seen.get(node_key)
        if known is not None and known >= remaining:
            return None
        seen[node_key] = remaining
        expanded_total += 1
        if expanded_total % 500 == 0:
            print("undo_progress", expanded_total, len(path), remaining,
                  flush=True)
        ordered = list(macros(node.frame()))
        ordered.sort(key=lambda macro: (len(macro), macro))
        for macro in ordered:
            macro_cost = len(macro)
            if macro_cost > remaining:
                continue
            before = key(node)
            for action in macro:
                safe_step(node, action)
            after = key(node)
            result = None
            if after != before:
                result = search(remaining - macro_cost, path + macro, seen)
            for _ in macro:
                safe_step(node, 7)
            if key(node) != before:
                raise AssertionError(("undo mismatch", macro))
            if result is not None:
                return result
        return None

    minimum = int(os.environ.get("OPT_MIN_COST", "0"))
    for limit in range(minimum, bound):
        print("undo_limit", limit, flush=True)
        try:
            result = search(limit, (), {})
        except Exception as exc:
            print("undo_error", repr(exc), flush=True)
            raise
        if result is not None:
            return result, expanded_total, expanded_total
        if expanded_total >= max_states:
            break
    return None, expanded_total, expanded_total


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "4"))
    segment_choice = int(os.environ.get("OPT_SEGMENT", "0"))
    max_states = int(os.environ.get("OPT_STATES", "1200"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    start = end = None
    entry = None
    for index, action in enumerate(campaign):
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            start = index + 1
            entry = env.clone()
        if prior < desired <= current:
            end = index + 1
            break
        prior = current

    if os.environ.get("OPT_CURRENT") == "1":
        if desired != 4:
            raise ValueError("current-route mode currently supports level 4")
        recorder = Recorder(entry.clone())
        solve_compact_bridge_carrier_peg_solitaire(recorder)
        level_path = tuple(recorder.path)
    else:
        level_path = campaign[start:end]
    level_units = units(level_path)
    node = entry.clone()
    segment_start = node.clone()
    segment_actions = []
    segments = []
    prior_pegs = pegs(node.frame())
    for unit in level_units:
        before = arr(node.frame()).copy()
        for action in unit:
            safe_step(node, action)
        after_pegs = pegs(node.frame())
        delta = frame_delta(before, node.frame())["count"]
        segment_actions.extend(unit)
        milestone = (
            after_pegs != prior_pegs
            or int(node.levels_completed) >= desired
        )
        if milestone:
            segments.append((segment_start, tuple(segment_actions),
                             key(node), after_pegs, delta))
            segment_start = node.clone()
            segment_actions = []
        prior_pegs = after_pegs
    if segment_actions:
        segments.append((segment_start, tuple(segment_actions),
                         key(node), pegs(node.frame()), 0))
    print("segments", tuple((len(path), peg_count, delta)
                            for _, path, _, peg_count, delta in segments),
          flush=True)
    if os.environ.get("OPT_SHOW_ORIGINAL") == "1":
        print("originals", tuple(path for _, path, _, _, _ in segments),
              flush=True)
    root, original, target, _, _ = segments[segment_choice]
    search_fn = (
        shortest_undo if os.environ.get("OPT_UNDO") == "1"
        else shortest_replay if os.environ.get("OPT_REPLAY") == "1"
        else shortest
    )
    solution, states, expanded = search_fn(root, target, len(original), max_states)
    print("shortcut", segment_choice, len(original),
          None if solution is None else len(solution), states, expanded,
          solution, flush=True)


arena.run_program("lf52", probe)
