"""Accept alternate capture states only when the reproduced suffix still wins."""

import json
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves, solve_bridge_carrier_peg_solitaire
from perception import arr, connected_components, safe_step


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


def peg_count(frame):
    return sum(blob.size == (4, 4)
               for blob in connected_components(frame, colors=(14,)))


def macros(frame):
    out = [(action,) for action in (1, 2, 3, 4)]
    out.extend(
        ((6, source[1] + 1, source[0] + 1),
         (6, destination[1] + 1, destination[0] + 1))
        for _, source, destination in _bridge_carrier_moves(frame)
    )
    return tuple(out)


def path_units(path):
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


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "5"))
    max_states = int(os.environ.get("OPT_STATES", "300"))
    segment_choice = int(os.environ.get("OPT_SEGMENT", "1"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = None
    start = end = None
    for index, action in enumerate(campaign):
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            start = index + 1
        if prior < desired <= current:
            end = index + 1
            break
        prior = current
    level_path = campaign[start:end]

    node = entry.clone()
    segment_root = node.clone()
    segment_actions = []
    segments = []
    consumed = 0
    prior_pegs = peg_count(node.frame())
    for unit in path_units(level_path):
        for action in unit:
            safe_step(node, action)
        segment_actions.extend(unit)
        consumed += len(unit)
        after_pegs = peg_count(node.frame())
        if after_pegs != prior_pegs or int(node.levels_completed) >= desired:
            segments.append((segment_root, tuple(segment_actions), consumed))
            segment_root = node.clone()
            segment_actions = []
        prior_pegs = after_pegs
    root, original, consumed = segments[segment_choice]
    suffix = level_path[consumed:]
    bound = int(os.environ.get("OPT_COST", str(len(original))))
    root_pegs = peg_count(root.frame())
    serial = 0
    queue = [(0, serial, root.clone(), ())]
    best = {key(root): 0}
    captures_tested = 0
    expanded = 0
    while queue and len(best) <= max_states:
        cost, _, node, path = heappop(queue)
        if cost != best.get(key(node)) or cost >= bound:
            continue
        expanded += 1
        if expanded % 100 == 0:
            print("progress", expanded, len(best), cost, captures_tested,
                  flush=True)
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
            if peg_count(child.frame()) < root_pegs:
                captures_tested += 1
                if os.environ.get("OPT_SHOW_CAPTURES") == "1":
                    print("capture", captures_tested, child_cost,
                          child_path, flush=True)
                if (
                    os.environ.get("OPT_REPLAN") == "1"
                    and captures_tested <= int(os.environ.get(
                        "OPT_REPLAN_LIMIT", "1"
                    ))
                ):
                    recorder = Recorder(child.clone())
                    solve_bridge_carrier_peg_solitaire(
                        recorder,
                        max_align_states=int(os.environ.get(
                            "OPT_REPLAN_STATES", "650"
                        )),
                    )
                    print("replan", captures_tested, child_cost,
                          len(recorder.path), int(recorder.levels_completed),
                          tuple(recorder.path), flush=True)
                result = child.clone()
                for action in suffix:
                    safe_step(result, action)
                    if int(result.levels_completed) >= desired:
                        print("solution", segment_choice, len(original),
                              child_cost, captures_tested,
                              len(best), expanded, child_path, flush=True)
                        return
            if child_cost >= best.get(child_key, 10 ** 9):
                continue
            best[child_key] = child_cost
            serial += 1
            heappush(queue, (child_cost, serial, child, child_path))
    print("none", segment_choice, len(original), len(best), expanded,
          captures_tested, flush=True)


arena.run_program("lf52", probe)
