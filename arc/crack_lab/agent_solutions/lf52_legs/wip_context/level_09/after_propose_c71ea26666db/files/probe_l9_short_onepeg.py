"""Try to align the carrier after a short, non-carried one-peg opening."""

import json
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step
from probe_key_neighborhood_events import generic_moves


SHORT_OPENING = (
    ((48, 30), (36, 30)),
    ((48, 42), (48, 30)),
    ((48, 30), (48, 18)),
    ((48, 18), (36, 18)),
)


def parse(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14))
    cells = sorted({
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.color in (1, 9, 12, 14)
    })
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    bridges = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    carrier = next(
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    )
    return cells, pegs, bridges, carrier


def move(node, source, destination):
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def key(node):
    return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()


def state(node):
    cells, pegs, bridges, carrier = parse(node.frame())
    return (tuple(sorted(pegs)), tuple(sorted(bridges)), carrier, len(cells))


def goal(node):
    try:
        _, pegs, _, carrier = parse(node.frame())
    except StopIteration:
        return False
    return pegs == {carrier}


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    safe_step(env, 7)
    for source, destination in SHORT_OPENING:
        move(env, source, destination)
    print("short_root", state(env), int(env.levels_completed), flush=True)

    bound = int(os.environ.get("OPT_COST", "19"))
    max_states = int(os.environ.get("OPT_STATES", "5000"))
    serial = 0
    queue = [(0, serial, env.clone(), ())]
    best = {key(env): 0}
    expanded = 0
    while queue and len(best) <= max_states:
        cost, _, node, path = heappop(queue)
        if cost != best.get(key(node)) or cost >= bound:
            continue
        expanded += 1
        macros = tuple((action,) for action in (1, 2, 3, 4))
        macros += tuple(
            ((6, source[1] + 1, source[0] + 1),
             (6, destination[1] + 1, destination[0] + 1))
            for _, source, destination in generic_moves(node.frame())
        )
        for macro in macros:
            child_cost = cost + len(macro)
            if child_cost > bound:
                continue
            child = node.clone()
            for action in macro:
                safe_step(child, action)
            child_key = key(child)
            if child_key == key(node):
                continue
            child_path = path + macro
            if goal(child):
                print("short_goal", 9 + child_cost, child_cost,
                      len(best), expanded, child_path, state(child), flush=True)
                return
            if child_cost >= best.get(child_key, 10 ** 9):
                continue
            best[child_key] = child_cost
            serial += 1
            heappush(queue, (child_cost, serial, child, child_path))
        if expanded % 100 == 0:
            print("short_progress", expanded, len(best), cost, flush=True)
    print("short_none", bound, len(best), expanded, flush=True)


def checked(env):
    try:
        probe(env)
    except Exception as error:
        print("short_error", repr(error), flush=True)
        raise


arena.run_program("lf52", checked)
