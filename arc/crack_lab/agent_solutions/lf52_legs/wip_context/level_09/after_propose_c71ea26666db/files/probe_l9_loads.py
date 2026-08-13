"""Enumerate symbolic carrier-loading transitions from pristine level 9."""

import json
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


def parse(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14))
    holes = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    bridges = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    carriers = {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    return frozenset(holes | pegs | bridges), frozenset(pegs), frozenset(bridges), next(iter(carriers))


def compact(frame):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(7, 9, 11, 12, 14, 15))
        if blob.color not in (9, 12) or blob.area >= 12
    )


def loading_paths(frame, max_cost=34, max_states=100000):
    static_cells, pegs, bridges, carrier = parse(frame)
    start = (pegs, bridges, carrier)
    serial = 0
    queue = [(0, serial, start, ())]
    best = {start: 0}
    loads = {}
    while queue and len(best) <= max_states:
        cost, _, state, path = heappop(queue)
        if cost != best.get(state) or cost >= max_cost:
            continue
        state_pegs, state_bridges, state_carrier = state
        for action, dc in ((3, -6), (4, 6)):
            destination = (state_carrier[0], state_carrier[1] + dc)
            if not (42 <= destination[1] <= 60):
                continue
            child = (state_pegs, state_bridges, destination)
            child_cost = cost + 1
            if child_cost < best.get(child, 10 ** 9):
                best[child] = child_cost
                serial += 1
                heappush(queue, (child_cost, serial, child, path + (action,)))

        occupied = state_pegs | state_bridges
        destinations = static_cells | {state_carrier}
        for kind, source in (
            tuple(("peg", cell) for cell in sorted(state_pegs))
            + tuple(("bridge", cell) for cell in sorted(state_bridges))
        ):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint not in occupied
                    or destination not in destinations
                    or destination in occupied
                ):
                    continue
                clicks = (
                    (6, source[1] + 1, source[0] + 1),
                    (6, destination[1] + 1, destination[0] + 1),
                )
                child_cost = cost + 2
                child_pegs = set(state_pegs)
                child_bridges = set(state_bridges)
                if kind == "peg":
                    child_pegs.remove(source)
                    child_pegs.add(destination)
                    if midpoint in child_pegs:
                        child_pegs.remove(midpoint)
                else:
                    child_bridges.remove(source)
                    child_bridges.add(destination)
                child = (
                    frozenset(child_pegs), frozenset(child_bridges),
                    state_carrier,
                )
                child_path = path + clicks
                if destination == state_carrier:
                    load_key = (kind, child[0], child[1], state_carrier)
                    old = loads.get(load_key)
                    if old is None or child_cost < old[0]:
                        loads[load_key] = (child_cost, child_path)
                    continue
                if child_cost < best.get(child, 10 ** 9):
                    best[child] = child_cost
                    serial += 1
                    heappush(queue, (child_cost, serial, child, child_path))
    ordered = sorted(
        ((cost, key[0], key[1:], path) for key, (cost, path) in loads.items()),
        key=lambda item: (item[0], item[1], item[2]),
    )
    return tuple(ordered), len(best)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    base_frame = arr(env.frame()).copy()
    print("entry_model", parse(env.frame()), flush=True)
    loads, states = loading_paths(
        env.frame(), max_cost=int(os.environ.get("L9_LOAD_COST", "34"))
    )
    print("load_models", states, len(loads),
          [(x[0], x[1], len(x[2][0]), tuple(sorted(x[2][0])),
            tuple(sorted(x[2][1])), x[2][2]) for x in loads[:200]],
          flush=True)
    if os.environ.get("L9_WORLD_LOADS") == "1":
        from probe_l9_world_model import parse as parse_world

        batch_start = int(os.environ.get("OPT_BATCH_START", "0"))
        batch_size = int(os.environ.get("OPT_BATCH_SIZE", "10"))
        print("world_load_batch", batch_start, batch_size, len(loads),
              flush=True)
        for index, (cost, kind, model, path) in enumerate(
                loads[batch_start:batch_start + batch_size], batch_start):
            opened = env.clone()
            for action in path:
                safe_step(opened, action)
            scan = opened.clone()
            world_bridges = set()
            world_pegs = None
            for scan_index in range(15):
                offset = scan_index * 6
                _, visible_bridges, visible_pegs, _ = parse_world(
                    scan.frame(), offset
                )
                world_bridges |= visible_bridges
                if scan_index == 0:
                    world_pegs = visible_pegs
                if scan_index < 14:
                    safe_step(scan, 4)
            print("world_load", index, cost, kind,
                  tuple(sorted(world_pegs)), tuple(sorted(world_bridges)),
                  (() if os.environ.get("OPT_COMPACT") == "1"
                   else tuple(sorted(model[1]))),
                  (() if os.environ.get("OPT_COMPACT") == "1" else path),
                  flush=True)
        return
    replay_limit = int(os.environ.get("L9_LOAD_REPLAY", "40"))
    for index, (cost, kind, model, path) in enumerate(loads[:replay_limit]):
        clone = env.clone()
        for action in path:
            safe_step(clone, action)
        changed = int((arr(clone.frame())[1:] != base_frame[1:]).sum())
        print("load_replay", index, cost, kind, changed,
              int(clone.levels_completed), compact(clone.frame()), path,
              flush=True)


if __name__ == "__main__":
    arena.run_program("lf52", probe)
