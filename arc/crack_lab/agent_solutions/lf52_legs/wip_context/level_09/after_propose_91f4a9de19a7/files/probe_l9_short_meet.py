"""Test a direct carrier meeting after the first remote bridge transfer."""

import json
import sys
from heapq import heappop, heappush

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import action_deltas, arr, connected_components, safe_step


FIRST_RELAY = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)

FAR = (
    ((18, 58), (18, 46)), ((18, 52), (18, 40)),
    ((18, 46), (18, 34)), ((18, 40), (18, 28)),
    ((18, 34), (18, 22)), ((18, 28), (18, 16)),
    ((18, 22), (18, 10)), ((18, 16), (18, 4)),
)


def move(env, source, destination):
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def compact(frame):
    return tuple(
        (blob.color, blob.top_left, blob.size, blob.area)
        for blob in connected_components(frame, colors=(9, 12, 14, 15))
        if blob.color not in (9, 15) or blob.area == 12
    )


def legal_moves(frame):
    blobs = connected_components(frame, colors=(1, 9, 12, 14, 15))
    holes = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    bridges = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4) and blob.area == 12
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    carriers = {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4) and blob.area == 12
    }
    cells = holes | bridges | pegs | carriers | fixed
    occupied = bridges | pegs | fixed
    out = []
    for kind, sources in (("B", bridges), ("P", pegs)):
        for source in sources:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (
                    midpoint in occupied
                    and destination in cells
                    and destination not in occupied
                ):
                    out.append((kind, source, destination))
    return tuple(sorted(out))


def search(root, max_cost=12, max_states=800):
    def state_key(node):
        return int(node.levels_completed), arr(node.frame())[1:, :].tobytes()

    base_level = int(root.levels_completed)
    serial = 0
    queue = [(0, serial, root.clone(), ())]
    best = {state_key(root): 0}
    while queue and len(best) <= max_states:
        cost, _, node, path = heappop(queue)
        if cost != best.get(state_key(node)) or cost >= max_cost:
            continue
        macros = [((action,), None) for action in (3, 4)]
        macros.extend((
            (
                (6, source[1] + 1, source[0] + 1),
                (6, destination[1] + 1, destination[0] + 1),
            ), (kind, source, destination)
        ) for kind, source, destination in legal_moves(node.frame()))
        for macro, _ in macros:
            child_cost = cost + len(macro)
            if child_cost > max_cost:
                continue
            child = node.clone()
            for action in macro:
                safe_step(child, action)
                if int(child.levels_completed) > base_level:
                    return path + macro, len(best)
            child_key = state_key(child)
            if child_cost >= best.get(child_key, 10 ** 9):
                continue
            best[child_key] = child_cost
            serial += 1
            heappush(queue, (child_cost, serial, child, path + macro))
    return None, len(best)


def probe(env):
    print("probe_start", int(env.levels_completed), flush=True)
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        env.step(action)
    print("probe_entry", int(env.levels_completed), flush=True)
    node = env.clone()
    for source, destination in FIRST_RELAY:
        move(node, source, destination)
    print("probe_relay", int(node.levels_completed), flush=True)
    for _ in range(9):
        safe_step(node, 4)
    for source, destination in FAR:
        move(node, source, destination)
    safe_step(node, 3)
    move(node, (18, 16), (18, 4))
    move(node, (12, 4), (24, 4))
    solution, explored = search(node)
    print("short_search", explored, solution, flush=True)

    direct = node.clone()
    safe_step(direct, 3)
    safe_step(direct, 3)
    print("direct_legal", legal_moves(direct.frame()), flush=True)
    move(direct, (18, 22), (30, 22))
    move(direct, (36, 22), (24, 22))
    move(direct, (24, 22), (24, 10))
    print("direct", int(direct.levels_completed), compact(direct.frame()),
          flush=True)

    for left in range(7):
        candidate = node.clone()
        for _ in range(left):
            safe_step(candidate, 3)
        deltas = action_deltas(candidate, (1, 2, 3, 4, 7))
        print("left", left, compact(candidate.frame()),
              tuple((action, delta["count"], delta["bbox"])
                    for action, delta in deltas.items()), flush=True)
        for vertical in (1, 2):
            child = candidate.clone()
            before = compact(child.frame())
            safe_step(child, vertical)
            after = compact(child.frame())
            if after != before:
                print("turn", left, vertical, after, flush=True)


arena.run_program("lf52", probe)
