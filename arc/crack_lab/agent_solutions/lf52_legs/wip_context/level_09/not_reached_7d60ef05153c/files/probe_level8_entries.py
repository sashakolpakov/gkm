"""Search every key-reachable level-8 carrier entry symbolically."""

from collections import deque
import json
import os

import gkm_try

from perception import arr, connected_components, safe_step


LEVEL_START = 476
STAGE = int(os.environ.get("STAGE", "0"))
STAGE_GROUPS = {0: 0, 1: 11, 2: 16, 3: 20}
MAX_KEY_STATES = int(os.environ.get("MAX_KEY_STATES", "500"))
MAX_KEY_DEPTH = int(os.environ.get("MAX_KEY_DEPTH", "16"))


def geometry(frame):
    blobs = connected_components(frame, colors=(1, 9, 11, 12, 14, 15))
    holes = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4)
    }
    carriers = {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    } | {
        (blob.bbox[0] + 1, blob.bbox[1] + 1) for blob in blobs
        if blob.color == 11 and blob.area >= 4
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    bridges = {
        blob.top_left for blob in blobs
        if blob.color == 9 and blob.size == (4, 4)
    }
    fixed = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4)
    }
    return holes, carriers, pegs, bridges, fixed


def entry_solution(frame, final=False, max_states=100000):
    holes, carriers, root_pegs, root_bridges, fixed = geometry(frame)
    slots = frozenset(holes | carriers | root_pegs | root_bridges)
    start = frozenset(root_pegs), frozenset(root_bridges)
    starting_loaded = frozenset(root_pegs & carriers)
    queue = deque([(start, ())])
    seen = {start}
    while queue and len(seen) <= max_states:
        (pegs, bridges), path = queue.popleft()
        loaded = (pegs & carriers) - starting_loaded
        if (final and len(pegs) == 1) or (not final and loaded):
            return path, tuple(sorted(loaded)), len(seen)
        occupied = pegs | bridges
        sources = tuple(("P", point) for point in sorted(pegs))
        sources += tuple(("B", point) for point in sorted(bridges))
        for kind, source in sources:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = source[0] + dr, source[1] + dc
                destination = source[0] + 2 * dr, source[1] + 2 * dc
                if (
                    midpoint not in occupied | fixed
                    or destination not in slots
                    or destination in occupied
                ):
                    continue
                child_pegs, child_bridges = set(pegs), set(bridges)
                if kind == "P":
                    child_pegs.remove(source); child_pegs.add(destination)
                    child_pegs.discard(midpoint)
                else:
                    child_bridges.remove(source); child_bridges.add(destination)
                child = frozenset(child_pegs), frozenset(child_bridges)
                if child in seen:
                    continue
                seen.add(child)
                queue.append((child, path + ((kind, source, destination),)))
    return None, (), len(seen)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:LEVEL_START]:
        safe_step(env, action)

    segment = prefix[LEVEL_START:544]
    index = group = 0
    while index < len(segment) and group < STAGE_GROUPS[STAGE]:
        while index < len(segment) and not isinstance(segment[index], list):
            safe_step(env, segment[index]); index += 1
        count = 0
        while index < len(segment) and isinstance(segment[index], list) and count < 2:
            safe_step(env, segment[index]); index += 1; count += 1
        group += 1

    queue = deque([(env.clone(), ())])
    seen = {arr(env.frame())[1:, :].tobytes()}
    results = []
    while queue and len(seen) <= MAX_KEY_STATES:
        node, keys = queue.popleft()
        solution, loaded, searched = entry_solution(node.frame(), final=STAGE == 3)
        if solution is not None:
            results.append((len(keys) + 2 * len(solution), keys, solution, loaded, searched))
        if len(keys) >= MAX_KEY_DEPTH:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone(); safe_step(child, action)
            key = arr(child.frame())[1:, :].tobytes()
            if key in seen:
                continue
            seen.add(key); queue.append((child, keys + (action,)))

    results.sort(key=lambda item: (item[0], len(item[1]), item[1]))
    print("L8_ENTRY_SEARCH", STAGE, len(seen), len(results))
    for result in results[:12]:
        print("L8_ENTRY", result)
    for rank, (cost, keys, moves, loaded, _) in enumerate(results[:12]):
        replay = env.clone()
        for action in keys:
            safe_step(replay, action)
        for _, source, destination in moves:
            safe_step(replay, (6, source[1] + 1, source[0] + 1))
            safe_step(replay, (6, destination[1] + 1, destination[0] + 1))
        actual = geometry(replay.frame())
        valid = (
            int(replay.levels_completed) >= 8
            if STAGE == 3 else bool(set(loaded) & actual[2])
        )
        print(
            "L8_ENTRY_VERIFY", STAGE, rank, cost, valid,
            int(replay.levels_completed), actual[1:4],
        )


gkm_try.A.run_program("lf52", probe)
