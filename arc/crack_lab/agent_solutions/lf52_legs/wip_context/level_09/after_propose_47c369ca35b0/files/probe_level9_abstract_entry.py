"""Solve and clone-verify the empirically derived level-9 entry board."""

from collections import deque
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import frame_delta


def extract(frame):
    array = np.asarray(frame)
    destinations = set()
    bridges = set()
    pegs = set()
    carriers = set()
    for row in range(0, 61, 6):
        for col in range(0, 61, 6):
            patch = array[row:row + 4, col:col + 4]
            if int(np.isin(patch, (1, 9, 12, 14)).sum()) < 12:
                continue
            position = (row, col)
            destinations.add(position)
            if int(np.count_nonzero(patch == 14)) >= 12:
                pegs.add(position)
            elif int(np.count_nonzero(patch == 9)) >= 12:
                bridges.add(position)
            elif int(np.count_nonzero(patch == 12)) == 16:
                carriers.add(position)
    return frozenset(destinations), frozenset(bridges), frozenset(pegs), frozenset(carriers)


def solve(frame):
    destinations, bridges, pegs, carriers = extract(frame)
    start = (bridges, pegs)
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        (state_bridges, state_pegs), path = queue.popleft()
        if len(state_pegs) == 1 and state_pegs <= carriers:
            return path, len(seen)
        occupied = state_bridges | state_pegs
        for kind, pieces in (("bridge", state_bridges), ("peg", state_pegs)):
            for source in sorted(pieces):
                for delta_row, delta_col in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                    midpoint = (source[0] + delta_row, source[1] + delta_col)
                    destination = (source[0] + 2 * delta_row, source[1] + 2 * delta_col)
                    if (
                        midpoint not in occupied
                        or destination not in destinations
                        or destination in occupied
                    ):
                        continue
                    child_bridges = set(state_bridges)
                    child_pegs = set(state_pegs)
                    if kind == "bridge":
                        child_bridges.remove(source)
                        child_bridges.add(destination)
                    else:
                        child_pegs.remove(source)
                        child_pegs.add(destination)
                        child_pegs.discard(midpoint)
                    child = (frozenset(child_bridges), frozenset(child_pegs))
                    if child not in seen:
                        seen.add(child)
                        queue.append((
                            child,
                            path + ((kind, source, destination),),
                        ))
    return None, len(seen)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        env.step(action)
    entry = env.clone()
    solution, states = solve(entry.frame())
    clone = entry.clone()
    deltas = []
    for kind, source, destination in solution:
        before = clone.frame()
        clone.step(6, source[1] + 1, source[0] + 1)
        clone.step(6, destination[1] + 1, destination[0] + 1)
        deltas.append(frame_delta(before, clone.frame())["count"])
    print("EXTRACTED", extract(entry.frame()))
    print("SOLUTION", {"moves": len(solution), "states": states, "path": solution})
    print("VERIFY", {
        "levels": clone.levels_completed,
        "deltas": deltas,
        "post_extract": extract(clone.frame()),
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})
