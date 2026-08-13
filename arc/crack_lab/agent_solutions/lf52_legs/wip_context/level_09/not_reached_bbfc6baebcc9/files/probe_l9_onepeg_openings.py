"""Replay level-9 local collapses whose survivor is not in the carrier."""

import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step


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
    carrier = next(
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    )
    return tuple(sorted(holes | pegs | bridges)), frozenset(pegs), frozenset(bridges), carrier


def move(env, source, destination):
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def onepeg_paths(frame):
    cells, pegs, bridges, carrier = parse(frame)
    cell_set = frozenset(cells) | {carrier}
    start = (pegs, bridges)
    queue = deque([(start, ())])
    seen = {start}
    goals = []
    while queue:
        (state_pegs, state_bridges), path = queue.popleft()
        if len(state_pegs) == 1:
            survivor = next(iter(state_pegs))
            if survivor != carrier:
                goals.append((len(path), survivor,
                              tuple(sorted(state_bridges)), path))
            continue
        occupied = state_pegs | state_bridges
        for kind, sources in (("P", state_pegs), ("B", state_bridges)):
            for source in sorted(sources):
                for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                    midpoint = (source[0] + dr, source[1] + dc)
                    destination = (source[0] + 2 * dr,
                                   source[1] + 2 * dc)
                    if (midpoint not in occupied
                            or destination not in cell_set
                            or destination in occupied):
                        continue
                    child_pegs = set(state_pegs)
                    child_bridges = set(state_bridges)
                    if kind == "P":
                        child_pegs.remove(source)
                        child_pegs.add(destination)
                        child_pegs.discard(midpoint)
                    else:
                        child_bridges.remove(source)
                        child_bridges.add(destination)
                    child = (frozenset(child_pegs), frozenset(child_bridges))
                    if child in seen:
                        continue
                    seen.add(child)
                    queue.append((child, path + ((source, destination),)))
    return tuple(sorted(goals, key=lambda item: (item[0], item[1], item[2])))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    goals = onepeg_paths(env.frame())
    print("onepeg_models", len(goals),
          tuple((depth, survivor, bridges)
                for depth, survivor, bridges, _ in goals), flush=True)
    for index, (depth, survivor, bridges, path) in enumerate(goals[:24]):
        child = env.clone()
        for source, destination in path:
            move(child, source, destination)
        _, actual_pegs, actual_bridges, _ = parse(child.frame())
        print("onepeg_replay", index, depth, survivor,
              int(child.levels_completed), tuple(sorted(actual_pegs)),
              tuple(sorted(actual_bridges)), path, flush=True)


if __name__ == "__main__":
    arena.run_program("lf52", probe)
