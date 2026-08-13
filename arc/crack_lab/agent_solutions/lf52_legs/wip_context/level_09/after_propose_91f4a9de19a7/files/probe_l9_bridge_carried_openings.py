"""Find one-local-peg openings that can load a bridge into the carrier."""

import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step
from probe_l9_onepeg_openings import move, onepeg_paths, parse


def bridge_load_path(cells, carrier, survivor, start, max_depth=12):
    destinations = frozenset(cells) | {carrier}
    queue = deque([(frozenset(start), ())])
    seen = {frozenset(start)}
    while queue:
        bridges, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        occupied = bridges | {survivor}
        for source in sorted(bridges):
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (midpoint not in occupied
                        or destination not in destinations
                        or destination in occupied):
                    continue
                child_path = path + ((source, destination),)
                if destination == carrier:
                    return child_path, len(seen)
                child = frozenset((bridges - {source}) | {destination})
                if child in seen:
                    continue
                seen.add(child)
                queue.append((child, child_path))
    return None, len(seen)


def first_bridge_cargo(frame, max_depth=24):
    cells, pegs, bridges, carrier = parse(frame)
    destinations = frozenset(cells) | {carrier}
    start = (pegs, bridges)
    queue = deque([(start, ())])
    seen = {start}
    goals = []
    goal_depth = None
    while queue:
        (state_pegs, state_bridges), path = queue.popleft()
        if goal_depth is not None and len(path) >= goal_depth:
            continue
        if len(path) >= max_depth:
            continue
        occupied = state_pegs | state_bridges
        for kind, sources in (("P", state_pegs), ("B", state_bridges)):
            for source in sorted(sources):
                for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                    midpoint = (source[0] + dr, source[1] + dc)
                    destination = (source[0] + 2 * dr,
                                   source[1] + 2 * dc)
                    if (midpoint not in occupied
                            or destination not in destinations
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
                    child_path = path + ((source, destination),)
                    if kind == "B" and destination == carrier:
                        goal_depth = len(child_path)
                        goals.append((child, child_path))
                        continue
                    if child in seen:
                        continue
                    seen.add(child)
                    queue.append((child, child_path))
    return tuple(goals), len(seen), carrier


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    cargo_goals, cargo_states, cargo_carrier = first_bridge_cargo(env.frame())
    print("first_bridge_cargo", cargo_states, cargo_carrier,
          tuple((len(path), len(state[0]), tuple(sorted(state[0])),
                 tuple(sorted(state[1])), path)
                for state, path in cargo_goals), flush=True)
    cells, _, _, carrier = parse(env.frame())
    candidates = []
    for depth, survivor, bridge_tuple, path in onepeg_paths(env.frame()):
        bridge_path, states = bridge_load_path(
            cells, carrier, survivor, bridge_tuple
        )
        if bridge_path is not None:
            candidates.append((depth + len(bridge_path), depth, survivor,
                               bridge_tuple, bridge_path, path, states))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    print("bridge_load_models", len(candidates),
          tuple((total, depth, survivor, bridge_path, states)
                for total, depth, survivor, _, bridge_path, _, states
                in candidates), flush=True)

    for index, candidate in enumerate(candidates[:8]):
        total, depth, survivor, _, bridge_path, path, _ = candidate
        node = env.clone()
        for step in path:
            move(node, *step)
        for step in bridge_path:
            move(node, *step)
        scan = []
        for offset in range(13):
            blobs = connected_components(node.frame(), colors=(9, 12, 14))
            pieces = tuple(sorted(
                (blob.color, blob.top_left)
                for blob in blobs
                if blob.size == (4, 4)
                and (blob.color != 9 or blob.area == 12)
            ))
            scan.append((offset, pieces))
            safe_step(node, 4)
        print("bridge_load_replay", index, 2 * total, survivor,
              int(node.levels_completed), tuple(scan), flush=True)


arena.run_program("lf52", probe)
