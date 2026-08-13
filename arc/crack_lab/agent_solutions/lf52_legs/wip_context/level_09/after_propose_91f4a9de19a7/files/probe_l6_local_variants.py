"""Enumerate all shortest local level-6 peg/bridge carrier layouts."""

import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board
from perception import safe_step


def move(node, source, destination):
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def goals(frame, max_depth=20):
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    bridge = next(iter(bridges))
    start = (frozenset(pegs), bridge)
    queue = deque([(start, ())])
    seen = {start}
    out = []
    destinations = slots | carriers
    while queue:
        (state_pegs, state_bridge), path = queue.popleft()
        depth = len(path)
        if len(state_pegs) == 1 and next(iter(state_pegs)) in carriers:
            out.append((state_pegs, state_bridge, path))
        if depth >= max_depth:
            continue
        occupied = state_pegs | {state_bridge}
        pieces = tuple(("P", peg) for peg in sorted(state_pegs))
        pieces += (("B", state_bridge),)
        for kind, source in pieces:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = (source[0] + dr, source[1] + dc)
                destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                if (destination not in destinations
                        or destination in occupied
                        or midpoint not in occupied
                        or (kind == "B" and midpoint not in state_pegs)):
                    continue
                child_pegs = set(state_pegs)
                child_bridge = state_bridge
                if kind == "B":
                    child_bridge = destination
                else:
                    child_pegs.remove(source)
                    child_pegs.add(destination)
                    child_pegs.discard(midpoint)
                child = (frozenset(child_pegs), child_bridge)
                if child in seen:
                    continue
                seen.add(child)
                queue.append((child, path + ((source, destination),)))
    return tuple(out), len(seen), tuple(sorted(carriers))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    prior = int(env.levels_completed)
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
        current = int(env.levels_completed)
        if prior < 5 <= current:
            break
        prior = current
    variants, states, carriers = goals(env.frame())
    print("l6_local_goals", states, carriers, len(variants),
          tuple((len(path), tuple(pegs), bridge, path)
                for pegs, bridge, path in variants), flush=True)
    for index, (pegs, bridge, path) in enumerate(variants):
        node = env.clone()
        for source, destination in path:
            move(node, source, destination)
        _, empty_carriers, visible_bridges, visible_pegs = (
            _movable_bridge_board(node.frame())
        )
        options = []
        occupied = visible_bridges | visible_pegs
        for source in visible_bridges:
            for destination in empty_carriers:
                dr = (destination[0] - source[0]) // 2
                dc = (destination[1] - source[1]) // 2
                midpoint = (source[0] + dr, source[1] + dc)
                if (abs(destination[0] - source[0])
                        + abs(destination[1] - source[1]) == 12
                        and midpoint in occupied):
                    options.append((source, destination))
        print("l6_local_replay", index, tuple(sorted(visible_pegs)),
              tuple(sorted(visible_bridges)), tuple(sorted(empty_carriers)),
              tuple(options), flush=True)


arena.run_program("lf52", probe)
