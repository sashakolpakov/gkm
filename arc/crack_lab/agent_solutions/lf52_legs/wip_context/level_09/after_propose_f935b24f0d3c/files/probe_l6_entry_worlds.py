"""Enumerate reproduced local-board goal worlds at level 6."""

from collections import deque
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _movable_bridge_board
from perception import safe_step


ENTRY = 238
MAX_DEPTH = int(os.environ.get("ENTRY_MAX_DEPTH", "14"))


def play(node, action):
    safe_step(node, tuple(action) if isinstance(action, list) else action)


def clicks(source, destination):
    return ((6, source[1] + 1, source[0] + 1),
            (6, destination[1] + 1, destination[0] + 1))


def paths(frame):
    slots, carriers, bridges, pegs = _movable_bridge_board(frame)
    start = (frozenset(pegs), next(iter(bridges)))
    queue = deque([(start, ())])
    seen = {start}
    goals = []
    while queue:
        (state_pegs, bridge), path = queue.popleft()
        if len(state_pegs) == 1 and state_pegs <= carriers:
            goals.append(path)
            continue
        if len(path) >= MAX_DEPTH:
            continue
        occupied = state_pegs | {bridge}
        for kind, pieces in (("peg", sorted(state_pegs)),
                             ("bridge", (bridge,))):
            for source in pieces:
                for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                    midpoint = (source[0] + dr, source[1] + dc)
                    destination = (source[0] + 2 * dr, source[1] + 2 * dc)
                    if (destination not in slots | carriers
                            or destination in occupied
                            or midpoint not in occupied
                            or (kind == "bridge" and midpoint not in state_pegs)):
                        continue
                    child_pegs = set(state_pegs)
                    child_bridge = bridge
                    if kind == "peg":
                        child_pegs.remove(source)
                        child_pegs.add(destination)
                        child_pegs.discard(midpoint)
                    else:
                        child_bridge = destination
                    child = (frozenset(child_pegs), child_bridge)
                    if child in seen:
                        continue
                    seen.add(child)
                    queue.append((child, path + ((kind, source, destination),)))
    return goals, len(seen)


def physical_key(node):
    frame = np.asarray(node.frame()).copy()
    frame[0, :] = 0
    return frame.tobytes()


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign[:ENTRY]:
        play(env, action)
    root = env.clone()
    goals, states = paths(root.frame())
    worlds = {}
    for path in goals:
        node = root.clone()
        actions = []
        for _, source, destination in path:
            for action in clicks(source, destination):
                play(node, action)
                actions.append(action)
        worlds.setdefault(physical_key(node), (actions, node))
    print("L6_ENTRY", {"symbolic_states": states, "goals": len(goals),
                       "worlds": len(worlds)}, flush=True)
    for index, (actions, node) in enumerate(worlds.values()):
        board = _movable_bridge_board(node.frame())
        print("WORLD", {"index": index, "cost": len(actions),
                        "actions": actions,
                        "carriers": sorted(board[1]),
                        "bridges": sorted(board[2]),
                        "pegs": sorted(board[3])}, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error)})
