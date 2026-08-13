"""Exact joint loading search on the pristine level-6 local board."""

from collections import deque
import json

import gkm_try

from legs import _movable_bridge_board
from perception import safe_step


LEVEL_START = 238


def joint(frame, max_states=200000):
    holes, carriers, bridges, pegs = _movable_bridge_board(frame)
    if len(bridges) != 1:
        return None, 0
    bridge = next(iter(bridges))
    slots = frozenset(holes | carriers | bridges | pegs)
    start = frozenset(pegs), bridge
    queue = deque([(start, ())])
    seen = {start}
    while queue and len(seen) <= max_states:
        (state_pegs, state_bridge), path = queue.popleft()
        if (
            len(state_pegs) == 1
            and next(iter(state_pegs)) in carriers
            and state_bridge in carriers
        ):
            return path, len(seen)
        occupied = state_pegs | {state_bridge}
        sources = tuple(("P", point) for point in sorted(state_pegs))
        sources += (("B", state_bridge),)
        for kind, source in sources:
            for dr, dc in ((-6, 0), (6, 0), (0, -6), (0, 6)):
                midpoint = source[0] + dr, source[1] + dc
                destination = source[0] + 2 * dr, source[1] + 2 * dc
                if midpoint not in occupied or destination not in slots or destination in occupied:
                    continue
                child_pegs, child_bridge = set(state_pegs), state_bridge
                if kind == "P":
                    child_pegs.remove(source); child_pegs.add(destination)
                    child_pegs.discard(midpoint)
                else:
                    child_bridge = destination
                child = frozenset(child_pegs), child_bridge
                if child in seen:
                    continue
                seen.add(child); queue.append((child, path + ((kind, source, destination),)))
    return None, len(seen)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:LEVEL_START]:
        safe_step(env, action)
    solution, searched = joint(env.frame())
    print("L6_JOINT", searched, solution)
    if solution is None:
        return
    replay = env.clone()
    for _, source, destination in solution:
        safe_step(replay, (6, source[1] + 1, source[0] + 1))
        safe_step(replay, (6, destination[1] + 1, destination[0] + 1))
    print("L6_JOINT_VERIFY", 2 * len(solution), _movable_bridge_board(replay.frame())[1:])


gkm_try.A.run_program("lf52", probe)
