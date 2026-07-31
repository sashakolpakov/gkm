"""Exact symbolic model and dense-progress planner for lp85 level 6."""
import heapq
import itertools
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from probe_l6 import CONTROLS
from solve import solve


DIRS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1))
BOARDS = (
    ((15, 14), 3, (27, 26)),
    ((15, 44), 5, (27, 32)),
    ((45, 29), 0, (33, 29)),
)


def read_board(frame, spec):
    (cr, cc), _port_angle, (br, bc) = spec
    cells = []
    for radius in range(1, 4):
        for dr, dc in DIRS:
            cells.append(int(frame[cr + 3 * radius * dr,
                                   cc + 3 * radius * dc]))
    return tuple(cells) + (int(frame[br, bc]),)


def angular(state):
    out = list(state)
    for radius in range(3):
        row = state[radius * 8:(radius + 1) * 8]
        out[radius * 8:(radius + 1) * 8] = row[1:] + row[:1]
    return tuple(out)


def radial(state):
    return state[16:24] + state[0:16] + state[24:]


def bad_columns(state):
    return sum(len({state[a], state[8 + a], state[16 + a]}) < 3
               for a in range(8))


def edge_conflicts(state):
    """Equal-color neighbors on visible rings and radial spokes."""
    ring_edges = sum(
        state[8 * r + a] == state[8 * r + (a + 1) % 8]
        for r in range(3) for a in range(8)
    )
    radial_edges = sum(
        state[8 * r + a] == state[8 * (r + 1) + a]
        for r in range(2) for a in range(8)
    )
    return ring_edges + radial_edges


def swap(state, index):
    out = list(state)
    out[index], out[24] = out[24], out[index]
    return tuple(out)


def find_distinct_plan(start, max_states=200000):
    serial = itertools.count()
    heap = [(bad_columns(start), 0, next(serial), start, ())]
    best = {start: 0}
    while heap and len(best) <= max_states:
        _f, cost, _serial, state, path = heapq.heappop(heap)
        if cost != best[state]:
            continue
        bad = bad_columns(state)
        if bad == 0:
            return path, state, len(best)
        for index in range(24):
            child = swap(state, index)
            child_cost = cost + 1
            if child_cost >= best.get(child, 10**9):
                continue
            best[child] = child_cost
            heapq.heappush(
                heap,
                (child_cost + bad_columns(child), child_cost,
                 next(serial), child, path + (index,)),
            )
    return None, None, len(best)


def sampled(frame):
    return tuple(read_board(frame, spec) for spec in BOARDS)


def run(env):
    solve(env)
    frame = np.asarray(env.frame())
    states = sampled(frame)
    predicted = []
    for board, state in enumerate(states):
        predicted.append((angular(state), radial(state)))
        print("board", board, "bad", bad_columns(state),
              "edges", edge_conflicts(state), "buffer", state[24])
        plan, final, seen = find_distinct_plan(state)
        print("plan", plan, "len", None if plan is None else len(plan),
              "final_bad", None if final is None else bad_columns(final),
              "final_edges", None if final is None else edge_conflicts(final),
              "seen", seen)

    for board, (angular_control, radial_control) in enumerate(
            ((CONTROLS[0], CONTROLS[1]),
             (CONTROLS[4], CONTROLS[6]),
             (CONTROLS[2], CONTROLS[3]))):
        for kind, control, expected in (
                ("angular", angular_control, predicted[board][0]),
                ("radial", radial_control, predicted[board][1])):
            clone = env.clone()
            clone.step(6, *control)
            actual = read_board(np.asarray(clone.frame()), BOARDS[board])
            print("validate", board, kind, actual == expected)


if __name__ == "__main__":
    A.run_program("lp85", run)
