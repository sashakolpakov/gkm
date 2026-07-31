"""C serves bottom plus one side socket; sweep the remaining connector."""
import itertools
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_local_l6 import path_for, replay, safe_order


PORT_TOP = {"UL": 20, "R": 29, "LL": 35}
PORT_ROW = {"UL": 23, "R": 32, "LL": 38}


def placements(piece, port):
    if piece in ("A", "D"):
        return ((PORT_TOP[port], 14 if port != "R" else 44),)
    left = 14 if port != "R" else 47
    gap = PORT_ROW[port]
    return tuple(
        (top, left) for top in range(11, 39, 3)
        if top - 3 <= gap <= top + 12
    )


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    started = time.monotonic()
    tested = 0
    steps = 0
    ports = ("UL", "R", "LL")
    only_port = sys.argv[1] if len(sys.argv) > 1 else None
    for c_port in ports:
        if only_port and c_port != only_port:
            continue
        remaining_ports = tuple(port for port in ports if port != c_port)
        for assigned in itertools.permutations("ABD", 2):
            if "--db" in sys.argv and set(assigned) != {"B", "D"}:
                continue
            connector = next(piece for piece in "ABD" if piece not in assigned)
            choices = tuple(
                placements(piece, port)
                for piece, port in zip(assigned, remaining_ports)
            )
            for assigned_positions in itertools.product(*choices):
                fixed = {
                    "C": (PORT_ROW[c_port], 23),
                    **dict(zip(assigned, assigned_positions)),
                }
                for top, left in itertools.product(
                    tuple(range(11, 45, 3)),
                    tuple(range(14, 45, 3)),
                ):
                    targets = dict(fixed)
                    targets[connector] = (top, left)
                    order = safe_order(targets)
                    if order is None:
                        continue
                    path = path_for(targets, order)
                    result = replay(env, path)
                    tested += 1
                    steps += len(path)
                    if result.levels_completed > env.levels_completed:
                        print(
                            "DUAL_MARKER_WIN", targets, "ORDER", order,
                            "TESTED", tested, "STEPS", steps, "PATH", path,
                        )
                        return
                    delay = steps / 280.0 - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(delay)
    print("DUAL_MARKER_NONE", "TESTED", tested, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
