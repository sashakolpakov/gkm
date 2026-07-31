import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


TRANSFER_HORIZONTAL = (
    [3] * 6 + [2] + [3] * 3 + [2] * 3 + [4] * 2 + [1] * 4
    + [(6, 35, 34)] + [1]
)


def pieces(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(0, 5, 11, 14), min_area=2
        )
    )


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    for left in (6, 7, 8):
        for delay in range(6):
            node = env.clone()
            path = TRANSFER_HORIZONTAL + [3] * left + [2] * delay + [1] * delay
            for item in path:
                node.step(*item) if isinstance(item, tuple) else node.step(item)
            # The down/up pair is phase-only when the ring is below the barrier.
            for _ in range(3):
                node.step(1)
            before = pieces(node)
            changed = []
            for index in range(1, 7):
                node.step(1)
                after = pieces(node)
                if after != before:
                    changed.append((index, after))
                before = after
            print("x", 33 - 3 * left, "phase", delay, "trace", changed)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
