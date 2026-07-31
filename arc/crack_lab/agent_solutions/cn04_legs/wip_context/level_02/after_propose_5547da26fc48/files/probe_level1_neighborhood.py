"""Map rewarded level-1 placements near the known final maneuver."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception


def main_blob(env):
    blobs = perception.connected_components(env.frame(), colors=(0,), min_area=4)
    return None if not blobs else max(blobs, key=lambda b: b.area).bbox


def black_cells(env):
    a = perception.arr(env.frame())
    return sorted(
        (r // 3, c // 3)
        for r in range(3, 61, 3)
        for c in range(3, 61, 3)
        if int(a[r, c]) == 0
    )


def probe(env):
    for down in range(5, 10):
        row = []
        for right in range(2, 7):
            node = perception.replay(env, [2] * down + [4] * right + [5] * 3)
            row.append((down, right, node.levels_completed, main_blob(node)))
        print(row)
    neighbor = perception.replay(env, [2] * 7 + [4] * 3 + [5] * 3)
    print("left_neighbor_cells", black_cells(neighbor))


arena.run_program("cn04", probe)
