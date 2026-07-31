import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _body_groups, _solid_playfield_squares


reason = G._workspace_taint_reason(os.getcwd())
if reason:
    raise SystemExit(f"TAINTED WORKSPACE: {reason}")


LARGE_TARGET = (41, 11)
SMALL_TARGET = (55, 11)


def distance(blob, target):
    row, col = map(round, blob.centroid)
    return max(abs(row - target[0]), abs(col - target[1]))


def key(env):
    bodies = tuple(
        (color, tuple(_body_groups(env, color))) for color in (7, 14, 13)
    )
    solids = tuple(
        (blob.color, blob.bbox)
        for blob in _solid_playfield_squares(env, colors=(6, 8, 15))
    )
    return bodies, solids


def staged(env):
    large = _solid_playfield_squares(env, colors=(8,))
    small = _solid_playfield_squares(env, colors=(15,))
    return (
        len(large) == 1
        and len(small) == 1
        and distance(large[0], LARGE_TARGET) <= 1
        and distance(small[0], SMALL_TARGET) <= 1
    )


def rank(env, depth):
    large = _solid_playfield_squares(env, colors=(8,))
    merged = _solid_playfield_squares(env, colors=(15,))
    pieces = _solid_playfield_squares(env, colors=(6,))
    if len(large) != 1:
        return 10000 + depth
    large_cost = distance(large[0], LARGE_TARGET)
    if len(merged) == 1:
        small_cost = distance(merged[0], SMALL_TARGET)
    elif sum(blob.area for blob in pieces) == 8:
        small_cost = 12 + min(distance(blob, SMALL_TARGET) for blob in pieces)
    else:
        return 9000 + depth
    return 20 * (large_cost + small_cost) + depth


def actions(env):
    proposed = []
    large = _solid_playfield_squares(env, colors=(8,))
    merged = _solid_playfield_squares(env, colors=(15,))
    pieces = _solid_playfield_squares(env, colors=(6,))
    for blobs, target in ((large, LARGE_TARGET), (merged, SMALL_TARGET)):
        if len(blobs) == 1 and distance(blobs[0], target) > 1:
            blob = blobs[0]
            row, col = blob.centroid
            step = blob.size[0] + 1
            click_row = row + max(-step, min(step, target[0] - row))
            click_col = col + max(-step, min(step, target[1] - col))
            proposed.append((6, round(click_col), round(click_row)))
    if not merged:
        for blob in pieces:
            proposed.append((6, blob.bbox[1], blob.bbox[0]))
    for color in (7, 14, 13):
        for group in _body_groups(env, color):
            proposed.extend((6, col, row) for row, col in group)
    proposed.append((6, 32, 32))
    return tuple(dict.fromkeys(proposed))


def program(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(*action)

    frontier = [(env.clone(), ())]
    seen = {key(env)}
    for depth in range(1, 13):
        candidates = []
        for node, path in frontier:
            for action in actions(node):
                child = node.clone()
                child.step(*action)
                child_rank = rank(child, depth)
                if child_rank >= 9000:
                    continue
                child_path = path + (action,)
                if staged(child):
                    print("STAGED", child_path, key(child), flush=True)
                    return
                child_key = key(child)
                if child_key in seen:
                    continue
                seen.add(child_key)
                candidates.append((child_rank, child_key, child, child_path))
        candidates.sort(key=lambda item: (item[0], item[1]))
        frontier = [(item[2], item[3]) for item in candidates[:128]]
        print(
            "DEPTH",
            depth,
            "BEST",
            candidates[0][0] if candidates else None,
            "KEPT",
            len(frontier),
            "SEEN",
            len(seen),
            flush=True,
        )
        if not frontier:
            return


print("RUN", A.run_program("su15", program)[0])
