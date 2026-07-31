"""Narrow local-effect beam search over the four-turn lane-six frontier."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_final_bfs import frontier, step
from probe_l9_control_row import compact
from probe_l9_route_deletions import enter_level_9


def state_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return int(env.levels_completed), bool(env.terminal()), frame.tobytes()


def avatar_cell(env):
    avatars = connected_components(env.frame(), colors=(9,), min_area=3)
    if not avatars:
        return None
    return (
        round((avatars[0].centroid[0] - 3) / 6),
        round((avatars[0].centroid[1] - 3) / 6),
    )


def actions(env):
    cell = avatar_cell(env)
    if cell is None:
        return ()
    _, avatar_col = cell
    result = [3, 4]
    for blob in connected_components(
        env.frame(), colors=(7, 8, 12, 14, 15), min_area=3
    ):
        if blob.bbox[0] >= 63 or blob.area != 21:
            continue
        col = round((blob.centroid[1] - 3) / 6)
        if blob.color in (7, 8, 14) or abs(col - avatar_col) <= 2:
            result.append(
                (6, round(blob.centroid[1]), round(blob.centroid[0]))
            )
    return tuple(dict.fromkeys(result))


def score(env):
    frame = env.frame()
    goals = [
        blob for blob in connected_components(frame, colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    ]
    yellow = [
        blob.bbox[0]
        for blob in connected_components(frame, colors=(14,), min_area=3)
        if blob.bbox[0] < 63
    ]
    return (
        int(env.levels_completed) >= 9,
        bool(goals),
        not env.terminal(),
        60 - min(yellow) if yellow else -1,
        avatar_cell(env) is not None,
    )


def probe(env):
    enter_level_9(env)
    root = frontier(env)
    beam = [(root, ())]
    for depth in range(1, 5):
        unique = {}
        for node, path in beam:
            for action in actions(node):
                child = node.clone()
                step(child, action)
                child_path = path + (action,)
                if int(child.levels_completed) >= 9:
                    print("WIN", child_path, compact(child), flush=True)
                    return
                key = state_key(child)
                current = unique.get(key)
                if current is None:
                    unique[key] = (child, child_path)
        ranked = sorted(
            unique.values(),
            key=lambda item: score(item[0]),
            reverse=True,
        )
        beam = ranked[:80]
        print(
            "DEPTH",
            depth,
            "unique",
            len(unique),
            "kept",
            len(beam),
            "top",
            [(score(node), path, compact(node)) for node, path in beam[:3]],
            flush=True,
        )
    print("NO_WIN", flush=True)
    for node, path in beam[:15]:
        print("BEST", score(node), path, compact(node), flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
