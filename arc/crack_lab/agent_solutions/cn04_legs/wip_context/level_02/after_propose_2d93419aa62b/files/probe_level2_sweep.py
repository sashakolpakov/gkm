"""Systematically visit in-bounds avatar poses on a level-2 clone."""
import sys
import traceback

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


def avatar_bbox(env):
    blobs = [
        b
        for b in perception.connected_components(env.frame(), colors=(0,), min_area=4)
        if b.bbox[0] > 0
    ]
    return max(blobs, key=lambda b: b.area).bbox


def do(env, path, action):
    env.step(action)
    path.append(action)
    return env.levels_completed > 1


def move_to_origin(env, path):
    while avatar_bbox(env)[0] > 2:
        if do(env, path, 1):
            return True
    while avatar_bbox(env)[1] > 2:
        if do(env, path, 3):
            return True
    return False


def probe(env):
    try:
        play_level_1(env)
        node = env.clone()
        path = []
        for orientation in range(4):
            if move_to_origin(node, path):
                print("solved", len(path), path)
                return
            r0, c0, r1, c1 = avatar_bbox(node)
            rows = (63 - (r1 - r0) - r0) // 3 + 1
            cols = (63 - (c1 - c0) - c0) // 3 + 1
            direction = 4
            for row in range(rows):
                for _ in range(cols - 1):
                    if do(node, path, direction):
                        print("solved", len(path), path)
                        return
                if row + 1 < rows and do(node, path, 2):
                    print("solved", len(path), path)
                    return
                direction = 3 if direction == 4 else 4
            if do(node, path, 5):
                print("solved", len(path), path)
                return
        print("unsolved", len(path), "level", node.levels_completed)
    except Exception:
        traceback.print_exc()


print("run", arena.run_program("cn04", probe))
