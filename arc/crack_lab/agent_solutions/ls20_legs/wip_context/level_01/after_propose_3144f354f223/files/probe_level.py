import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
from perception import (
    action_deltas,
    bounded_bfs,
    color_counts,
    connected_components,
    level_goal,
    replay,
)


def state(env):
    keep = {0, 8, 9, 11, 12}
    blobs = connected_components(env.frame(), colors=keep, min_area=1)
    return (
        int(env.levels_completed),
        bool(env.terminal()),
        [(b.color, b.bbox, b.area) for b in blobs],
    )


def probe(env):
    print("actions", list(env.actions))
    print("colors", color_counts(env.frame()))
    blobs = connected_components(env.frame(), min_area=2)
    print("blobs", [(b.color, b.bbox, b.area) for b in blobs])
    print(
        "deltas",
        {
            action: (delta["count"], delta["bbox"], delta["samples"][:8])
            for action, delta in action_deltas(env, env.actions).items()
        },
    )
    tests = {
        "U": [1],
        "UU": [1, 1],
        "UUUUUUUU": [1] * 8,
        "U9": [1] * 9,
        "U10": [1] * 10,
        "U12": [1] * 12,
        "U8L": [1] * 8 + [3],
        "U8R": [1] * 8 + [4],
        "LLLLLLLL": [3] * 8,
        "RRRRRRRR": [4] * 8,
        "ULULULUL": [1, 3] * 4,
        "idleD8": [2] * 8,
        "idleD30": [2] * 30,
        "idleD42": [2] * 42,
        "idleD45": [2] * 45,
        "idleD60": [2] * 60,
        "idleD90": [2] * 90,
        "idleD130": [2] * 130,
        "idleD140": [2] * 140,
    }
    print("paths", {name: state(replay(env, path)) for name, path in tests.items()})
    def avatar_key(node):
        blobs = connected_components(node.frame(), colors={12}, min_area=1)
        return blobs[0].bbox if blobs else None

    path = bounded_bfs(
        env,
        level_goal(int(env.levels_completed)),
        actions=env.actions,
        key_fn=avatar_key,
        max_states=1000,
        max_depth=100,
    )
    print("avatar_bfs", path, state(replay(env, path or [])))
    def mechanism_key(node):
        frame = __import__("numpy").asarray(node.frame())
        return __import__("numpy").where(
            __import__("numpy").isin(frame, (0, 8, 9, 12)), frame, 255
        ).astype("uint8").tobytes()

    path = bounded_bfs(
        env,
        level_goal(int(env.levels_completed)),
        actions=env.actions,
        key_fn=mechanism_key,
        max_states=20000,
        max_depth=100,
    )
    print("mechanism_bfs", path, state(replay(env, path or [])))


arena.run_program("ls20", probe)
