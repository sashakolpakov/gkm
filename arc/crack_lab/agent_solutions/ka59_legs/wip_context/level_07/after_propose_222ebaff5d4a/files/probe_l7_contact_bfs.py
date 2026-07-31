import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, bounded_bfs, connected_components


def pieces(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63],
            colors=(0, 5, 11, 12, 13, 14),
            min_area=1,
        )
    )


def large(env):
    return tuple(p for p in pieces(env) if p[0] == 11)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    initial_large = large(env)
    for name, click in (("horizontal", (35, 52)),):
        root = env.clone()
        if click:
            root.step(6, *click)
        path = bounded_bfs(
            root,
            lambda node, _: large(node) != initial_large,
            actions=(1, 2, 3, 4),
            key_fn=pieces,
            max_states=12000,
            max_depth=160,
        )
        print(name, "path", path)
        if path:
            node = root.clone()
            for action in path:
                node.step(action)
            print(name, "verified", len(path), initial_large, large(node))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
