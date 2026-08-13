"""Reproduce level 8's aligned opening and print its symbolic lattice."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components, safe_step


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:476]:
        safe_step(env, action)
    for action in (3, 3, 1, 1, 1, 1, 4):
        safe_step(env, action)
    groups = {}
    for blob in connected_components(env.frame(), colors=(1, 9, 11, 12,
                                                           14, 15)):
        if blob.area < 4:
            continue
        groups.setdefault((blob.color, blob.size, blob.area), []).append(
            blob.top_left
        )
    print("ALIGNED", {key: tuple(value) for key, value in sorted(groups.items())})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})
