"""Test all in-frame destinations for non-lattice level-9 outcomes."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step
from probe_l9_nonlocal_moves import FIRST_RELAY, board, move


def click(node, point):
    safe_step(node, (6, point[1] + 1, point[0] + 1))


def selected(frame):
    return any(blob.area >= 4 for blob in connected_components(
        frame, colors=(3,)
    ))


def signature(node):
    holes, bridges, pegs, fixed, carriers = board(node.frame())
    return tuple(frozenset(items)
                 for items in (bridges, pegs, fixed, carriers)) + (
                     int(node.levels_completed),
                 )


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    for source, destination in FIRST_RELAY:
        move(env, source, destination)
    context = os.environ.get("OPT_CONTEXT", "far")
    if context == "far":
        for _ in range(9):
            safe_step(env, 4)

    holes, bridges, pegs, fixed, carriers = board(env.frame())
    cells = holes | bridges | pegs | fixed | carriers
    sources = bridges | pegs | fixed | carriers
    ordinary = {signature(env)}
    for source in sources:
        for destination in cells - {source}:
            child = env.clone()
            click(child, source)
            click(child, destination)
            if not selected(child.frame()):
                ordinary.add(signature(child))

    step = int(os.environ.get("OPT_GRID_STEP", "2"))
    tested = 0
    novel = []
    for source in sorted(sources):
        for y in range(0, 64, step):
            for x in range(0, 64, step):
                child = env.clone()
                click(child, source)
                safe_step(child, (6, x, y))
                tested += 1
                child_signature = signature(child)
                if (
                    not selected(child.frame())
                    and child_signature not in ordinary
                ):
                    novel.append((source, (x, y), child_signature))
    print("surface_moves", context, "sources", tuple(sorted(sources)),
          "tested", tested, "novel", tuple(novel), flush=True)


arena.run_program("lf52", probe)
