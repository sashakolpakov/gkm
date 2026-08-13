"""Test coordinate selections that persist across level-9 carrier scrolling."""

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


def points(frame):
    holes, bridges, pegs, fixed, carriers = board(frame)
    return holes | bridges | pegs | fixed | carriers, (
        bridges | pegs | fixed | carriers
    )


def ordinary_signatures(root):
    cells, sources = points(root.frame())
    outcomes = {signature(root)}
    for source in sources:
        for destination in cells - {source}:
            child = root.clone()
            click(child, source)
            click(child, destination)
            if not selected(child.frame()):
                outcomes.add(signature(child))
    return outcomes


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    for source, destination in FIRST_RELAY:
        move(env, source, destination)

    start_offset = int(os.environ.get("OPT_OFFSET", "0"))
    direction = int(os.environ.get("OPT_DIRECTION", "4"))
    steps = int(os.environ.get("OPT_STEPS", "14"))
    for _ in range(start_offset):
        safe_step(env, 4)
    _, initial_sources = points(env.frame())
    baseline = env.clone()
    contexts = []
    for _ in range(steps):
        safe_step(baseline, direction)
        cells, _ = points(baseline.frame())
        contexts.append((ordinary_signatures(baseline), cells))

    novel = []
    tested = 0
    for source in sorted(initial_sources):
        selected_root = env.clone()
        click(selected_root, source)
        for count, (ordinary, cells) in enumerate(contexts, 1):
            safe_step(selected_root, direction)
            for destination in sorted(cells):
                child = selected_root.clone()
                click(child, destination)
                tested += 1
                if (
                    not selected(child.frame())
                    and signature(child) not in ordinary
                ):
                    novel.append((source, count, destination,
                                  int(child.levels_completed),
                                  signature(child)))
    print("selection_scroll", start_offset, direction, steps,
          "sources", tuple(sorted(initial_sources)), "tested", tested,
          "novel", tuple(novel), flush=True)


def checked(env):
    try:
        probe(env)
    except Exception as error:
        print("selection_scroll_error", repr(error), flush=True)
        raise


arena.run_program("lf52", checked)
