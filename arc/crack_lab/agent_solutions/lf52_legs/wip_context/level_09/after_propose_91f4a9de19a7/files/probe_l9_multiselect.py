"""Test whether several level-9 pieces can be selected and moved together."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step
from probe_l9_nonlocal_moves import FIRST_RELAY, board, move


def signature(node):
    holes, bridges, pegs, fixed, carriers = board(node.frame())
    return tuple(frozenset(items)
                 for items in (bridges, pegs, fixed, carriers)) + (
                     int(node.levels_completed),
                 )


def selection(frame):
    return tuple(sorted(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(3,))
        if blob.area >= 4
    ))


def click(node, point):
    safe_step(node, (6, point[1] + 1, point[0] + 1))


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    for source, destination in FIRST_RELAY:
        move(env, source, destination)
    if os.environ.get("OPT_CONTEXT", "far") == "far":
        for _ in range(9):
            safe_step(env, 4)

    holes, bridges, pegs, fixed, carriers = board(env.frame())
    cells = holes | bridges | pegs | fixed | carriers
    sources = bridges | pegs | fixed | carriers
    base = signature(env)
    ordinary = {base}
    for source in sources:
        for destination in cells - {source}:
            child = env.clone()
            click(child, source)
            click(child, destination)
            ordinary.add(signature(child))

    selected_pairs = []
    novel = []
    tested = 0
    for first in sorted(sources):
        for second in sorted(sources - {first}):
            staged = env.clone()
            click(staged, first)
            click(staged, second)
            staged_selection = selection(staged.frame())
            if len(staged_selection) > 1:
                selected_pairs.append((first, second, staged_selection))
            for destination in sorted(cells - {first, second}):
                child = staged.clone()
                click(child, destination)
                tested += 1
                child_signature = signature(child)
                if child_signature not in ordinary:
                    novel.append((first, second, destination,
                                  staged_selection,
                                  selection(child.frame()),
                                  child_signature))
    print("multiselect", os.environ.get("OPT_CONTEXT", "far"),
          "sources", tuple(sorted(sources)), "tested", tested,
          "selected_pairs", tuple(selected_pairs),
          "novel", tuple(novel), flush=True)


def checked(env):
    try:
        probe(env)
    except Exception as error:
        print("multiselect_error", repr(error), flush=True)
        raise


arena.run_program("lf52", checked)
