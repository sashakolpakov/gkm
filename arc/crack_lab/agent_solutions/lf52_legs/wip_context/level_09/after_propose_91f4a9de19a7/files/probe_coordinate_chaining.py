"""Check whether a moved peg remains selected for a following jump."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


def click_move(env, source, destination):
    safe_step(env, (6, source[1] + 1, source[0] + 1))
    safe_step(env, (6, destination[1] + 1, destination[0] + 1))


def peg_key(env):
    return tuple(
        blob.top_left
        for blob in connected_components(env.frame(), colors=(14,))
        if blob.size == (4, 4)
    )


def piece_key(env):
    return tuple(
        (blob.color, blob.top_left, blob.size, blob.area)
        for blob in connected_components(env.frame(), colors=(3, 8, 9, 14))
        if blob.size in ((2, 2), (4, 4)) or blob.color == 3
    )


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    prior = int(env.levels_completed)
    entry = None
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
        current = int(env.levels_completed)
        if prior < 8 <= current:
            entry = env.clone()
            break
        prior = current

    setup = (
        ((42, 18), (42, 30)),
        ((48, 24), (36, 24)),
        ((42, 24), (30, 24)),
        ((36, 24), (24, 24)),
        ((30, 24), (18, 24)),
    )
    root = entry.clone()
    for source, destination in setup:
        click_move(root, source, destination)

    normal = root.clone()
    click_move(normal, (18, 24), (18, 36))
    click_move(normal, (18, 36), (30, 36))

    chained = root.clone()
    click_move(chained, (18, 24), (18, 36))
    safe_step(chained, (6, 37, 31))

    print(
        "coordinate_chain",
        peg_key(root),
        peg_key(normal),
        peg_key(chained),
        arr(normal.frame())[1:, :].tobytes()
        == arr(chained.frame())[1:, :].tobytes(),
        "root", piece_key(root),
        "normal", piece_key(normal),
        "chained", piece_key(chained),
        flush=True,
    )


arena.run_program("lf52", probe)
