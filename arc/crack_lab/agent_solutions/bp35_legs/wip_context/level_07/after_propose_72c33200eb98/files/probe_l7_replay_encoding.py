"""Compare encoded replay entries with unpacked coordinate actions."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape
from perception import frame_delta
from probe_level7_reward_recovery import avatar_cell, controls, lattice


def summary(node):
    return (
        int(node.levels_completed),
        bool(node.terminal()),
        None if node.terminal() else avatar_cell(node.frame()),
        () if node.terminal() else tuple(controls(node.frame())),
        "" if node.terminal() else lattice(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    with open("frontier_scaffold.json") as stream:
        scaffold = json.load(stream)
    raw = scaffold["staged_prefix_actions"]
    base = env.frame()

    encoded = env.clone()
    encoded.step(raw[0])
    unpacked = env.clone()
    unpacked.step(*raw[0])
    print(
        "REPLAY_ENCODING_ONE",
        raw[0],
        "encoded", frame_delta(base, encoded.frame()),
        _cell_shape(encoded.frame(), 2, 2),
        "unpacked", frame_delta(base, unpacked.frame()),
        _cell_shape(unpacked.frame(), 2, 2),
        flush=True,
    )

    node = env.clone()
    route = []
    for entry in raw:
        node.step(entry)
        route.append(entry)
        if node.terminal() or node.levels_completed > 6:
            break
    print("REPLAY_ENCODING_STAGE", len(route), summary(node), flush=True)
    if node.levels_completed > 6:
        print("REPLAY_ENCODING_WIN", route, flush=True)
        return
    if node.terminal():
        return
    for entry in [3, 3, 3, 3]:
        node.step(entry)
        route.append(entry)
        if node.terminal() or node.levels_completed > 6:
            break
    print("REPLAY_ENCODING_WALK", len(route), summary(node), flush=True)
    if node.levels_completed > 6:
        print("REPLAY_ENCODING_WIN", route, flush=True)
        return
    if node.terminal():
        return
    for y in controls(node.frame()):
        for encoded_final in (True, False):
            child = node.clone()
            action = [6, 3, y]
            if encoded_final:
                child.step(action)
            else:
                child.step(*action)
            print(
                "REPLAY_ENCODING_FINAL", y, encoded_final,
                summary(child), flush=True,
            )
            if child.levels_completed > 6:
                print(
                    "REPLAY_ENCODING_WIN",
                    [*route, action], encoded_final, flush=True,
                )
                return


arena.run_program("bp35", probe)
