"""Print compact carrier/controller changes along one admitted campaign level."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state
from perception import connected_components, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def split(path):
    groups = []
    keys = []
    index = 0
    while index < len(path):
        if isinstance(path[index], int):
            keys.append(path[index])
            index += 1
        else:
            groups.append((tuple(keys), (path[index], path[index + 1])))
            keys = []
            index += 2
    return tuple(groups)


def controller(frame):
    slots, pegs, carriers, bridges, borders, selected = (
        _bridge_carrier_state(frame)
    )
    return (tuple(sorted(carriers)), tuple(sorted(bridges)),
            tuple(sorted(borders)), selected, len(slots), len(pegs))


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "5"))
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    prior = int(env.levels_completed)
    entry = None
    start = end = None
    for index, action in enumerate(campaign):
        safe_step(env, action)
        current = int(env.levels_completed)
        if prior < desired - 1 <= current:
            entry = env.clone()
            start = index + 1
        if prior < desired <= current:
            end = index + 1
            break
        prior = current
    node = entry
    for stage, (keys, clicks) in enumerate(split(campaign[start:end])):
        start_frame = node.frame()
        trace = [(None, controller(node.frame()))]
        for action in keys:
            safe_step(node, action)
            trace.append((action, controller(node.frame())))
        stage_start = int(os.environ.get("OPT_STAGE_START", "0"))
        stage_end = int(os.environ.get("OPT_STAGE_END", "999"))
        if stage_start <= stage <= stage_end:
            if os.environ.get("OPT_STATE") == "1":
                borders = tuple(
                    (blob.bbox, blob.size, blob.area)
                    for blob in connected_components(
                        start_frame, colors=(11,)
                    )
                )
                print("raw_state", stage,
                      _bridge_carrier_state(start_frame),
                      "moves", _bridge_carrier_moves(start_frame),
                      "border_blobs", borders, flush=True)
            if os.environ.get("OPT_COMPACT") == "1":
                print("stage", stage, "keys", keys, "start", trace[0][1],
                      "end", trace[-1][1], "clicks", clicks, flush=True)
            else:
                print("stage", stage, "trace", tuple(trace), "clicks", clicks,
                      flush=True)
        for action in clicks:
            safe_step(node, action)


arena.run_program("lf52", probe)
