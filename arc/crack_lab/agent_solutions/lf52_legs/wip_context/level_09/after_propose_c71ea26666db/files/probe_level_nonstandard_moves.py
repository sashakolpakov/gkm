"""Find completed coordinate moves omitted by the one-lattice move model."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import connected_components, safe_step
from probe_key_neighborhood_events import generic_moves


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def split(path):
    groups = []
    keys = []
    index = 0
    while index < len(path):
        action = normalize(path[index])
        if isinstance(action, int):
            keys.append(action)
            index += 1
        else:
            groups.append((tuple(keys), (action, normalize(path[index + 1]))))
            keys = []
            index += 2
    return tuple(groups)


def candidates(frame):
    blobs = connected_components(frame, colors=(1, 8, 9, 11, 12, 14))
    sources = {
        blob.top_left for blob in blobs
        if (
            blob.size == (4, 4) and blob.color in (8, 9, 12, 14)
            and blob.area >= 12
        ) or (
            blob.color == 11 and blob.size == (6, 6) and blob.area == 20
        )
    }
    destinations = {
        blob.top_left for blob in blobs
        if blob.size == (4, 4) and blob.area >= 12
    }
    return tuple(sorted(sources)), tuple(sorted(destinations))


def probe(env):
    desired = int(os.environ.get("OPT_LEVEL", "7"))
    first_stage = int(os.environ.get("OPT_FIRST_STAGE", "0"))
    last_stage = int(os.environ.get("OPT_LAST_STAGE", "999"))
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
        for action in keys:
            safe_step(node, action)
        if first_stage <= stage <= last_stage:
            before_state = _bridge_carrier_state(node.frame())
            modeled = {
                (source, destination)
                for _, source, destination in generic_moves(node.frame())
            }
            sources, destinations = candidates(node.frame())
            tested = 0
            hits = []
            for source in sources:
                for destination in destinations:
                    if source == destination or (source, destination) in modeled:
                        continue
                    child = node.clone()
                    safe_step(child, (6, source[1] + 1, source[0] + 1))
                    safe_step(child, (6, destination[1] + 1,
                                      destination[0] + 1))
                    tested += 1
                    after_state = _bridge_carrier_state(child.frame())
                    if (
                        after_state[-1] is None
                        and after_state != before_state
                    ):
                        hits.append((source, destination,
                                     int(child.levels_completed)))
            print("nonstandard", desired, stage, "tested", tested,
                  "hits", tuple(hits), flush=True)
        for action in clicks:
            safe_step(node, action)


arena.run_program("lf52", probe)
