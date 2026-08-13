"""Map action 7 before and after carrier loading on level 7."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import arr, connected_components, frame_delta, safe_step


LOAD_KEYS = (3, 3, 1, 1, 3, 3, 3)
UNLOAD_KEYS = (4, 4, 4, 2, 2, 3, 3, 2)


def delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    out = frame_delta(left, right)
    return out["count"], out["bbox"]


def compact(frame):
    state = _bridge_carrier_state(frame)
    arrows = tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(7,))
    )
    return (tuple(sorted(state[1])), tuple(sorted(state[2])),
            tuple(sorted(state[4])), state[5], arrows)


def describe(name, node):
    before = node.frame()
    outcomes = []
    frames = {}
    for action in (1, 2, 3, 4, 7):
        child = node.clone()
        safe_step(child, action)
        frames[action] = arr(child.frame())[1:, :].tobytes()
        outcomes.append((action, delta(before, child.frame()),
                         compact(child.frame())))
    equivalents = tuple((action, other)
                        for action in (1, 2, 3, 4, 7)
                        for other in (1, 2, 3, 4, 7)
                        if action < other and frames[action] == frames[other])
    repeated = node.clone()
    trail = []
    for count in range(1, 13):
        prior = repeated.frame()
        safe_step(repeated, 7)
        trail.append((count, delta(prior, repeated.frame()),
                      compact(repeated.frame())))
    print("context", name, "base", compact(before),
          "outcomes", tuple(outcomes), "equivalent", equivalents,
          "trail7", tuple(trail), flush=True)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    prior = int(env.levels_completed)
    entry = None
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
        current = int(env.levels_completed)
        if prior < 6 <= current:
            entry = env.clone()
            break
        prior = current

    describe("entry", entry.clone())
    aligned = entry.clone()
    for action in LOAD_KEYS:
        safe_step(aligned, action)
    describe("aligned", aligned.clone())
    loaded = aligned.clone()
    safe_step(loaded, (6, 7, 13))
    safe_step(loaded, (6, 7, 25))
    describe("loaded", loaded.clone())
    ready = loaded.clone()
    for action in UNLOAD_KEYS:
        safe_step(ready, action)
    describe("ready", ready.clone())
    unloaded = ready.clone()
    safe_step(unloaded, (6, 13, 43))
    safe_step(unloaded, (6, 13, 55))
    describe("unloaded", unloaded.clone())


arena.run_program("lf52", probe)
