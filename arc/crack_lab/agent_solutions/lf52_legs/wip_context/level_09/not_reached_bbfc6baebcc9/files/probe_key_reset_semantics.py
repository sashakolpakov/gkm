"""Determine whether action 7 rewinds carrier keys while preserving cargo moves."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import arr, frame_delta, safe_step


LOAD_KEYS = (3, 3, 1, 1, 3, 3, 3)
UNLOAD_KEYS = (4, 4, 4, 2, 2, 3, 3, 2)


def delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    right[0] = left[0]
    out = frame_delta(left, right)
    return out["count"], out["bbox"]


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

    initial = entry.frame()
    keyed = entry.clone()
    for action in LOAD_KEYS:
        safe_step(keyed, action)
    after_keys = keyed.frame()
    safe_step(keyed, 7)
    print("keys_reset", delta(initial, after_keys),
          delta(initial, keyed.frame()),
          _bridge_carrier_state(initial),
          _bridge_carrier_state(keyed.frame()), flush=True)

    loaded = entry.clone()
    for action in LOAD_KEYS:
        safe_step(loaded, action)
    safe_step(loaded, (6, 7, 13))
    safe_step(loaded, (6, 7, 25))
    after_load = loaded.frame()
    for action in UNLOAD_KEYS:
        safe_step(loaded, action)
    before_reset = loaded.frame()
    safe_step(loaded, 7)
    print("cargo_reset_one", delta(after_load, before_reset),
          delta(after_load, loaded.frame()),
          _bridge_carrier_state(after_load),
          _bridge_carrier_state(loaded.frame()), flush=True)
    safe_step(loaded, 7)
    print("cargo_reset_two", delta(after_load, loaded.frame()),
          _bridge_carrier_state(loaded.frame()), flush=True)


arena.run_program("lf52", probe)
