"""Compact bridge/carrier states at validated levels 4 and 5."""

import json
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves_from_state, _bridge_carrier_state
from perception import action_deltas


BOUNDARIES = {87: 4, 149: 5}


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for index, action in enumerate(checkpoint["final_path"][:149], 1):
        env.step(action)
        if index in BOUNDARIES:
            state = _bridge_carrier_state(env.frame())
            deltas = {
                key: (delta["count"], delta["bbox"])
                for key, delta in action_deltas(env).items()
            }
            print(
                "LEVEL_ENTRY", BOUNDARIES[index],
                "state", state, "moves", _bridge_carrier_moves_from_state(state),
                "deltas", deltas,
            )


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)
