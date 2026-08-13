"""Trace level-5 carrier and piece positions along the validated solution."""

import json
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import connected_components, safe_step


PREFIX_END = 149
LEVEL_END = 238


def symbolic(frame):
    slots, pegs, _, bridges, _, _ = _bridge_carrier_state(frame)
    carriers = tuple(
        sorted(
            (blob.bbox[0] + 1, blob.bbox[1] + 1)
            for blob in connected_components(frame, colors=(11,))
            if blob.area >= 4
        )
    )
    contents = tuple(
        (position, "peg" if position in pegs else "bridge" if position in bridges else "empty")
        for position in carriers
    )
    fixed_bridges = tuple(sorted(bridges - set(carriers)))
    return tuple(sorted(pegs)), contents, fixed_bridges, len(slots)


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    full_path = checkpoint["final_path"]
    for action in full_path[:PREFIX_END]:
        env.step(action)
    actions = full_path[PREFIX_END:LEVEL_END]
    print("TRACE_START", symbolic(env.frame()))
    index = 0
    group = 0
    while index < len(actions):
        group += 1
        if isinstance(actions[index], int):
            states = []
            while index < len(actions) and isinstance(actions[index], int):
                action = actions[index]
                safe_step(env, action)
                states.append((action, symbolic(env.frame())))
                index += 1
            print("TRACE_KEYS", group, tuple(states))
        else:
            macro = actions[index:index + 2]
            before = symbolic(env.frame())
            for action in macro:
                safe_step(env, action)
            print("TRACE_MOVE", group, tuple(macro), before, symbolic(env.frame()))
            index += 2


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)
