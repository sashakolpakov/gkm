"""Minimize one current per-level route from a fresh arena."""

import os

import gkm_try

from probe_minimize_segments import CaptureSegments, minimize
from probe_pair_minimize import encode, pair_minimize, triple_minimize


def inspect(env):
    capture = CaptureSegments(env)
    gkm_try.m.players.play_level_9 = lambda level_env: None
    gkm_try.m.solve(capture)
    segments = {level + 1: route for level, route in capture.transitions}
    level = int(os.environ.get("GKM_FRESH_OPT_LEVEL", "8"))
    route = segments[level]
    mode = os.environ.get("GKM_FRESH_OPT_MODE", "segment")
    minimizer = {
        "segment": minimize,
        "pair": pair_minimize,
        "triple": triple_minimize,
    }[mode]
    best, turns = minimizer(capture.starts[level - 1], route)
    print(
        "FRESH_SEGMENT_RESULT",
        level,
        mode,
        len(route),
        len(best),
        turns,
        encode(best),
        best,
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
