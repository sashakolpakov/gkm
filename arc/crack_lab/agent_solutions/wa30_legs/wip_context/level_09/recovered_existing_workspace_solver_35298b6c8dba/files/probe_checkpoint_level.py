"""Show one checkpoint level entry through compact observational features."""

import os

import gkm_try

from probe_minimize_segments import CaptureSegments
from probe9_verify import boxes, tile_map


def inspect(env):
    capture = CaptureSegments(env)
    gkm_try.resumed_solve(capture)
    level = int(os.environ.get("GKM_LEVEL", "7"))
    start = capture.starts[level - 1]
    route = dict((prior + 1, path) for prior, path in capture.transitions)[
        level
    ]
    print(
        "CHECKPOINT_LEVEL",
        {
            "level": level,
            "route_len": len(route),
            "avatar": boxes(start.frame(), 14),
            "cargo": boxes(start.frame(), 4),
            "couriers": boxes(start.frame(), 12),
            "competitors": boxes(start.frame(), 15),
        },
        flush=True,
    )
    print("CHECKPOINT_MAP", *tile_map(start.frame()), sep="\n", flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
