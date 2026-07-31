import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from legs import solve_bridge_carrier_peg_solitaire


def probe(env):
    with open("checkpoint.json") as handle:
        prefix = json.load(handle)["final_path"]
    for action in prefix:
        env.step(*(action if isinstance(action, list) else [action]))
    base_level = env.levels_completed
    solve_bridge_carrier_peg_solitaire(
        env, max_align_states=1200, max_macros=80, alignment_lookahead=80
    )
    print("existing reward", env.levels_completed - base_level)


A.run_program("lf52", probe)
