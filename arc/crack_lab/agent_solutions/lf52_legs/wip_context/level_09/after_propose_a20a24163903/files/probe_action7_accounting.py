"""Observe how a reset action appears in the documented replay path."""

import json

import gkm_try

from perception import safe_step


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix:
        safe_step(env, action)
    safe_step(env, 4)
    safe_step(env, 7)


levels, path, error = gkm_try.A.run_program("lf52", probe)
print("ACTION7_ACCOUNTING", levels, len(path), tuple(path[-4:]), error)
