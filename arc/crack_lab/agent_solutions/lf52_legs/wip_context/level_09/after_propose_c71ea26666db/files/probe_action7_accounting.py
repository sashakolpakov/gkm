"""Check how the public harness records one action followed by action 7."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


def probe(env):
    env.step(1)
    env.step(7)


levels, path, error = arena.run_program("lf52", probe)
print("action7_accounting", levels, len(path), tuple(path), error, flush=True)
