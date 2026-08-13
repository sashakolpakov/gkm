"""Report action indexes at which the current solver earns each reward."""

import importlib.util
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


spec = importlib.util.spec_from_file_location("solve", "solve.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tracker:
    def __init__(self, inner):
        self.inner = inner
        self.actions_used = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def clone(self):
        return self.inner.clone()

    def step(self, action, *coordinates):
        before = int(self.inner.levels_completed)
        result = self.inner.step(action, *coordinates)
        self.actions_used += 1
        after = int(self.inner.levels_completed)
        if after > before:
            print("solver_boundary", after, self.actions_used, flush=True)
        return result


def probe(env):
    module.solve(Tracker(env))


arena.run_program("lf52", probe)
