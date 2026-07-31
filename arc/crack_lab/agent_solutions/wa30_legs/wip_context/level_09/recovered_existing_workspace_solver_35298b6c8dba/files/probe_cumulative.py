"""Report cumulative real-action handoffs for the current composed solver."""

import gkm_try


class Handoffs:
    def __init__(self, env):
        self.env = env
        self.turn = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        before = self.env.levels_completed
        result = self.env.step(action, *args)
        self.turn += 1
        if self.env.levels_completed > before:
            print("HANDOFF", self.env.levels_completed, self.turn, flush=True)
        return result


def inspect(env):
    wrapped = Handoffs(env)
    try:
        gkm_try.resumed_solve(wrapped)
    except RuntimeError as error:
        print("HANDOFF_ERROR", wrapped.turn, str(error), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
