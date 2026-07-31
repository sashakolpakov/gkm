"""Measure current solver transitions without replaying the checkpoint."""

import gkm_try


class TimedEnv:
    def __init__(self, env):
        self.env = env
        self.moves = 0
        self.transitions = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        before = self.env.levels_completed
        result = self.env.step(action, *args)
        self.moves += 1
        if self.env.levels_completed > before:
            self.transitions.append((self.env.levels_completed, self.moves))
        return result


def inspect(env):
    timed = TimedEnv(env)
    gkm_try.m.solve(timed)
    print(
        "SOLVER_TIMING",
        {
            "moves": timed.moves,
            "level": env.levels_completed,
            "terminal": env.terminal(),
            "transitions": tuple(timed.transitions),
        },
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
