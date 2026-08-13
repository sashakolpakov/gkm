"""Count real actions per existing player on a fresh public arena run."""

import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

import players


class CountingEnv:
    def __init__(self, env):
        self.wrapped_env = env
        self.steps = 0

    def step(self, *action):
        self.steps += 1
        return self.wrapped_env.step(*action)

    def __getattr__(self, name):
        return getattr(self.wrapped_env, name)


def profile(env):
    totals = []
    while not env.terminal() and env.levels_completed < 8:
        level = int(env.levels_completed) + 1
        player = getattr(players, f"play_level_{level}")
        counted = CountingEnv(env)
        before = int(env.levels_completed)
        player(counted)
        totals.append((level, counted.steps, int(env.levels_completed)))
        if env.levels_completed <= before:
            break
    print("PREFIX_PROFILE", tuple(totals), "sum", sum(item[1] for item in totals))


levels, path, error = arena.run_program("lf52", profile)
print("PROBE_RESULT", levels, len(path), error)
