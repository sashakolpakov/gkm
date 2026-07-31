import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import players


class LoggedEnv:
    def __init__(self, env):
        self.env = env
        self.actions = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def clone(self):
        return self.env.clone()

    def step(self, *action):
        self.actions.append(tuple(action))
        return self.env.step(*action)


def capture(env):
    logged = LoggedEnv(env)
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(logged)
    print("PREFIX_LEVEL", logged.levels_completed)
    print("PREFIX", tuple(logged.actions))


if __name__ == "__main__":
    A.run_program("bp35", capture)
