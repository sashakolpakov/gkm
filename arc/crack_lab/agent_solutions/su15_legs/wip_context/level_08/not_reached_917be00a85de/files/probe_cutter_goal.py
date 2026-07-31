import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from legs import merge_equal_squares_around_moving_cutter
from perception import connected_components


class Recorder:
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


def summary(env):
    frame = env.frame()
    pieces = tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=1)
        if (
            blob.bbox[0] >= 10
            and blob.size[0] == blob.size[1]
            and blob.area == blob.size[0] ** 2
            and blob.color not in (3, 4, 5, 7, 9)
        )
    )
    cutters = tuple(
        blob.bbox
        for blob in connected_components(frame, colors=(7,), min_area=4)
        if blob.bbox[0] >= 10
    )
    rings = tuple(
        blob.bbox
        for blob in connected_components(frame, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )
    return pieces, cutters, rings


def inspect(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    print("AT", int(env.levels_completed), summary(env))
    start = int(env.levels_completed)
    recorder = Recorder(env.clone())
    merge_equal_squares_around_moving_cutter(recorder)
    print("PLAN", recorder.actions)
    node = env.clone()
    for index, action in enumerate(recorder.actions, 1):
        before = summary(node)
        node.step(*action)
        if int(node.levels_completed) > start:
            print("WIN", index, "before", before)
            return
    print("NO_WIN", int(node.levels_completed), summary(node))


A.run_program("su15", inspect)
