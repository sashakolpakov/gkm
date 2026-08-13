"""Verify public undo restoration on key and coordinate actions."""

import json

import gkm_try
from perception import arr


def verify(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"][:149]
    for action in prefix:
        env.step(action)
    root = arr(env.frame())[1:, :].tobytes()

    env.step(3)
    key_changed = arr(env.frame())[1:, :].tobytes() != root
    env.step(7)
    key_restored = arr(env.frame())[1:, :].tobytes() == root

    env.step(6, 13, 25)
    selected = arr(env.frame())[1:, :].tobytes()
    env.step(6, 25, 25)
    moved = arr(env.frame())[1:, :].tobytes()
    env.step(7)
    destination_undone = arr(env.frame())[1:, :].tobytes() == selected
    env.step(7)
    source_undone = arr(env.frame())[1:, :].tobytes() == root
    print(
        "UNDO", key_changed, key_restored, moved != selected,
        destination_undone, source_undone, env.levels_completed,
    )


gkm_try.A.run_program("lf52", verify)
