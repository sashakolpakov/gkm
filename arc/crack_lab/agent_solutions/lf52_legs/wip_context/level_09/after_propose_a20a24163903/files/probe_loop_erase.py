"""Loop-erase the validated campaign by reproduced public frame states."""

import json

import gkm_try
from perception import arr


def erase(env):
    with open("checkpoint.json") as checkpoint_file:
        original = json.load(checkpoint_file)["final_path"]
    root = env.clone()

    def key(node):
        return node.levels_completed, arr(node.frame())[1:, :].tobytes()

    kept_actions = []
    kept_keys = [key(env)]
    positions = {kept_keys[0]: 0}
    removed = []
    for original_index, action in enumerate(original):
        env.step(action)
        state_key = key(env)
        if state_key in positions:
            keep_index = positions[state_key]
            removed.append((original_index, len(kept_actions) - keep_index + 1))
            for discarded_key in kept_keys[keep_index + 1:]:
                positions.pop(discarded_key, None)
            kept_actions = kept_actions[:keep_index]
            kept_keys = kept_keys[:keep_index + 1]
            continue
        kept_actions.append(action)
        kept_keys.append(state_key)
        positions[state_key] = len(kept_actions)

    replay = root.clone()
    marks = []
    previous = replay.levels_completed
    for index, action in enumerate(kept_actions, 1):
        replay.step(action)
        if replay.levels_completed != previous:
            marks.append((replay.levels_completed, index))
            previous = replay.levels_completed
    print(
        "LOOP_ERASE", len(original), len(kept_actions),
        replay.levels_completed, marks, removed,
    )
    print("LOOP_PATH", kept_actions)


gkm_try.A.run_program("lf52", erase)
