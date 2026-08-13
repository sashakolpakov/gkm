"""Test whether public undo history survives rewarded level transitions."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, frame_delta, safe_step


def normalize(action):
    return tuple(action) if isinstance(action, list) else action


def delta(before, after):
    left = arr(before).copy()
    right = arr(after).copy()
    left[0] = right[0]
    result = frame_delta(left, right)
    return result["count"], result["bbox"]


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = tuple(normalize(action)
                         for action in json.load(stream)["final_path"])
    for index, action in enumerate(campaign):
        old_level = int(env.levels_completed)
        safe_step(env, action)
        if int(env.levels_completed) == old_level:
            continue
        baseline = env.frame()
        outcomes = []
        for undos in (1, 2):
            node = env.clone()
            for _ in range(undos):
                safe_step(node, 7)
            after_undo = (int(node.levels_completed), delta(baseline,
                                                           node.frame()))
            replay_actions = campaign[max(0, index - undos + 1):index + 1]
            for replay_action in replay_actions:
                safe_step(node, replay_action)
            outcomes.append((undos, after_undo,
                             int(node.levels_completed), replay_actions,
                             delta(baseline, node.frame())))
        print("transition_undo", old_level + 1, index + 1,
              tuple(outcomes), flush=True)


if __name__ == "__main__":
    arena.run_program("lf52", probe)
