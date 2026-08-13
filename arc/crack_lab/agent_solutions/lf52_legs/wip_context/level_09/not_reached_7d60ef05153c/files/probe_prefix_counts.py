"""Compact per-level counts from the validated same-lineage checkpoint."""

import json

import gkm_try


def summarize(env):
    with open("checkpoint.json") as checkpoint_file:
        actions = json.load(checkpoint_file)["final_path"]
    previous = env.levels_completed
    marks = []
    for index, action in enumerate(actions, 1):
        env.step(action)
        if env.levels_completed != previous:
            marks.append((env.levels_completed, index, index - (marks[-1][1] if marks else 0)))
            previous = env.levels_completed
    print("PREFIX_COUNTS", marks)


gkm_try.A.run_program("lf52", summarize)
