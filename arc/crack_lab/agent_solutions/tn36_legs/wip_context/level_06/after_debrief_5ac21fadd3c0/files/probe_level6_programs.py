"""Test topology-derived six-step direction programs on pristine level 6."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import solve
from legs import click_largest_color_9_submit_disc, find_right_segment_panel
from perception import arr, frame_delta


DIRECTION_ROWS = {
    "D": (0, 1),
    "U": (0, 5),
    "R": (1,),
    "L": (0,),
}


def write_program(env, program):
    frame = arr(env.frame()).copy()
    _, rows, cols = find_right_segment_panel(frame)
    for col, direction in zip(cols, program):
        desired = DIRECTION_ROWS[direction]
        for index, row in enumerate(rows):
            if (int(frame[row][col]) == 5) != (index in desired):
                env.step(6, col, row)


def top_feedback(frame):
    return "".join(str(int(frame[1][col])) for col in range(1, 62))


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    solve.solve(env)

    base_level = env.levels_completed
    for program in (
        "RRRRUU",
        "RRRRRR",
        "RRRRRL",
        "RRRRRU",
        "RUUUUU",
        "UUUUUR",
        "LLLLLL",
        "DDDDDD",
    ):
        clone = env.clone()
        write_program(clone, program)
        before_submit = arr(clone.frame()).copy()
        click_largest_color_9_submit_disc(clone)
        print(
            "program",
            {
                "value": program,
                "level_delta": clone.levels_completed - base_level,
                "submit_delta": frame_delta(before_submit, clone.frame()),
                "feedback": top_feedback(clone.frame()),
            },
        )


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
