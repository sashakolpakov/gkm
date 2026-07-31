"""Screen earlier lane changes before descending toward the prize wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip4_switch_choices import report
from probe_l9_supported_final_alignment import aligned


def goals(env):
    return any(
        blob.bbox[0] < 63
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
    )


def probe(env):
    enter_level_9(env)
    for align_depth in range(7):
        for column in (5, 6, 7):
            child = aligned(env, align_depth, column)
            x = 3 + 6 * column
            first_goal = None
            for descent in range(13):
                if goals(child) or child.terminal() or int(child.levels_completed) >= 9:
                    first_goal = descent
                    break
                child.step(6, x, 33)
            print(
                "END",
                "align",
                align_depth,
                "column",
                column,
                "descent",
                first_goal,
                "terminal",
                bool(child.terminal()),
                "levels",
                int(child.levels_completed),
                "controls",
                controls(child),
                flush=True,
            )
            report((align_depth, column), child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
