"""Dense panel scan from every selector position at the exact ring dock."""
from collections import defaultdict
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_exact_crossings import (
    CENTER,
    TO_CENTER,
    placements_with_paths,
)
from probe_l6_right import avatar_position, enter_right


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    node = placements_with_paths(enter_right(env, 3))[13][0]
    position = avatar_position(node)
    if position != CENTER:
        node.step(TO_CENTER[position])
    for selector_move in (None, 1, 2, 3, 4):
        staged = node.clone()
        if selector_move is not None:
            staged.step(selector_move)
        before = perception.arr(staged.frame()).copy()
        effects = defaultdict(list)
        for y in range(22, 49):
            for x in range(42, 60):
                branch = staged.clone()
                branch.step(6, x, y)
                if branch.levels_completed > base_level:
                    print(
                        "DOCK_PANEL_WIN", selector_move,
                        avatar_position(staged), (x, y),
                    )
                    return
                delta = perception.frame_delta(before, branch.frame())
                if any(sample[0] < 63 for sample in delta["samples"]):
                    effects[(delta["count"], delta["bbox"])].append((x, y))
        print(
            "DOCK_PANEL_EFFECTS", selector_move, avatar_position(staged),
            tuple((key, points) for key, points in effects.items()),
        )


arena.run_program("dc22", observe)
