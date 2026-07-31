"""Consume the upper and right glyphs in one lineage, then map destinations."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import (
    MAIN,
    SELECTOR,
    avatar_position,
    movement_reach,
)
from probe_l6_second_glyph import clear_left_half, enter_glyph, return_to_hub


RIGHT0_LOOP = [1, 1, 4, 2, 2, 3]


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def exits(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(11,), min_area=4
        )
        if blob.area == 4 and blob.size == (2, 2) and blob.bbox[1] < 40
    )


def patches(env):
    frame = perception.arr(env.frame())
    return (
        frame[18:20, 6:10].tolist(),
        frame[48:50, 34:38].tolist(),
    )


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    node = clear_left_half(enter_glyph(env))
    print(
        "COMBINED_UPPER", avatar_position(node), patches(node),
        "level", node.levels_completed, "exits", exits(node), flush=True,
    )
    return_to_hub(node)
    node.step(*MAIN)
    print(
        "COMBINED_RIGHT0_ENTRY", avatar_position(node), patches(node),
        "level", node.levels_completed, flush=True,
    )
    for index, action in enumerate(RIGHT0_LOOP, start=1):
        node.step(action)
        if node.levels_completed > base_level:
            print("COMBINED_WIN_RIGHT0", index, flush=True)
            return
    print(
        "COMBINED_BOTH", avatar_position(node), patches(node),
        "level", node.levels_completed, "exits", exits(node), flush=True,
    )

    # Return to the hub, then test every selected endpoint from this lineage.
    node.step(*MAIN)
    print("COMBINED_HUB", avatar_position(node), node.levels_completed)

    cumulative = node.clone()
    # Phase 0 -> phase 2: consume the top endpoint, then return to the hub.
    apply(cumulative, [SELECTOR, SELECTOR, MAIN])
    print(
        "COMBINED_CUM_TOP", avatar_position(cumulative),
        cumulative.levels_completed, "exits", exits(cumulative), flush=True,
    )
    if cumulative.levels_completed > base_level:
        return
    cumulative.step(*MAIN)
    print(
        "COMBINED_CUM_TOP_RETURN", avatar_position(cumulative),
        cumulative.levels_completed, flush=True,
    )
    # Phase 2 -> phase 3: visit the D-pad endpoint and return.
    apply(cumulative, [SELECTOR, MAIN])
    print(
        "COMBINED_CUM_CONTROLLER", avatar_position(cumulative),
        cumulative.levels_completed, "exits", exits(cumulative), flush=True,
    )
    if cumulative.levels_completed > base_level:
        return
    cumulative.step(*MAIN)
    print(
        "COMBINED_CUM_CONTROLLER_RETURN", avatar_position(cumulative),
        cumulative.levels_completed, flush=True,
    )
    # Phase 3 -> phase 1 is the otherwise missing destination.
    apply(cumulative, [SELECTOR, SELECTOR])
    before_missing = avatar_position(cumulative)
    cumulative.step(*MAIN)
    print(
        "COMBINED_CUM_MISSING", before_missing,
        avatar_position(cumulative), cumulative.levels_completed,
        "exits", exits(cumulative), flush=True,
    )
    cumulative_reach, cumulative_win = movement_reach(cumulative)
    print(
        "COMBINED_CUM_MISSING_REACH",
        sorted(
            position for position in cumulative_reach
            if position is not None
        ),
        cumulative_win, flush=True,
    )
    if cumulative_win is not None or cumulative.levels_completed > base_level:
        return

    for selector_offset in range(4):
        branch = node.clone()
        apply(branch, [SELECTOR] * selector_offset)
        before = avatar_position(branch)
        branch.step(*MAIN)
        after = avatar_position(branch)
        reached, win = movement_reach(branch)
        print(
            "COMBINED_DEST", selector_offset,
            "before", before, "after", after,
            "positions", sorted(
                position for position in reached if position is not None
            ),
            "win", win, "level", branch.levels_completed,
            "exits", exits(branch), flush=True,
        )
        if win is not None or branch.levels_completed > base_level:
            return


arena.run_program("dc22", observe)
