"""Walk every destination cell under all ring/global configurations."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_exact_crossings import (
    CENTER,
    TO_CENTER,
    placement_label,
    placements_with_paths,
)
from probe_l6_right import (
    MAIN,
    SELECTOR,
    TOP,
    avatar_position,
    enter_right,
)


DIRECTIONS = (
    (1, 2, -2, 0),
    (2, 1, 2, 0),
    (3, 4, 0, -2),
    (4, 3, 0, 2),
)


def walk_tour(env):
    frame = perception.arr(env.frame())
    start = avatar_position(env)
    if start is None:
        return []
    seen = {start}
    actions = []

    def visit(position):
        row, col = position
        for action, inverse, dr, dc in DIRECTIONS:
            target = row + dr, col + dc
            nr, nc = target
            if target in seen or not (0 <= nr < 62 and 0 <= nc < 40):
                continue
            block = frame[nr:nr + 2, nc:nc + 2]
            support = sum(
                int(value) not in {0, 4, 5, 15}
                for value in block.flat
            )
            if support < 2:
                continue
            seen.add(target)
            actions.append(action)
            visit(target)
            actions.append(inverse)

    visit(start)
    return actions


def exits(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(11,), min_area=4
        )
        if blob.area == 4 and blob.size == (2, 2) and blob.bbox[1] < 40
    )


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    placements = placements_with_paths(enter_right(env, 3))
    checked = 0
    walked = 0
    for index, (placement, _) in enumerate(placements):
        centered = placement.clone()
        position = avatar_position(centered)
        if position != CENTER:
            centered.step(TO_CENTER[position])
        for orientation in range(2):
            oriented = centered.clone()
            if orientation:
                oriented.step(4)
                oriented.step(*MAIN)
                oriented.step(3)
            for bridge_phase in range(6):
                staged = oriented.clone()
                for _ in range(bridge_phase):
                    staged.step(*TOP)
                staged.step(*MAIN)
                for selector_offset in range(4):
                    destination = staged.clone()
                    for _ in range(selector_offset):
                        destination.step(*SELECTOR)
                    destination.step(*MAIN)
                    checked += 1
                    tour = walk_tour(destination)
                    for step_index, action in enumerate(tour, start=1):
                        destination.step(action)
                        walked += 1
                        if destination.levels_completed > base_level:
                            print(
                                "GLOBAL_TOUR_WIN", (
                                    index, placement_label(placement),
                                    orientation, bridge_phase,
                                    selector_offset,
                                ),
                                "tour_step", step_index,
                                "tour_prefix", tour[:step_index],
                                "checked", checked, "walked", walked,
                                flush=True,
                            )
                            return
                    visible = exits(destination)
                    if visible:
                        print(
                            "GLOBAL_TOUR_EXIT", (
                                index, placement_label(placement),
                                orientation, bridge_phase, selector_offset,
                            ),
                            visible, "checked", checked, flush=True,
                        )
                        return
        print(
            "GLOBAL_TOUR_DONE", index,
            "checked", checked, "walked", walked, flush=True,
        )
    print(
        "GLOBAL_TOUR_NO_WIN", checked, walked, flush=True,
    )


if __name__ == "__main__":
    arena.run_program("dc22", observe)
