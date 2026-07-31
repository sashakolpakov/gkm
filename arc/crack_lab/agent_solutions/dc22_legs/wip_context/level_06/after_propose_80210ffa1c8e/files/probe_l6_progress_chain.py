"""Carry the consumed right-hand glyph into ring and physical-world probes."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_exact_crossings import (
    CENTER,
    TO_CENTER,
    exact_reach,
    horizontal_entry,
    placement_label,
    placements_with_paths,
)
from probe_l6_right import (
    MAIN,
    SELECTOR,
    avatar_position,
    enter_right,
)


def exit_tiles(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(11,), min_area=4
        )
        if blob.area == 4
        and blob.size == (2, 2)
        and blob.bbox[1] < 40
    )


def cleared_right0(env):
    node = enter_right(env, 0)
    for action in (1, 1, 4, 4, 3, 3, 2, 2):
        node.step(action)
    node.step(*MAIN)
    for _ in range(3):
        node.step(*SELECTOR)
    node.step(*MAIN)
    return node


def exact_walk(root):
    base_level = root.levels_completed
    queue = deque([(root.clone(), [])])
    seen = {perception.arr(root.frame())[:63].tobytes()}
    while queue:
        node, path = queue.popleft()
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_path = path + [action]
            if child.levels_completed > base_level:
                return child_path
            key = perception.arr(child.frame())[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, child_path))
    return None


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    root = cleared_right0(env)
    placements = placements_with_paths(root)
    print(
        "PROGRESS_ROOT", len(placements), exit_tiles(root),
        root.levels_completed - base_level,
    )
    for index, (placement, cargo_path) in enumerate(placements):
        if placement.levels_completed > base_level or exit_tiles(placement):
            print(
                "PROGRESS_RING_HIT", index, cargo_path,
                placement_label(placement), exit_tiles(placement),
                placement.levels_completed - base_level,
            )
            return
        hub = placement.clone()
        position = avatar_position(hub)
        if position != CENTER:
            hub.step(TO_CENTER[position])
        hub.step(*MAIN)
        destination = hub.clone()
        for selector_offset in range(4):
            destination.step(*MAIN)
            win = exact_walk(destination)
            if (
                win is not None
                or destination.levels_completed > base_level
                or exit_tiles(destination)
            ):
                print(
                    "PROGRESS_PORTAL_HIT", index, selector_offset,
                    avatar_position(destination), win,
                    exit_tiles(destination),
                    destination.levels_completed - base_level,
                    cargo_path,
                )
                return
            # Return to the hub when this state has a destination.
            if avatar_position(destination) != (48, 18):
                # The entry cell is the active portal for states 0, 2, and 3.
                destination.step(*MAIN)
            destination.step(*SELECTOR)
        print("PROGRESS_PORTALS_DONE", index)
    for index in (11, 13, 18):
        physical = horizontal_entry(placements[index][0])
        win, states, partial = exact_reach(
            physical, max_states=300, max_depth=50
        )
        print(
            "PROGRESS_PHYSICAL", index, placement_label(placements[index][0]),
            states, len(partial), win,
            physical.levels_completed - base_level,
            exit_tiles(physical),
        )
        if win is not None:
            print("PROGRESS_CARGO_PATH", placements[index][1])
            return


arena.run_program("dc22", observe)
