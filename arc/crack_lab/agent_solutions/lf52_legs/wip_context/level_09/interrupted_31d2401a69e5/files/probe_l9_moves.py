"""Enumerate currently selectable level-9 pieces and highlighted landings."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, color_counts, replay, safe_step


POINTS = tuple((row, col) for row in range(6, 61, 6)
               for col in range(6, 61, 6))


def selectable_moves(node):
    base = arr(node.frame())
    found = []
    for row, col in POINTS:
        child = node.clone()
        safe_step(child, (6, col + 1, row + 1))
        after = arr(child.frame())
        if not (base[1:, :] != after[1:, :]).any():
            continue
        destinations = []
        for dest_row, dest_col in POINTS:
            if (after[dest_row:dest_row + 4, dest_col:dest_col + 4] == 2).any():
                destinations.append((dest_row, dest_col))
        found.append(((row, col), int(base[row + 1, col + 1]),
                      tuple(destinations)))
    return tuple(found)


def observe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)

    first = ((6, 19, 43), (6, 31, 43))
    second = first + ((6, 25, 43), (6, 37, 43))
    vertical_bridge = ((6, 25, 49), (6, 25, 37))
    contexts = {
        "root": (),
        "peg_across_bridge": first,
        "bridge_across_peg": second,
        "bridge_across_bridge": vertical_bridge,
        "vertical_then_peg": vertical_bridge + first,
    }
    for count in range(1, 5):
        contexts[f"carrier_right_{count}"] = (4,) * count
        contexts[f"two_moves_then_right_{count}"] = second + (4,) * count

    for name, path in contexts.items():
        node = replay(env, path)
        print("CONTEXT", name, "path_len", len(path),
              "colors", color_counts(node.frame()),
              "moves", selectable_moves(node))


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
