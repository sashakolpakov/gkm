import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components


def search(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    base_level = env.levels_completed
    blobs = connected_components(env.frame(), min_area=4)
    palette = {
        blob.color: (round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in blobs
        if blob.bbox[0] >= 54 and blob.area == 16 and blob.color != 5
    }
    slots = [(22, 22), (22, 28), (22, 40),
             (36, 22), (36, 28), (36, 34), (36, 40)]
    examined = 0

    def visit(node, depth, remaining, assignment):
        nonlocal examined
        if depth == len(slots):
            examined += 1
            child = node.clone()
            child.step(5)
            if child.levels_completed > base_level:
                return assignment
            if examined % 1000 == 0:
                print("examined", examined, flush=True)
            return None
        row, col = slots[depth]
        for color in remaining:
            child = node.clone()
            child.step(6, *palette[color])
            child.step(6, col, row)
            result = visit(
                child,
                depth + 1,
                tuple(value for value in remaining if value != color),
                assignment + [color],
            )
            if result is not None:
                return result
        return None

    result = visit(env.clone(), 0, tuple(sorted(palette)), [])
    print("solution", result, "examined", examined)


A.run_program("sb26", search)
