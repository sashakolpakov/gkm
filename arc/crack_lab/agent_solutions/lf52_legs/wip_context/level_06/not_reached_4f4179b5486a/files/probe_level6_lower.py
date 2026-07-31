import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P


LOCAL_MACROS = (
    ((18, 18), (30, 18)),
    ((24, 18), (36, 18)),
    ((30, 18), (42, 18)),
    ((36, 18), (48, 18)),
    ((48, 12), (48, 24)),
    ((54, 24), (42, 24)),
    ((42, 18), (42, 30)),
    ((42, 24), (42, 36)),
    ((42, 30), (42, 42)),
    ((42, 36), (42, 48)),
)
UPPER_MACROS = (
    ((30, 28), (18, 28)),
    ((18, 28), (18, 40)),
    ((18, 46), (18, 34)),
    ((18, 40), (18, 28)),
    ((18, 28), (30, 28)),
)


def click(env, cell):
    env.step(6, cell[1] + 1, cell[0] + 1)


def pieces(frame):
    blobs = P.connected_components(frame, colors=(1, 8, 12, 14, 15))
    holes = {
        blob.top_left for blob in blobs
        if blob.color == 1 and blob.size == (4, 4) and blob.area == 16
    }
    carriers = {
        blob.top_left for blob in blobs
        if blob.color == 12 and blob.size == (4, 4)
    }
    movable = {
        blob.top_left for blob in blobs
        if blob.color == 8 and blob.size == (4, 4)
    }
    pegs = {
        blob.top_left for blob in blobs
        if blob.color == 14 and blob.size == (4, 4)
    }
    static = {
        (blob.bbox[0] + 1, blob.bbox[1]) for blob in blobs
        if blob.color == 15 and blob.size == (4, 4)
    }
    return holes, carriers, movable, pegs, static


def moves(frame):
    holes, carriers, movable, pegs, static = pieces(frame)
    occupied = movable | pegs | static
    out = []
    for source in sorted(movable | pegs):
        for destination in sorted(holes | carriers):
            dr = destination[0] - source[0]
            dc = destination[1] - source[1]
            if (abs(dr), abs(dc)) not in ((12, 0), (0, 12)):
                continue
            midpoint = (source[0] + dr // 2, source[1] + dc // 2)
            if midpoint in occupied - {source}:
                out.append((source, midpoint, destination))
    return tuple(out)


def brief(env):
    _, carriers, movable, pegs, _ = pieces(env.frame())
    return tuple(sorted(carriers)), tuple(sorted(movable)), tuple(sorted(pegs)), moves(env.frame())


def stage_upper(env):
    with open("checkpoint.json") as handle:
        prefix = json.load(handle)["final_path"]
    for action in prefix:
        env.step(*(action if isinstance(action, list) else [action]))
    for source, destination in LOCAL_MACROS:
        click(env, source)
        click(env, destination)
    _, carriers, movable, _, _ = pieces(env.frame())
    click(env, next(iter(movable)))
    click(env, next(iter(carriers)))
    for _ in range(9):
        env.step(4)
    for action in (1, 1, 4, 1):
        env.step(action)
    click(env, (30, 28))
    click(env, (18, 28))
    env.step(2)
    for _ in range(3):
        env.step(3)
    for _ in range(2):
        env.step(1)
    for source, destination in UPPER_MACROS:
        click(env, source)
        click(env, destination)


def probe(env):
    stage_upper(env)
    base_level = env.levels_completed
    print("upper", brief(env))
    for action in (4, 3):
        node = env.clone()
        trace = []
        previous = None
        for count in range(7):
            current = brief(node)
            if current == previous:
                break
            trace.append((count, current))
            previous = current
            node.step(action)
        print("reload scan", action, trace)

    reloaded = env.clone()
    reloaded.step(4)
    reloaded.step(4)
    print("reload ready", brief(reloaded))
    click(reloaded, (18, 34))
    click(reloaded, (30, 34))
    print("reloaded", brief(reloaded))

    for vertical in (1, 2):
        node = reloaded.clone()
        trace = []
        previous = None
        for count in range(7):
            current = brief(node)
            if current == previous:
                break
            trace.append((count, current))
            previous = current
            node.step(vertical)
        print("vertical", vertical, trace)

    lower = reloaded.clone()
    lower.step(2)
    lower.step(2)
    print("lower rail", brief(lower))
    for action in (3, 4):
        node = lower.clone()
        trace = []
        previous = None
        for count in range(12):
            current = brief(node)
            if current == previous:
                break
            trace.append((count, current))
            previous = current
            node.step(action)
        print("lower horizontal", action, trace)
    print("reward", reloaded.levels_completed - base_level)


if __name__ == "__main__":
    A.run_program("lf52", probe)
