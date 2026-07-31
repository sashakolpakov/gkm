import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import legs
import perception as P


def parts(node):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in P.connected_components(
            node.frame(), colors=(6, 9, 10, 13, 14, 15), min_area=2
        )
    )


def trace_moves(env, level, action, count):
    for index in range(1, count + 1):
        before = parts(env)
        old_level = env.levels_completed
        env.step(action)
        after = parts(env)
        if before != after or env.levels_completed != old_level:
            print("MOVE", level, index, action, old_level, env.levels_completed, before, after)
        if env.levels_completed != old_level:
            break


def probe(env):
    print("START", 1, parts(env))
    legs.prime_board(env)
    legs.select_grid_cells_of_color(env, (25, 30, 35), (50, 55, 60), 0)
    print("READY", 1, parts(env))
    trace_moves(env, 1, 3, 16)

    print("START", 2, parts(env))
    legs.select_grid_cells_of_color(env, (25, 30, 35), (50, 55, 60), 0)
    print("READY", 2, parts(env))
    trace_moves(env, 2, 1, 8)

    print("START", 3, parts(env))
    trace_moves(env, 3, 4, 3)
    legs.select_grid_cells_of_color(env, (25, 30, 35), (50, 55, 60), 0)
    print("READY", 3, parts(env))
    trace_moves(env, 3, 3, 4)
    trace_moves(env, 3, 2, 4)
    trace_moves(env, 3, 3, 2)


levels, path, err = A.run_program("sc25", probe)
print("DONE", levels, len(path), err)
