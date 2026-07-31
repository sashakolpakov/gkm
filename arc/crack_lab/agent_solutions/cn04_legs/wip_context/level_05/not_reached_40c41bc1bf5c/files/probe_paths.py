import gkm_try as harness
from perception import color_counts, connected_components


def compact(env):
    frame = env.frame()
    blobs = [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=4)
        if b.color != 15
    ]
    return env.levels_completed, color_counts(frame), blobs


def probe(env):
    harness.resumed_solve(env)
    paths = {
        "connect": [4] * 4 + [[6, 5, 38]] + [1] * 4 + [4] * 10
                   + [[6, 47, 47]] + [1] * 7,
        "all_center": [4] * 4 + [[6, 5, 38]] + [1] * 5 + [4] * 10
                      + [[6, 47, 47]] + [1] * 8,
        "all_up": [1] * 30 + [[6, 5, 38]] + [1] * 30
                  + [[6, 47, 47]] + [1] * 30,
    }
    for name, path in paths.items():
        node = env.clone()
        old_colors = None
        for i, action in enumerate(path):
            node.step(action)
            colors = color_counts(node.frame())
            marker = tuple(colors.get(c, 0) for c in range(16))
            if name == "all_up" and marker != old_colors:
                print("TRACE", i + 1, action, node.levels_completed, colors)
                old_colors = marker
            if node.levels_completed > 4:
                print(name, "WIN", i + 1, path[: i + 1])
                break
        else:
            print(name, "END", compact(node))


harness.A.run_program("cn04", probe)
