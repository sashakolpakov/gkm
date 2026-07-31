import gkm_try as harness
from perception import action_deltas, color_counts, frame_delta, object_candidates


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in object_candidates(frame)
    ]


def probe(env):
    harness.resumed_solve(env)
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    base = env.frame()
    print("COLORS", color_counts(base))
    print("OBJECTS", compact_objects(base))
    print("KEY_DELTAS", {k: (v["count"], v["bbox"]) for k, v in action_deltas(env).items()})
    for action in (1, 2, 3, 4, 5):
        child = env.clone()
        child.step(action)
        print("AFTER", action, child.levels_completed, compact_objects(child.frame()))
    chars = {0: "o", 4: "#", 8: "x", 11: "A", 15: "."}
    print("MAP")
    for row in base:
        line = "".join(chars.get(int(v), "?") for v in row)
        if line != "." * 64:
            print(line)
    for color, bbox, area in compact_objects(base):
        if color in (4, 8, 11):
            r0, c0, r1, c1 = bbox
            x, y = c0, r0
            selected = env.clone()
            selected.step(6, x, y)
            select_delta = frame_delta(base, selected.frame())
            moves = action_deltas(selected)
            print("SELECT", color, (x, y), area, (select_delta["count"], select_delta["bbox"]),
                  {k: (v["count"], v["bbox"]) for k, v in moves.items()})
            if color in (8, 11):
                print("SELECTED_OBJECTS", color, compact_objects(selected.frame()),
                      color_counts(selected.frame()))


harness.A.run_program("cn04", probe)
