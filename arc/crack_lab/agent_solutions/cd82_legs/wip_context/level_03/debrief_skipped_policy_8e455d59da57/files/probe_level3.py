import gkm_try as harness
from legs import move_vessel_below_and_apply, apply_current_then_select_and_apply_southeast
from perception import arr, color_counts, frame_delta


CH = {0: "0", 2: ".", 3: "-", 4: "#", 5: " ", 8: "8", 9: "9",
      11: "B", 12: "C", 14: "E", 15: "X"}


def crop(frame, r0, r1, c0, c1):
    a = arr(frame)
    return "/".join("".join(CH[int(x)] for x in row)
                    for row in a[r0:r1, c0:c1])


def summary(env, label):
    a = arr(env.frame())
    vessel = [(o.color, o.bbox, o.area) for o in __import__("perception").connected_components(a, min_area=8)
              if 18 <= o.bbox[0] <= 45 and o.color in (0, 2, 15)]
    print(label, "level", env.levels_completed, "colors", color_counts(a),
          "work", crop(a, 34, 44, 27, 37), "vessel", vessel)


def probe(env):
    move_vessel_below_and_apply(env)
    apply_current_then_select_and_apply_southeast(env, 46, 4)
    print("LEVEL", env.levels_completed + 1, "ACTIONS", env.actions)
    print("GOAL", crop(env.frame(), 3, 13, 3, 13))
    for x in (23, 29, 35, 41, 47, 53, 59):
        print("SELECTOR", x, crop(env.frame(), 2, 7, x - 2, x + 3))
    summary(env, "START")
    for path in ((5,), (3,), (4,), (3, 2), (4, 2), (3, 2, 2, 4, 5),
                 (5, 6), (5, 4, 2, 2, 5)):
        c = env.clone()
        for action in path:
            if action == 6:
                c.step(6, 46, 4)
            else:
                c.step(action)
        summary(c, str(path))
    c = env.clone()
    c.step(5)
    c.step(6, 46, 4)
    c.step(5)
    summary(c, "X_N then E_N")


if __name__ == "__main__":
    harness.A.run_program("cd82", probe)
