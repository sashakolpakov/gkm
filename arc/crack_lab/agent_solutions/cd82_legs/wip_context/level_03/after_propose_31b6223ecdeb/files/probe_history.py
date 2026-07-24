import gkm_try as harness
from legs import move_vessel_below_and_apply, apply_current_then_select_and_apply_southeast
from perception import arr, color_counts


CH = {0: "0", 2: ".", 3: "-", 4: "#", 5: " ", 8: "8", 9: "9",
      11: "B", 12: "C", 14: "E", 15: "X"}


def crop(frame, r0, r1, c0, c1):
    a = arr(frame)
    return "/".join("".join(CH.get(int(x), "?") for x in row)
                    for row in a[r0:r1, c0:c1])


def show(env, label):
    print(label, "level", env.levels_completed,
          "upper", crop(env.frame(), 3, 13, 3, 13),
          "work", crop(env.frame(), 34, 44, 27, 37),
          "colors", color_counts(env.frame()))


def probe(env):
    show(env, "L1 start")
    move_vessel_below_and_apply(env)
    show(env, "L1 end/L2 start")
    apply_current_then_select_and_apply_southeast(env, 46, 4)
    show(env, "L2 end/L3 start")


if __name__ == "__main__":
    harness.A.run_program("cd82", probe)
