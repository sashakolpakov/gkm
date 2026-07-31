import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

E = (6, 52, 42)
F = (6, 52, 46)
C = (6, 50, 28)
D = (6, 56, 28)


def avatar(env):
    ys, xs = np.where(np.asarray(env.frame()) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def run(env, label, path):
    clone = env.clone()
    out = [avatar(clone)]
    for action in path:
        try:
            clone.step(*action) if isinstance(action, tuple) else clone.step(action)
        except Exception as exc:
            out.append(type(exc).__name__)
            break
        out.append(avatar(clone))
    print(label, out, int(clone.levels_completed))


def probe(env):
    solver.solve(env)
    upper = [3, E, 3, 3]
    lower = [2, 3, E, 3, 3]
    for label, prefix in (("R17", upper), ("R18", lower)):
        run(env, label + "_F", prefix + [F, 1, 2, 3, 4])
        run(env, label + "_FE", prefix + [F, E, 1, 2, 3, 4])
        run(env, label + "_EF", prefix + [E, F, 1, 2, 3, 4])
    run(env, "ACTIVE_ENTRY", [F, 3, E, 3, 3, 1, 2])
    run(env, "START_TELEPORT_V", [F, 2, 1, 2])
    run(env, "START_TELEPORT_H", [F, 3, 4, 3])
    run(env, "START_TELEPORT_CLICK", [2, F, 1, 2])
    target = [C, C, C, D]
    run(env, "TARGET_WEST", target + [3, E, 3, 3, 3])
    run(env, "TARGET_NORTH", target + [3, E, 3, 1, 1, 1])
    run(env, "TARGET_SOUTH", target + [2, 3, E, 3, 2, 2])
    A_CTL = (6, 46, 22)
    B_CTL = (6, 56, 22)
    all_ready = [A_CTL] * 5 + [B_CTL] * 4 + target + [F]
    run(env, "ALL_WEST", all_ready + [3, E, 3, 3, 3])
    run(env, "ALL_NORTH", all_ready + [3, E, 3, 1, 1, 1])
    run(env, "PLATFORM_NEAR",
        [D, D, D, 3, E, 3, 1, 1, 3, 3, 3])
    run(env, "USE_BASE", [6])
    run(env, "USE_TARGET", target + [6])
    run(env, "USE_BRIDGE", [3, E, 3, 3, 6])
    run(env, "SYNC_E_LEFT", [3] + [item for _ in range(10) for item in (E, 3)])
    run(env, "SYNC_E_UP", [3, E, 3] + [item for _ in range(8) for item in (E, 1)])
    GAME_BRIDGE = (6, 28, 34)
    run(env, "CLICK_BRIDGE_ADJ", [3, E, GAME_BRIDGE, 3, 3, 3])
    run(env, "CLICK_BRIDGE_ON", [3, E, 3, GAME_BRIDGE, 3, 3])
    PLATFORM_GLYPH = (6, 52, 35)
    run(env, "GLYPH_BASE", [PLATFORM_GLYPH])
    run(env, "GLYPH_TARGET", target + [PLATFORM_GLYPH])
    run(env, "GLYPH_TARGET_WEST",
        target + [PLATFORM_GLYPH, 3, E, 3, 3, 3])
    LEFT_PORTAL = (6, 11, 35)
    TOP_PORTAL = (6, 25, 7)
    run(env, "ACTIVE_CLICK_LEFT", [F, LEFT_PORTAL, 1, 2, 3, 4])
    run(env, "ACTIVE_CLICK_TOP", [F, TOP_PORTAL, 1, 2, 3, 4])
    TOP_SWITCH = (6, 11, 18)
    PLATFORM_HEAD = (6, 13, 18)
    run(env, "TARGET_CLICK_SWITCH", target + [TOP_SWITCH])
    run(env, "TARGET_CLICK_HEAD", target + [PLATFORM_HEAD])
    run(env, "TARGET_CLICK_SWITCH_WEST",
        target + [TOP_SWITCH, 3, E, 3, 3, 3])
    DOCK_ACTIVE = [C, C, C, D, D, D, (6, 52, 35)]
    run(env, "DOCK_ACTIVE_WEST", DOCK_ACTIVE + [3, E, 3, 3, 3])
    run(env, "DOCK_ACTIVE_NORTH", DOCK_ACTIVE + [3, E, 3, 1, 1, 1])


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
