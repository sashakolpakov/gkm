"""Small context probes from the real level-9 dispatcher entry."""

import gkm_try

from probe9_verify import boxes, target_state


def state(env):
    target = target_state(env.frame())
    return {
        "avatar": boxes(env.frame(), 14),
        "cargo": boxes(env.frame(), 4),
        "courier_12": boxes(env.frame(), 12),
        "courier_15": boxes(env.frame(), 15),
        "empty": target["empty"],
        "filled": target["filled"],
        "level": env.levels_completed,
        "terminal": env.terminal(),
    }


def trace(base, name, actions):
    clone = base.clone()
    print("PATH", name, "START", state(clone), flush=True)
    for turn, action in enumerate(actions, 14):
        clone.step(action)
        print("STEP", name, turn, action, state(clone), flush=True)


def idle_trace(base):
    clone = base.clone()
    prior = None
    for turn in range(13, 81):
        current = state(clone)
        condensed = (
            current["empty"],
            current["filled"],
            current["courier_12"],
            current["courier_15"],
            current["level"],
            current["terminal"],
        )
        if condensed != prior:
            print("IDLE", turn, current, flush=True)
        if clone.terminal() or clone.levels_completed > base.levels_completed:
            break
        prior = condensed
        clone.step(5)


def inspect(env):
    gkm_try.resumed_solve(env)
    traces = {
        "up_left_use": [1, 3, 5],
        "left_up_use": [3, 1, 5],
        "up_left_use_down": [1, 3, 5, 2],
        "left_up_use_down": [3, 1, 5, 2],
    }
    for name, actions in traces.items():
        trace(env, name, actions)
    idle_trace(env)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
