"""Find a shortest second-delivery ordering with an earlier dismissal."""

from itertools import combinations, product

import gkm_try

from perception import arr
from probe9_verify import (
    ReachedLevel9,
    StopAtLevel9,
    boxes,
    target_state,
)


def second_pick_prefix():
    remote_pick = [2] + [4] * 6 + [1, 5]
    first_delivery = remote_pick + [3] * 6 + [1] * 3 + [3, 5]
    return first_delivery + [2] * 2 + [4] * 5 + [1, 5]


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    picked = env.clone()
    prefix = second_pick_prefix()
    for action in prefix:
        picked.step(action)

    endings = {}
    for up_positions in combinations(range(9), 3):
        up_positions = set(up_positions)
        ordering = [
            1 if index in up_positions else 3
            for index in range(9)
        ]
        route = [2] + ordering + [5]
        state = picked.clone()
        for action in route:
            state.step(action)
        target = target_state(state.frame())
        if (
            boxes(state.frame(), 14) == ((25, 24, 27, 27),)
            and target["signatures"][(5, 6)] == (3, 9)
        ):
            endings.setdefault(
                arr(state.frame()).tobytes(),
                (state, route),
            )
    print("REROUTE_ENDINGS", len(endings), flush=True)

    wins = []
    for state, route in endings.values():
        for depth in range(1, 4):
            for suffix in product(state.actions, repeat=depth):
                child = state.clone()
                for action in suffix:
                    child.step(action)
                target = target_state(child.frame())
                if (
                    (5, 6) in target["filled"]
                    and not boxes(child.frame(), 15)
                ):
                    wins.append(
                        {
                            "route": route,
                            "contact": list(suffix),
                            "turn": len(prefix) + len(route) + depth,
                            "avatar": boxes(child.frame(), 14),
                            "courier_12": boxes(child.frame(), 12),
                        }
                    )
            if wins:
                break
    print("REROUTE_WINS", wins, flush=True)


gkm_try.A.run_program("wa30", inspect)
